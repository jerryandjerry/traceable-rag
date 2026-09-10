#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import copy
import re

from visionagent.vendor.ragflow.api.db import ParserType
from io import BytesIO
from visionagent.vendor.ragflow.rag.nlp import rag_tokenizer, tokenize, tokenize_table, bullets_category, title_frequency, tokenize_chunks, docx_question_level
from visionagent.vendor.ragflow.rag.utils import num_tokens_from_string
from visionagent.vendor.ragflow.deepdoc.parser import PdfParser, PlainParser, DocxParser
from docx import Document
from PIL import Image


class Pdf(PdfParser):
    def __init__(self):
        self.model_speciess = ParserType.MANUAL.value
        super().__init__()

    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):
        from timeit import default_timer as timer

        # =================== OCR process =======================
        start = timer()
        callback(msg="OCR started")
        self.__images__(
            filename if not binary else binary,
            zoomin, # Zoom factor for processing
            from_page, # start of the page to process
            to_page, # end of the page to process
            callback # Progress callback function
        )
        callback(msg="OCR finished ({:.2f}s)".format(timer() - start))
        # for bb in self.boxes:
        #    for b in bb:
        #        print(b)
        logging.debug("OCR: {}".format(timer() - start))

        # =================== layout recognition =======================
        start = timer()
        self._layouts_rec(zoomin)
        callback(0.65, "Layout analysis ({:.2f}s)".format(timer() - start))
        logging.debug("layouts: {}".format(timer() - start))
        
        # =================== layout recognition =======================
        start = timer()
        self._table_transformer_job(zoomin)
        callback(0.67, "Table analysis ({:.2f}s)".format(timer() - start))

        # =================== final process =======================
        start = timer()
        self._text_merge() # merge text horizontally
        tbls = self._extract_table_figure(True, zoomin, True, True) # extract tables and figures
        self._concat_downward() # merge text vertically
        self._filter_forpages()
        callback(0.68, "Text merged ({:.2f}s)".format(timer() - start))

        # clean mess
        for b in self.boxes:
            b["text"] = re.sub(r"([\t 　]|\u3000){2,}", " ", b["text"].strip())

        return [(b["text"], b.get("layoutno", ""), self.get_position(b, zoomin))
                for i, b in enumerate(self.boxes)], tbls


class Docx(DocxParser):
    def __init__(self):
        pass

    def get_picture(self, document, paragraph):
        img = paragraph._element.xpath('.//pic:pic')
        if not img:
            return None
        img = img[0]
        embed = img.xpath('.//a:blip/@r:embed')[0]
        related_part = document.part.related_parts[embed]
        image = related_part.image
        image = Image.open(BytesIO(image.blob))
        return image

    def concat_img(self, img1, img2):
        if img1 and not img2:
            return img1
        if not img1 and img2:
            return img2
        if not img1 and not img2:
            return None
        width1, height1 = img1.size
        width2, height2 = img2.size

        new_width = max(width1, width2)
        new_height = height1 + height2
        new_image = Image.new('RGB', (new_width, new_height))

        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (0, height1))

        return new_image

    def __call__(self, filename, binary=None, from_page=0, to_page=100000, callback=None):
        self.doc = Document(
            filename) if not binary else Document(BytesIO(binary))
        pn = 0
        last_answer, last_image = "", None
        question_stack, level_stack = [], []
        ti_list = []
        for p in self.doc.paragraphs:
            if pn > to_page:
                break
            question_level, p_text = 0, ''
            if from_page <= pn < to_page and p.text.strip():
                question_level, p_text = docx_question_level(p)
            if not question_level or question_level > 6: # not a question
                last_answer = f'{last_answer}\n{p_text}'
                current_image = self.get_picture(self.doc, p)
                last_image = self.concat_img(last_image, current_image)
            else:   # is a question
                if last_answer or last_image:
                    sum_question = '\n'.join(question_stack)
                    if sum_question:
                        ti_list.append((f'{sum_question}\n{last_answer}', last_image))
                    last_answer, last_image = '', None

                i = question_level
                while question_stack and i <= level_stack[-1]:
                    question_stack.pop()
                    level_stack.pop()
                question_stack.append(p_text)
                level_stack.append(question_level)
            for run in p.runs:
                if 'lastRenderedPageBreak' in run._element.xml:
                    pn += 1
                    continue
                if 'w:br' in run._element.xml and 'type="page"' in run._element.xml:
                    pn += 1
        if last_answer:
            sum_question = '\n'.join(question_stack)
            if sum_question:
                ti_list.append((f'{sum_question}\n{last_answer}', last_image))
                
        tbls = []
        for tb in self.doc.tables:
            html= "<table>"
            for r in tb.rows:
                html += "<tr>"
                i = 0
                while i < len(r.cells):
                    span = 1
                    c = r.cells[i]
                    for j in range(i+1, len(r.cells)):
                        if c.text == r.cells[j].text:
                            span += 1
                            i = j
                    i += 1
                    html += f"<td>{c.text}</td>" if span == 1 else f"<td colspan='{span}'>{c.text}</td>"
                html += "</tr>"
            html += "</table>"
            tbls.append(((None, html), ""))
        return ti_list, tbls


def chunk(file_name, file_path, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Only pdf is supported.
    """
    pdf_parser = None
    doc = {
        "docnm": file_name
    }
    doc["docnm_tks"] = rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", doc["docnm"]))
    doc["docnm_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["docnm_tks"])
    # is it English
    eng = lang.lower() == "english"  # pdf_parser.is_english

    # for pdf file
    if re.search(r"\.pdf$", file_path, re.IGNORECASE):

        # ==================== pdf_parser(deepdoc) PDF Processing ==============================
        pdf_parser = Pdf()
        if kwargs.get("layout_recognize", "DeepDOC") == "Plain Text":
            pdf_parser = PlainParser()
        
        # sections[i] -> ('content with weights', content type, [(page_num, left, right, top, bottom)])
        # tbls[i] -> ((<PIL.Image.Image image mode=RGB size=1583x859 at 0x732E15741690>, ['captions']), [(page_num, left, right, top, bottom)])
        sections, tbls = pdf_parser(file_path if not binary else binary, from_page=from_page, to_page=to_page, callback=callback)
        # if simple parser is used, there is no position info
        if sections and len(sections[0]) < 3:
            sections = [(t, lvl, [[0] * 5]) for t, lvl in sections]
        # print('dtype sections from pdf_parser:', type(sections))
        # print('[section[0] from pdf_parser]:', sections[0])
        # print('dtype tbls from pdf_parser:', type(tbls))
        # print('[tbls[0] from pdf_parser]:', tbls[0])

        # ==================== title matching by pdf outlines or predefined re template ==============================
        if len(sections) > 0 and len(pdf_parser.outlines) / len(sections) > 0.03:
            max_lvl = max([lvl for _, lvl in pdf_parser.outlines])
            most_level = max(0, max_lvl - 1)
            levels = []
            for txt, _, _ in sections:
                for t, lvl in pdf_parser.outlines: # check if the title match pdf.outlines, if match, set it as level and add it to levels
                    tks = set([t[i] + t[i + 1] for i in range(len(t) - 1)])
                    tks_ = set([txt[i] + txt[i + 1] for i in range(min(len(t), len(txt) - 1))])
                    if len(set(tks & tks_)) / max([len(tks), len(tks_), 1]) > 0.8:
                        levels.append(lvl)
                        break
                else: # if not match, set it as mox_lvl +1 as content level
                    levels.append(max_lvl + 1)
        else:
            bull = bullets_category([txt for txt, _, _ in sections])
            most_level, levels = title_frequency(
                bull, [(txt, lvl) for txt, lvl, _ in sections])

        # ==================== use level as pivot to generate grouping mask(id) ==============================
        assert len(sections) == len(levels)
        sec_ids = []
        sid = 0
        for i, lvl in enumerate(levels):
            if lvl <= most_level and i > 0 and lvl != levels[i - 1]:
                sid += 1
            sec_ids.append(sid)
            # print(lvl, self.boxes[i]["text"], most_level, sid)

        # ==================== add tbls(title only) into sections, assign group id to each section ==============================
        # create section as tuple ( txt, group id, poss)
        sections = [(txt, sec_ids[i], poss) for i, (txt, _, poss) in enumerate(sections)]
        for (img, rows), poss in tbls:
            if not rows:
                continue
            # then append (tbls_dict, -1, poss)
            tbls_dict = {'image': img, 'caption': rows if isinstance(rows, str) else rows[0]}
            sections.append((tbls_dict, -1,
                            [(p[0] + 1 - from_page, p[1], p[2], p[3], p[4]) for p in poss]))
        # print('[sections dtype]: \n', type(sections))
        # print('[one item in sections]: \n', sections[0])
        # sections[i] -> ('content with weights', group_id, [(page_num, left, right, top, bottom)])
        
        # ==================== merge sections into chunk based on group id and token limit ==============================
        # crate temporary markers for use in final processing, will be removed in final result
        def tag(pn, left, right, top, bottom):
            if pn + left + right + top + bottom == 0:
                return ""
            return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##" \
                .format(pn, left, right, top, bottom)

        chunks = []
        chunk_images = []  # Store tbls images for each chunk
        last_sid = -2
        tk_cnt = 0
        # sort section by x[-1][0][0]  = x[poss list][poss][page num], 
        # then x[-1][0][3]  = x[poss list][poss][top], finally x[-1][0][1]  = x[poss list][poss][left]
        for txt, sec_id, poss in sorted(sections, key=lambda x: (x[-1][0][0], x[-1][0][3], x[-1][0][1])):
            # Handle tbls dict format
            if isinstance(txt, dict):  # tbls section
                txt_content = txt['caption']
                txt_image = txt['image']
            else:  # regular text section
                txt_content = txt
                txt_image = None
                
            poss = "\t".join([tag(*pos) for pos in poss])
            if tk_cnt < 32 or (tk_cnt < 1024 and (sec_id == last_sid or sec_id == -1)):
                if chunks:
                    chunks[-1] += "\n" + txt_content + poss
                    tk_cnt += num_tokens_from_string(txt_content)
                    # Store tbls image if present
                    if txt_image and sec_id == -1:  # tbls section
                        chunk_images[-1].append(txt_image)
                    continue
            chunks.append(txt_content + poss)
            # Store tbls image if present
            if txt_image and sec_id == -1:  # tbls section
                chunk_images.append([txt_image])
            else:
                chunk_images.append([])
            tk_cnt = num_tokens_from_string(txt_content)
            if sec_id > -1:
                last_sid = sec_id

        # ==================== process tbls and sections, and merge them into final result ==============================
        res = tokenize_table(tbls, doc, eng)
        res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser, chunk_images))

        # print('[dtype of res]: \n', type(res))
        # print('[one item from res]: \n', res[-6])
        # print('[one item from res]: \n', res[-7])
        # print('[one item from res]: \n', res[3])
        # for i, chunk in enumerate(res):
        #     chunk['image'].save(f'test_image/chunk_{i}_image.png')
        return res

        """
        this is what res[i] looks like:    
        {
       'docnm': '960c-Performance-Standards-for-Mid-Rise-Buildings.pdf', 
       'title_tks': 'document 960c perform standard for mid rise build', 
       'title_sm_tks': 'document 960c perform standard for mid rise build', 
       'content_with_weight': 'captions', 
       'content_ltks': 'caption tokens', 
       'content_sm_ltks': 'caption fine-grain tokens', 
       'image': <PIL.Image.Image image mode=RGB size=1583x859 at 0x732E15741690>, 
       'page_num_int': [3], 
       'position_int': [(3, 50, 578, 442, 728)], 
       'top_int': [442]
       }

        """


    elif re.search(r"\.docx?$", file_path, re.IGNORECASE):
        docx_parser = Docx()
        ti_list, tbls = docx_parser(file_path, binary,
                                    from_page=0, to_page=10000, callback=callback)
        res = tokenize_table(tbls, doc, eng)
        for text, image in ti_list:
            d = copy.deepcopy(doc)
            d['image'] = image
            tokenize(d, text, eng)
            res.append(d)
        return res
    else:
        raise NotImplementedError("file type not supported yet(pdf and docx supported)")
    

if __name__ == "__main__":
    import sys
    import time

    def progress_callback(prog=None, msg=""):
        elapsed = time.time() - start_time
        if prog is not None:
            print(f"[{elapsed:6.2f}s] {prog*100:5.1f}% - {msg}")
        elif msg:
            print(f"[{elapsed:6.2f}s] {msg}")

    start_time = time.time()
    chunk("documents/Performance-Standards-for-Mid-Rise-Buildings.pdf", callback=progress_callback)

# PYTHONPATH=backend/app python backend/app/service/core/rag/app/manual.py
