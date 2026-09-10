#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
import json
import re
from visionagent.vendor.ragflow.rag.utils.doc_store_conn import MatchTextExpr

from visionagent.vendor.ragflow.rag.nlp import rag_tokenizer, term_weight, synonym


class FulltextQueryer:
    """
    Main class for full-text search query processing and similarity calculation.
    Handles tokenization, query building, and similarity scoring for both English and Chinese text.
    """
    def __init__(self):
        self.tw = term_weight.Dealer()
        self.syn = synonym.Dealer()
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    @staticmethod
    def subSpecialChar(line):
        """
        Escape special characters in search queries to prevent Elasticsearch query syntax errors.
        Adds backslashes before special chars like :, {, }, /, [, ], -, *, ", (, ), |, +, ~, ^
        """
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()

    @staticmethod
    def isChinese(line):
        """
        Detect if input text is primarily Chinese/non-English (>=70% non-English tokens).
        Used to determine processing strategy - Chinese vs English text handling.
        Returns True if Chinese/mixed language, False if primarily English.
        """
        arr = re.split(r"[ \t]+", line)
        if len(arr) <= 3:
            return True
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        return e * 1.0 / len(arr) >= 0.7

    @staticmethod
    def rmWWW(txt):
        """
        Remove common question words and stop words from queries to improve search relevance.
        Handles both Chinese question words (什么, 哪里, 怎么样, etc.) and English (what, who, how, etc.).
        Also removes common English articles, pronouns, and auxiliary verbs.
        """
        patts = [
            (
                r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ")
        ]
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        return txt

    # user query -> es query(term weight, synonum expansion, language processing(CN/ENG), remove stop/question words)
    def question(self, txt, tbl="qa", min_match: float = 0.6):
        """
        Convert user question into Elasticsearch query with term weights and synonyms.
        Main query building method that handles both English and Chinese text differently.
        Returns MatchTextExpr for Elasticsearch and extracted keywords for highlighting.
        """
        # print('query before process:',txt)

        # ===== normalize text (lowercase, trad->sim chinese, remove punct/question/stop words) ===
        # print(f'[DEBUG] FulltextQueryer.question() txt type: {type(txt)}, value: {txt}')
        txt = re.sub(
            r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+",
            " ",
            rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
        ).strip()
        txt = FulltextQueryer.rmWWW(txt) # remove question words, stop words

        if not self.isChinese(txt): # english processing >70% english tokens
            txt = FulltextQueryer.rmWWW(txt) # Remove stop words again for English text
            tks = rag_tokenizer.tokenize(txt).split() # Split text into tokens list
            keywords = [t for t in tks if t] # Create initial keywords list (non-empty tokens)
            tks_w = self.tw.weights(tks, preprocess=False) # Assign importance weights to each token
            # token cleanup
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w] # Remove backslashes, quotes, and carets
            tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk] # Remove single alphanumeric characters
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk] # Remove leading + or - signs
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()] # Remove whitespace and filter empty tokens
            
            # find synonym and append to keywords list
            syns = []
            # tks_w is a list of tuple: [("token1", weight1), ("token2", weight2), ...]
            # print('tks_w list:', tks_w)
            for tk, w in tks_w:
                syn = self.syn.lookup(tk)
                syn = rag_tokenizer.tokenize(" ".join(syn)).split()
                keywords.extend(syn) # add to keywords
                syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
                # syns[0] looks like ['"ml"^0.3750 "ai"^0.3750 "artificial"^0.3750 "intelligence"^0.3750']
                syns.append(" ".join(syn)) # this is syn with weight (w/4)
            # print("syn list:", syns)

            # ================ build formated query for es =============
            # ================ txt_example = "what is machine learning" =============
            # create weightd term with synonyms "(term^weight synonyms)"
            q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if
                 tk and not re.match(r"[.^+\(\)-]", tk)]
            # print('q before for loop:', q)
            # q before for loop: ['(mzo^0.3333 )', '(stand^0.3333 "point"^0.0833 "of"^0.0833 "view"^0.0833 "stall"^0.0833 "base"^0.0833 "bandstand"^0.0833 "rack"^0.0833 "resist"^0.0833 "digest"^0.0833)', '(for^0.3333 )']

            for i in range(1, len(tks_w)):
                # left: token from previous tuple
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                # prevent any edge cases even you have clean it up
                if not left or not right:
                    continue
                # ("machine", 1.5) and ("learning", 1.2) -> "machine learning"^3.0000
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,
                    )
                )
            # print('q after for loop:', q)
            if not q:
                q.append(txt)
            query = " ".join(q)
            # print('final query for es:', query)

            # MatchTextExpr: object of {fields, matching_text, topn, extra_options}
            return MatchTextExpr(self.query_fields, query, 100), keywords


        # =============== for Chinese text ===========
        def need_fine_grained_tokenize(tk):
            if len(tk) < 3:
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        txt = FulltextQueryer.rmWWW(txt)
        qs, keywords = [], []
        for tt in self.tw.split(txt)[:256]:  # .split():
            if not tt:
                continue
            keywords.append(tt)
            twts = self.tw.weights([tt])
            syns = self.syn.lookup(tt)
            if syns and len(keywords) < 32:
                keywords.extend(syns)
            logging.debug(json.dumps(twts, ensure_ascii=False))
            tms = []
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                sm = (
                    rag_tokenizer.fine_grained_tokenize(tk).split()
                    if need_fine_grained_tokenize(tk)
                    else []
                )
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m,
                    )
                    for m in sm
                ]
                sm = [FulltextQueryer.subSpecialChar(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                if len(keywords) < 32:
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))
                    keywords.extend(sm)

                tk_syns = self.syn.lookup(tk)
                tk_syns = [FulltextQueryer.subSpecialChar(s) for s in tk_syns]
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s])
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]

                if len(keywords) >= 32:
                    break

                tk = FulltextQueryer.subSpecialChar(tk)
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk
                if tk_syns:
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
                if sm:
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            tms = " ".join([f"({t})^{w}" for t, w in tms])

            if len(twts) > 1:
                tms += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            syns = " OR ".join(
                [
                    '"%s"'
                    % rag_tokenizer.tokenize(FulltextQueryer.subSpecialChar(s))
                    for s in syns
                ]
            )
            if syns and tms:
                tms = f"({tms})^5 OR ({syns})^0.7"

            qs.append(tms)

        if qs:
            query = " OR ".join([f"({t})" for t in qs if t])
            return MatchTextExpr(
                self.query_fields, query, 100, {"minimum_should_match": min_match}
            ), keywords
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        """
        Calculate hybrid similarity combining vector similarity and token similarity.
        avec: query vector, bvecs: document vectors, atks: query tokens, btkss: document tokens
        Returns combined similarity scores, token similarity, and vector similarity separately.
        """
        from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
        import numpy as np

        sims = CosineSimilarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        """
        Calculate token-based similarity between query tokens and multiple document token sets.
        Uses term weights to compute weighted token overlap similarity scores.
        Returns list of similarity scores for each document.
        """
        def toDict(tks):
            d = {}
            if isinstance(tks, str):
                tks = tks.split()
            for t, c in self.tw.weights(tks, preprocess=False):
                if t not in d:
                    d[t] = 0
                d[t] += c
            return d

        atks = toDict(atks)
        btkss = [toDict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        """
        Calculate similarity between two token weight dictionaries.
        Core similarity function using weighted token overlap.
        Returns normalized similarity score between 0 and 1.
        """
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # * dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v
        return s / q

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        """
        Build search query for paragraph/content matching with important keywords.
        Used for tagging and content-based search. Extracts top weighted terms and synonyms.
        Returns MatchTextExpr suitable for Elasticsearch paragraph matching.
        """
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]
        tks_w = self.tw.weights(content_tks, preprocess=False)

        keywords = [f'"{k.strip()}"' for k in keywords]
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            tk_syns = self.syn.lookup(tk)
            tk_syns = [FulltextQueryer.subSpecialChar(s) for s in tk_syns]
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]
            tk = FulltextQueryer.subSpecialChar(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
            if tk:
                keywords.append(f"{tk}^{w}")

        return MatchTextExpr(self.query_fields, " ".join(keywords), 100,
                             {"minimum_should_match": min(3, len(keywords) / 10)})
