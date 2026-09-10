#!/usr/bin/env node
/**
 * The journey, through a real browser.
 *
 *   node test/journey/journey.ui.cjs        (playwright resolves from frontend/)
 *   make test-journey                       (this plus the backend half)
 *
 * The backend half (test_journey_e2e.py) proves the API returns grounded
 * answers. This proves a person sees them: the account is created through the
 * form, the document is uploaded through the picker, the questions are typed
 * into the composer, and every assertion reads rendered text.
 *
 * The browser half verifies markdown rendering, citation resolution and the
 * visible progress trace; the backend half verifies grounding at the API.
 *
 * Env: E2E_WEB_URL, E2E_API_URL
 * Exits non-zero if any case fails.
 */
const path = require('node:path')
const fs = require('node:fs')
const { execFileSync } = require('node:child_process')

const REPO = path.resolve(__dirname, '../..')
const FIXTURE = path.join(__dirname, 'fixture')

let chromium
try {
  ;({ chromium } = require(path.join(REPO, 'frontend', 'node_modules', 'playwright')))
} catch {
  console.error('playwright is not installed. cd frontend && npm i -D playwright')
  process.exit(2)
}

const WEB = process.env.E2E_WEB_URL || 'http://localhost:5181'
const PASS = 'PytestPass123!'
const USER = `journeyui_${Math.random().toString(36).slice(2, 10)}`
const SHOTS = path.join(__dirname, 'screenshots')
const PARSE_WAIT = Number(process.env.E2E_PARSE_WAIT || 900000)
const ANSWER_WAIT = Number(process.env.E2E_ANSWER_WAIT || 300000)

// `☒ searching user knowledge base (1.23s)` -- what the trace renders.
const DURATION = /\(\d+\.\d{2}s\)/

const results = []
const consoleErrors = []
const failedRequests = []

function rec(name, ok, detail = '') {
  results.push({ name, ok, detail })
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  |  ${detail}` : ''}`)
}

async function step(name, fn) {
  try {
    const detail = await fn()
    rec(name, true, detail || '')
    return true
  } catch (err) {
    rec(name, false, String(err.message).split('\n')[0].slice(0, 200))
    return false
  }
}

/**
 * The document the journey ingests, and the questions that interrogate it.
 *
 * The pinned Apache-2.0 RAGFlow benchmark PDF. Kept in step with conftest.py's
 * sample_doc fixture -- both halves ask the same questions.
 */
function sampleDoc() {
  const pdf = path.join(
    __dirname,
    '..',
    'fixtures',
    'ragflow',
    'Doc1.pdf',
  )
  if (!fs.existsSync(pdf)) {
    console.error(`missing ${pdf}`)
    process.exit(2)
  }
  return {
    pdf: 'Doc1.pdf',
    path: pdf,
    signature: 'Purpose of RAGFlow',
    questions: [
      {
        question: 'What is RAGFlow designed to turn raw documents into?',
        answer_should_contain: ['reliable', 'context'],
        fact: 'turn raw documents into reliable context',
      },
      {
        question: 'At question time, what does RAGFlow retrieve and send to the model?',
        answer_should_contain: ['passages', 'context'],
        fact: 'retrieves the most relevant passages and sends them to the model as context',
      },
      {
        question: 'What does RAGFlow say retrieval context reduces, and what does source-linked answering improve?',
        answer_should_contain: ['hallucinations', 'traceability'],
        fact: 'reduces hallucinations and improves traceability',
      },
    ],
  }
}

async function registerThroughTheForm(page) {
  await page.goto(`${WEB}/register`, { waitUntil: 'networkidle' })
  await page.waitForSelector('input', { timeout: 20000 })
  const inputs = await page.$$('input')
  await inputs[0].fill(USER)
  await inputs[1].fill(PASS)
  // Some forms carry a confirm field; fill any remaining password input.
  for (const extra of inputs.slice(2)) {
    const type = await extra.getAttribute('type')
    if (type === 'password') await extra.fill(PASS)
  }
  const [res] = await Promise.all([
    page.waitForResponse((r) => r.url().includes('/register'), { timeout: 60000 }),
    page.click('button[type="submit"]'),
  ])
  if (res.status() !== 200) throw new Error(`register HTTP ${res.status()}`)
  return `created ${USER}`
}

async function login(page) {
  await page.goto(`${WEB}/login`, { waitUntil: 'networkidle' })
  await page.waitForSelector('input', { timeout: 20000 })
  const inputs = await page.$$('input')
  await inputs[0].fill(USER)
  await inputs[1].fill(PASS)
  const [res] = await Promise.all([
    page.waitForResponse((r) => r.url().includes('/login'), { timeout: 60000 }),
    page.click('button[type="submit"]'),
  ])
  if (res.status() !== 200) throw new Error(`login HTTP ${res.status()}`)
  await page.waitForTimeout(2500)
  return `HTTP ${res.status()}`
}

async function isToggleOn(btn) {
  const cls = await btn.evaluate((el) => (el.closest('button') || el).className)
  return /ant-btn-variant-filled|ant-btn-color-primary/.test(cls)
}

async function setToggle(page, label, want) {
  const btn = await page.$(`button:has-text("${label}")`)
  if (!btn) throw new Error(`toggle "${label}" not found`)
  if ((await isToggleOn(btn)) !== want) {
    await btn.click()
    await page.waitForTimeout(300)
  }
}

async function uploadAndWaitForParse(page, pdfPath) {
  // The repository page is the document-ingestion UI. The chat composer also
  // accepts a file, but that path adds *session context* and never opens the
  // progress stream, so it is a different feature and cannot show a parse.
  await page.goto(`${WEB}/repository`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(1500)

  const add = await page.$('button:has-text("Add Files")')
  if (!add) throw new Error('no "Add Files" button on /repository')

  // No dialog: "Add Files" wraps the picker, so choosing a file starts the
  // batch and the file becomes a row in the table straight away.
  const input = await page.$('input[type="file"]')
  if (!input) throw new Error('no file input on /repository')

  const [res] = await Promise.all([
    page.waitForResponse((r) => /start-processing/.test(r.url()), { timeout: 120000 }),
    input.setInputFiles(pdfPath),
  ])
  if (res.status() !== 200) throw new Error(`start-processing -> ${res.status()}`)

  // Watched in place, without navigating.
  //
  // Not the "added" toast: antd dismisses it after a few seconds, so polling
  // for it races the toast and waits out the timeout even when the parse
  // succeeded.
  //
  // The row carries Upload -> Parse -> Encode -> Database and ticks each as it
  // passes; when the parse finishes the row is replaced by the ordinary
  // listing, so the progress strip disappearing is the completion signal. Class
  // names are CSS-module hashed, so match on the substring.
  const PROGRESS = '[class*="row-progress"]'
  const TICKS = '[class*="row-step"][class*="completed"]'
  const deadline = Date.now() + PARSE_WAIT
  let maxTicks = 0
  while (Date.now() < deadline) {
    const body = (await page.textContent('body')) || ''
    if (/Upload failed|Processing failed|Failed —/i.test(body)) {
      throw new Error('the row reported a processing failure')
    }
    maxTicks = Math.max(maxTicks, (await page.$$(TICKS)).length)
    if (maxTicks >= 4) return 'the row ticked all four steps'

    const stillRunning = await page.$(PROGRESS)
    if (!stillRunning && maxTicks > 0) {
      return `row finished after ${maxTicks} step(s) ticked`
    }
    await page.waitForTimeout(1500)
  }
  throw new Error(
    `parse did not finish within ${PARSE_WAIT}ms (row reached ${maxTicks}/4)`,
  )
}

async function documentIsListed(page, name) {
  // Re-navigating refetches the list; the table does not poll on its own.
  await page.goto(`${WEB}/repository`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(2000)
  const body = (await page.textContent('body')) || ''
  return body.includes(name.replace(/\.[^.]+$/, ''))
}

async function ask(page, question) {
  await page.goto(`${WEB}/`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(1200)
  await setToggle(page, 'Web Search', false)

  const box = await page.$('textarea, [contenteditable="true"]')
  if (!box) throw new Error('composer not found')
  await box.click()
  await box.fill(question)

  const [res] = await Promise.all([
    page.waitForResponse((r) => /ai_search/.test(r.url()), { timeout: ANSWER_WAIT }),
    page.keyboard.press('Enter'),
  ])
  if (res.status() !== 200) throw new Error(`ai_search -> ${res.status()}`)

  // Wait for the *stream* to finish, not for the answer text to appear.
  //
  // The citations frame is the last one the backend sends, and it comes after
  // the answer has fully streamed and after the image and video lookups, which
  // are live network calls taking seconds. Asserting a moment after "Answer"
  // shows up therefore races the stream: the answer is on screen while the
  // Source list is still empty, and the assertion fails on a system that is
  // working correctly. Settle on the citations instead, and only give up on
  // them once the answer has been still for a while.
  const deadline = Date.now() + ANSWER_WAIT
  let sawAnswer = false
  let lastLength = 0
  let stableSince = Date.now()
  while (Date.now() < deadline) {
    const body = (await page.textContent('body')) || ''
    if (/Request failed with status code/i.test(body)) {
      throw new Error('UI rendered "Request failed with status code"')
    }
    if (/Answer/.test(body)) sawAnswer = true

    if ((await page.$$('.citation-icon')).length) return body

    if (body.length !== lastLength) {
      lastLength = body.length
      stableSince = Date.now()
    } else if (sawAnswer && Date.now() - stableSince > 25000) {
      // Nothing has changed for 25s after the answer: the stream is over and
      // no citation arrived. That is a real failure, so return and let the
      // assertions report it.
      return body
    }
    await page.waitForTimeout(1000)
  }
  throw new Error(`no Answer rendered within ${ANSWER_WAIT}ms`)
}

;(async () => {
  fs.mkdirSync(SHOTS, { recursive: true })
  const doc = sampleDoc()

  const browser = await chromium.launch()
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.on('console', (m) => {
    if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 180))
  })
  page.on('pageerror', (e) => consoleErrors.push(`PAGEERROR ${e.message.slice(0, 180)}`))
  page.on('requestfailed', (r) =>
    failedRequests.push(`${r.method()} ${r.url().slice(0, 80)} :: ${r.failure()?.errorText}`),
  )

  console.log(`\n=== CLEAN ACCOUNT (${USER}) ===`)
  await step('register through the form', () => registerThroughTheForm(page))
  await step('log in', () => login(page))
  await step('the new account shows no documents', async () => {
    if (await documentIsListed(page, doc.pdf)) {
      throw new Error('a fresh account already has the document')
    }
    return 'empty'
  })

  console.log('\n=== PARSE ===')
  const parsed = await step('upload the sample document and wait for parsing', () =>
    uploadAndWaitForParse(page, doc.path))
  await page.screenshot({ path: path.join(SHOTS, 'journey-parsed.png'), fullPage: true })
  await step('the parsed document is listed in the repository', async () => {
    if (!(await documentIsListed(page, doc.pdf))) {
      throw new Error(`${doc.pdf} is not listed after parsing`)
    }
    return doc.pdf
  })

  console.log('\n=== THREE QUESTIONS ===')
  let lastBody = ''
  for (const q of doc.questions) {
    if (!parsed) break
    await step(`ask: ${q.question.slice(0, 46)}`, async () => {
      lastBody = await ask(page, q.question)
      return 'answered'
    })
    await step(`the answer shows ${q.fact}`, () => {
      for (const token of q.answer_should_contain) {
        if (!lastBody.toLowerCase().includes(token.toLowerCase())) {
          throw new Error(`rendered answer never showed ${token}`)
        }
      }
      return q.fact
    })
    await step('the answer renders a clickable citation', async () => {
      const icons = await page.$$('.citation-icon')
      if (!icons.length) {
        const raw = /\[doc\]\[cite_[^\]]+\]/.test(lastBody)
        throw new Error(
          raw
            ? 'citation markers rendered as literal [doc][cite_...] text'
            : 'no citation was rendered at all',
        )
      }
      const id = await icons[0].getAttribute('data-citation-id')
      if (!id) throw new Error('citation icon carries no data-citation-id')
      return `${icons.length} citation(s), first=${id.slice(0, 34)}`
    })
    await step('the Source list carries the document as a source', async () => {
      // Read the Source entries rather than matching words in the answer.
      const entries = await page.$$('[class*="citation_item"]')
      if (!entries.length) throw new Error('the Source list rendered no entries')
      const texts = await Promise.all(entries.map((e) => e.textContent()))
      // And they must name *this document*: an answer with no retrieval still
      // produces citations, so counting entries proves nothing on its own. The
      // entry renders citation.title || docnm_kwd -- the document name, not the
      // passage, so the signature inside the text will never appear here.
      const stem = doc.pdf.replace(/\.[^.]+$/, '')
      if (!texts.some((t) => (t || '').includes(stem))) {
        throw new Error(
          `${entries.length} source(s) rendered but none names ${stem}: ` +
            texts.join(' | ').slice(0, 120),
        )
      }
      return `${entries.length} source(s), one naming ${stem}`
    })
    await step('the progress trace shows step durations', () => {
      const found = lastBody.match(DURATION)
      if (!found) {
        throw new Error('no (N.NNs) timing rendered anywhere on the page')
      }
      return `e.g. ${found[0]}`
    })
    await step('the progress trace names the knowledge-base step', () => {
      if (!/searching user knowledge base/i.test(lastBody)) {
        throw new Error('the retrieval step was never shown to the user')
      }
      return 'shown'
    })
  }
  await page.screenshot({ path: path.join(SHOTS, 'journey-answer.png'), fullPage: true })

  console.log('\n=== DIAGNOSTICS ===')
  console.log(`  console errors  : ${consoleErrors.length}`)
  consoleErrors.slice(0, 5).forEach((e) => console.log(`    ${e}`))
  console.log(`  failed requests : ${failedRequests.length}`)
  failedRequests.slice(0, 5).forEach((e) => console.log(`    ${e}`))

  await browser.close()

  const passed = results.filter((r) => r.ok).length
  console.log(`\n===== ${passed}/${results.length} passed =====`)
  console.log(`screenshots: ${SHOTS}`)
  process.exit(passed === results.length ? 0 : 1)
})().catch((err) => {
  console.error(err)
  process.exit(1)
})
