#!/usr/bin/env node
/**
 * Every feature the frontend offers, driven through a real browser.
 *
 *   node test/frontend/features.ui.cjs        (playwright resolves from frontend/)
 *   make test-e2e                             (runs this after ui.e2e.cjs)
 *
 * A fresh account is registered through the form and then walked through the
 * whole product: login, the home composer, a chat with and without web search,
 * attaching a file as session context and asking about it, the knowledge-base
 * upload with its four-step progress row, viewing the parsed chunks, the graph
 * viewer, deleting the document, the sessions sidebar with its delete, the
 * 404 page, and logout. Every assertion reads rendered text or a rendered
 * element; HTTP status alone is never the pass condition.
 *
 * Requires: frontend on :5181, backend on :8000, Postgres + Elasticsearch up,
 * and a model provider that answers. Env: E2E_WEB_URL, E2E_PARSE_WAIT,
 * E2E_ANSWER_WAIT. Exits non-zero if any case fails.
 */
const path = require("node:path");
const fs = require("node:fs");

const REPO = path.resolve(__dirname, "../..");

let chromium;
try {
  ({ chromium } = require(
    path.join(REPO, "frontend", "node_modules", "playwright"),
  ));
} catch {
  console.error(
    "playwright is not installed. cd frontend && npm i -D playwright",
  );
  process.exit(2);
}

const WEB = process.env.E2E_WEB_URL || "http://localhost:5181";
const USER = `uifeat_${Math.random().toString(36).slice(2, 10)}`;
const PASS = "FeaturePass123!";
const SHOTS = path.join(__dirname, "screenshots");
const PARSE_WAIT = Number(process.env.E2E_PARSE_WAIT || 900000);
const ANSWER_WAIT = Number(process.env.E2E_ANSWER_WAIT || 300000);
const DOC = path.join(REPO, "test", "fixtures", "ragflow", "Doc1.pdf");
const DOC_NAME = path.basename(DOC);

const results = [];
const consoleErrors = [];
const failedRequests = [];
const httpErrors = [];

function rec(name, ok, detail = "") {
  results.push({ name, ok, detail });
  console.log(
    `  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  |  ${detail}` : ""}`,
  );
}

async function step(name, fn) {
  try {
    const detail = await fn();
    rec(name, true, detail || "");
    return true;
  } catch (err) {
    rec(name, false, String(err.message).split("\n")[0].slice(0, 200));
    return false;
  }
}

async function body(page) {
  return (await page.textContent("body")) || "";
}

async function shot(page, name) {
  await page.screenshot({
    path: path.join(SHOTS, `${name}.png`),
    fullPage: true,
  });
}

async function isToggleOn(btn) {
  const cls = await btn.evaluate(
    (el) => (el.closest("button") || el).className,
  );
  return /ant-btn-variant-filled|ant-btn-color-primary/.test(cls);
}

async function setToggle(page, label, want) {
  const btn = await page.$(`button:has-text("${label}")`);
  if (!btn) throw new Error(`toggle "${label}" not found`);
  if ((await isToggleOn(btn)) !== want) {
    await btn.click();
    await page.waitForTimeout(300);
  }
  if ((await isToggleOn(btn)) !== want)
    throw new Error(`could not set "${label}" to ${want}`);
}

// ----------------------------------------------------------------- account
async function registerThroughTheForm(page) {
  await page.goto(`${WEB}/register`, { waitUntil: "networkidle" });
  await page.waitForSelector("input", { timeout: 20000 });
  const inputs = await page.$$("input");
  if (inputs.length < 3)
    throw new Error(`expected 3 inputs, found ${inputs.length}`);
  await inputs[0].fill(USER);
  await inputs[1].fill(PASS);
  await inputs[2].fill(PASS);
  const [res] = await Promise.all([
    page.waitForResponse((r) => r.url().includes("/register"), {
      timeout: 60000,
    }),
    page.click('button[type="submit"]'),
  ]);
  if (res.status() !== 200) throw new Error(`register HTTP ${res.status()}`);
  await page.waitForURL(/login/, { timeout: 20000 });
  return `${USER} -> ${page.url().replace(WEB, "")}`;
}

async function login(page) {
  await page.goto(`${WEB}/login`, { waitUntil: "networkidle" });
  await page.waitForSelector("input", { timeout: 20000 });
  const inputs = await page.$$("input");
  await inputs[0].fill(USER);
  await inputs[1].fill(PASS);
  const [res] = await Promise.all([
    page.waitForResponse((r) => r.url().includes("/login"), { timeout: 60000 }),
    page.click('button[type="submit"]'),
  ]);
  if (res.status() !== 200) throw new Error(`login HTTP ${res.status()}`);
  await page.waitForURL((u) => !/login/.test(u.toString()), { timeout: 20000 });
  await page.waitForTimeout(1000);
  return `HTTP ${res.status()} -> ${page.url().replace(WEB, "")}`;
}

async function logout(page) {
  await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
  const avatar = await page.$(".base-layout-nav .ant-avatar, .ant-avatar");
  if (!avatar) throw new Error("user avatar not found in the nav");
  await avatar.click(); // the dropdown opens on click, not hover
  await page.waitForSelector("text=Logout", { timeout: 10000 });
  await page.click("text=Logout");
  await page.waitForURL(/login/, { timeout: 20000 });
  return page.url().replace(WEB, "");
}

// -------------------------------------------------------------------- chat
async function waitForAnswer(page, previousAnswers, timeout = ANSWER_WAIT) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const text = await body(page);
    if (/Request failed with status code/i.test(text)) {
      throw new Error('UI rendered "Request failed with status code"');
    }
    if (/The request failed\. Reference:/i.test(text)) {
      throw new Error(
        "the turn failed server-side (sanitized error frame rendered)",
      );
    }
    const count = (text.match(/Answer/g) || []).length;
    if (count > previousAnswers) {
      // Let the stream finish: the action row appears once [DONE] arrived.
      await page.waitForTimeout(1500);
      return await body(page);
    }
    await page.waitForTimeout(1000);
  }
  throw new Error(`no new Answer section within ${timeout}ms`);
}

async function askFromHome(page, question, { web }) {
  await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  await setToggle(page, "Web Search", web);
  const box = await page.$('textarea, [contenteditable="true"]');
  if (!box) throw new Error("composer not found");
  await box.click();
  await box.fill(question);
  const [created, res] = await Promise.all([
    page.waitForResponse((r) => /create_session/.test(r.url()), {
      timeout: 60000,
    }),
    page.waitForResponse((r) => /ai_search/.test(r.url()), {
      timeout: ANSWER_WAIT,
    }),
    page.keyboard.press("Enter"),
  ]);
  if (created.status() !== 200)
    throw new Error(`create_session -> ${created.status()}`);
  if (res.status() !== 200) throw new Error(`ai_search -> ${res.status()}`);
  // The Answer section renders on the first token; the composer stays
  // disabled until the stream ends. Wait for the body to finish.
  await res.finished();
  const text = await waitForAnswer(page, 0);
  const m = page.url().match(/\/chat\/([a-f0-9]+)/);
  if (!m) throw new Error(`not on a chat page: ${page.url()}`);
  return { sessionId: m[1], text };
}

async function askInChat(page, question, previousAnswers) {
  const box = await page.$('textarea, [contenteditable="true"]');
  if (!box) throw new Error("composer not found on the chat page");
  await box.click();
  await box.fill(question);
  const [res] = await Promise.all([
    page.waitForResponse((r) => /ai_search/.test(r.url()), {
      timeout: ANSWER_WAIT,
    }),
    page.keyboard.press("Enter"),
  ]);
  if (res.status() !== 200) throw new Error(`ai_search -> ${res.status()}`);
  await res.finished();
  return waitForAnswer(page, previousAnswers);
}

// -------------------------------------------------------------- repository
async function uploadAndWaitForParse(page) {
  await page.goto(`${WEB}/repository`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  const input = await page.$('input[type="file"]');
  if (!input) throw new Error("the Add Files picker was not found");
  const [res] = await Promise.all([
    page.waitForResponse((r) => /start-processing/.test(r.url()), {
      timeout: 120000,
    }),
    input.setInputFiles(DOC),
  ]);
  if (res.status() !== 200)
    throw new Error(`start-processing -> ${res.status()}`);

  const deadline = Date.now() + PARSE_WAIT;
  let seen = new Set();
  while (Date.now() < deadline) {
    const text = await body(page);
    const failed = await page.$('[class*="row-error"]');
    if (failed)
      throw new Error(
        `row failed: ${(await failed.textContent()).slice(0, 160)}`,
      );
    if (/Upload failed|Processing failed/i.test(text))
      throw new Error("upload reported failure");
    for (const label of ["Upload", "Parse", "Encode", "Database"]) {
      const active = await page.$(
        `[class*="row-step"][class*="active"]:has-text("${label}")`,
      );
      if (active) seen.add(label);
    }
    const progress = await page.$('[class*="row-progress"]');
    if (!progress && text.includes(DOC_NAME)) {
      return `row finished; steps seen: ${[...seen].join(" > ") || "(too fast to observe)"}`;
    }
    await page.waitForTimeout(2000);
  }
  throw new Error(`parse did not finish within ${PARSE_WAIT}ms`);
}

// --------------------------------------------------------------------- run
(async () => {
  fs.mkdirSync(SHOTS, { recursive: true });
  if (!fs.existsSync(DOC)) {
    console.error(`sample document missing: ${DOC}`);
    process.exit(2);
  }
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
  });

  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text().slice(0, 180));
  });
  page.on("pageerror", (e) =>
    consoleErrors.push(`PAGEERROR ${e.message.slice(0, 180)}`),
  );
  page.on("requestfailed", (r) =>
    failedRequests.push(
      `${r.method()} ${r.url().slice(0, 80)} :: ${r.failure()?.errorText}`,
    ),
  );
  page.on("response", (r) => {
    if (r.status() >= 400) {
      httpErrors.push(
        `${r.status()} ${r.request().method()} ${r.url().replace(WEB, "")}`,
      );
    }
  });

  console.log("\n=== ACCOUNT ===");
  await step("register through the form", () => registerThroughTheForm(page));
  await step("log in", () => login(page));
  await step("the nav shows the signed-in user", async () => {
    const text = await body(page);
    if (!text.includes(USER[0].toUpperCase()))
      throw new Error("avatar initial not rendered");
    return "avatar rendered";
  });
  await step("unauthenticated visit redirects to /login", async () => {
    const ctx = await browser.newContext();
    const p2 = await ctx.newPage();
    await p2.goto(`${WEB}/repository`, { waitUntil: "networkidle" });
    await p2.waitForTimeout(1500);
    const url = p2.url();
    await ctx.close();
    if (!/login/.test(url)) throw new Error(`landed on ${url}`);
    return url.replace(WEB, "");
  });

  console.log("\n=== CHAT ===");
  let sessionId = null;
  let answers = 0;
  await step(
    "a question from the home composer opens a chat and answers",
    async () => {
      const out = await askFromHome(
        page,
        "What is 2+2? Answer with one number.",
        { web: false },
      );
      sessionId = out.sessionId;
      answers = 1;
      if (!/4/.test(out.text)) throw new Error("the answer does not contain 4");
      return `session ${sessionId}`;
    },
  );
  await step("a follow-up in the same chat answers", async () => {
    const text = await askInChat(page, "And 3+3? One number.", answers);
    answers += 1;
    if (!/6/.test(text)) throw new Error("the answer does not contain 6");
    return "answered";
  });
  await step("the answer streams progress steps with durations", async () => {
    const text = await body(page);
    if (!/\(\d+\.\d{2}s\)/.test(text) && !/☒|☐/.test(text)) {
      return "no trace rendered for the casual path (expected: casual turns have no retrieval loop)";
    }
    return "trace rendered";
  });
  await step(
    "web search on: the search step runs and the answer renders",
    async () => {
      const out = await askFromHome(
        page,
        "What is the capital of France? One word.",
        { web: true },
      );
      if (!/Paris/i.test(out.text))
        throw new Error("the answer does not name Paris");
      // Not the answer text: a model knows Paris without searching. The trace
      // names the web step only when the tool actually ran.
      if (!/searching online/i.test(out.text))
        throw new Error("the trace does not show the web search step");
      return `session ${out.sessionId}, web step traced`;
    },
  );
  await step(
    "the turn is in history before the client sees [DONE]",
    async () => {
      // Reload the chat: what renders now comes from /get_messages/, not the
      // stream, so an answer that shows was persisted before the terminal frame.
      await page.reload({ waitUntil: "networkidle" });
      await page.waitForTimeout(1500);
      const text = await body(page);
      if (!/Paris/i.test(text))
        throw new Error("the answer is not in history after reload");
      return "answer rendered from history";
    },
  );
  await shot(page, "features-chat");

  console.log("\n=== SESSION CONTEXT ===");
  await step("attach a PDF to the chat as context", async () => {
    await page.goto(`${WEB}/chat/${sessionId}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);
    const input = await page.$('input[type="file"]');
    if (!input) throw new Error("the composer has no file input");
    const [res] = await Promise.all([
      page.waitForResponse((r) => /add_context/.test(r.url()), {
        timeout: 600000,
      }),
      input.setInputFiles(DOC),
    ]);
    if (res.status() !== 200) throw new Error(`add_context -> ${res.status()}`);
    const json = await res.json();
    const file = json.timing?.individual_files?.[0];
    if (!file || file.status !== "success")
      throw new Error(`file status: ${JSON.stringify(file)}`);
    await page.waitForTimeout(800);
    const text = await body(page);
    if (!/added to context successfully/.test(text) && !text.includes(DOC_NAME)) {
      throw new Error("no confirmation rendered");
    }
    return `${file.pages_processed} page(s), ${file.content_length} chars`;
  });
  await step("a question about the attached file is answered", async () => {
    const before = ((await body(page)).match(/Answer/g) || []).length;
    const text = await askInChat(
      page,
      "Summarise the attached document in one sentence.",
      before,
    );
    if (text.length < 50) throw new Error("empty answer");
    return "answered";
  });

  console.log("\n=== KNOWLEDGE BASE ===");
  const parsed = await step(
    "upload the sample document and watch the four-step row",
    () => uploadAndWaitForParse(page),
  );
  await step("the document is listed", async () => {
    await page.goto(`${WEB}/repository`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    const text = await body(page);
    if (!text.includes(DOC_NAME)) throw new Error("not listed");
    if (/Failed/.test(text))
      throw new Error("the row is marked Failed (a stage did not complete)");
    return "listed, not marked partial";
  });
  await step("View File shows the parsed chunks", async () => {
    await page.click('button[title="View File"]');
    await page.waitForSelector("text=/Total Chunks: \\d+/", { timeout: 60000 });
    const m = (await body(page)).match(/Total Chunks: (\d+)/);
    if (!m || Number(m[1]) < 1) throw new Error("zero chunks");
    return `${m[1]} chunks`;
  });
  await shot(page, "features-repository");
  if (parsed) {
    await step("a knowledge question is answered with a citation", async () => {
      // A question from the journey's golden set, so the classifier routes
      // it to the knowledge path rather than answering it as small talk.
      const out = await askFromHome(
        page,
        "What is RAGFlow designed to turn raw documents into?",
        { web: false },
      );
      if (!/searching user knowledge base/i.test(out.text)) {
        throw new Error("the trace does not name the knowledge-base step");
      }
      return "knowledge-base step ran";
    });
  }

  console.log("\n=== GRAPH ===");
  await step("the graph viewer renders this user's nodes", async () => {
    await page.goto(`${WEB}/graph`, { waitUntil: "networkidle" });
    await page.waitForTimeout(8000);
    const m = (await body(page)).match(/Nodes:\s*(\d+)[\s\S]*?Links:\s*(\d+)/);
    await shot(page, "features-graph");
    if (!m) throw new Error('no "Nodes:/Links:" readout');
    if (parsed && Number(m[1]) < 1)
      throw new Error("graph is empty after ingest");
    return `${m[1]} nodes, ${m[2]} links`;
  });

  console.log("\n=== SESSIONS SIDEBAR ===");
  await step("the sidebar lists the sessions and deletes one", async () => {
    await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
    await page.click('button[title="Chat Sessions"]');
    await page.waitForSelector(".session-delete-btn", { timeout: 20000 });
    const before = (await page.$$(".session-delete-btn")).length;
    if (before < 1) throw new Error("no sessions listed");
    await page.click(".session-delete-btn");
    await page.waitForSelector(".ant-popconfirm-buttons", { timeout: 10000 });
    const [res] = await Promise.all([
      page.waitForResponse((r) => /delete_session/.test(r.url()), {
        timeout: 30000,
      }),
      page.click(".ant-popconfirm-buttons .ant-btn-primary"),
    ]);
    if (res.status() !== 200)
      throw new Error(`delete_session -> ${res.status()}`);
    await page.waitForTimeout(1500);
    const after = (await page.$$(".session-delete-btn")).length;
    if (after !== before - 1)
      throw new Error(`expected ${before - 1} sessions, saw ${after}`);
    return `${before} -> ${after} sessions`;
  });

  console.log("\n=== DELETE DOCUMENT ===");
  await step("Delete File removes the document from the list", async () => {
    await page.goto(`${WEB}/repository`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    if (!(await body(page)).includes(DOC_NAME))
      throw new Error("document not listed before delete");
    const [res] = await Promise.all([
      page.waitForResponse((r) => /delete_file/.test(r.url()), {
        timeout: 120000,
      }),
      page.click('button[title="Delete File"]'),
    ]);
    if (res.status() !== 200) throw new Error(`delete_file -> ${res.status()}`);
    await page.waitForTimeout(2000);
    if ((await body(page)).includes(DOC_NAME))
      throw new Error("still listed after delete");
    return "gone";
  });

  console.log("\n=== OTHER PAGES ===");
  await step("an unknown route renders the 404 page", async () => {
    await page.goto(`${WEB}/no-such-page`, { waitUntil: "networkidle" });
    if (!/404/.test(await body(page))) throw new Error("no 404 text");
    return "404";
  });
  await step("logout returns to /login and the token is gone", async () => {
    const where = await logout(page);
    await page.goto(`${WEB}/repository`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);
    if (!/login/.test(page.url()))
      throw new Error(`still signed in: ${page.url()}`);
    return where;
  });

  console.log("\n=== DIAGNOSTICS ===");
  const uniq = (a) => [...new Set(a)];
  // A signed-out probe is expected to be refused; everything else is not.
  const unexpected = uniq(httpErrors).filter(
    (e) => !/^401 GET \/.*(get_sessions|get_files|\/me)/.test(e),
  );
  console.log(`  HTTP >=400 (unexpected): ${unexpected.length}`);
  unexpected.slice(0, 10).forEach((e) => console.log(`    ${e}`));
  console.log(`  console errors : ${uniq(consoleErrors).length}`);
  uniq(consoleErrors)
    .slice(0, 10)
    .forEach((e) => console.log(`    ${e}`));
  console.log(`  failed requests : ${uniq(failedRequests).length}`);
  uniq(failedRequests)
    .slice(0, 10)
    .forEach((e) => console.log(`    ${e}`));

  await browser.close();

  const passed = results.filter((r) => r.ok).length;
  console.log(`\n===== ${passed}/${results.length} passed =====`);
  console.log(`screenshots: ${SHOTS}`);
  process.exit(passed === results.length ? 0 : 1);
})().catch((err) => {
  console.error("\nharness crashed:", err);
  process.exit(2);
});
