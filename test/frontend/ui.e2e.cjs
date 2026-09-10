#!/usr/bin/env node
/**
 * End-to-end UI tests. Drives real Chromium against the running stack.
 *
 *   node test/frontend/ui.e2e.cjs   (from frontend/, so playwright resolves)
 *
 * Requires: frontend on :5181, backend on :8000, Postgres + Elasticsearch up.
 *
 * The .cjs extension preserves CommonJS under frontend/package.json's
 * "type": "module" boundary.
 * Env: E2E_WEB_URL; E2E_USER and E2E_PASS are required. The prefix avoids
 * collisions with the shell's USER variable.
 *
 * Exits non-zero if any case fails.
 */
const path = require("node:path");
const fs = require("node:fs");

const REPO = path.resolve(__dirname, "../..");

// Node resolves modules from the SCRIPT's directory, not the cwd, and this
// script lives outside frontend/ so that all e2e sits under test/. Point it at
// the frontend's node_modules explicitly rather than requiring a second copy
// of playwright at the repo root.
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
const missingCredentials = ["E2E_USER", "E2E_PASS"].filter(
  (name) => !process.env[name]?.trim(),
);
if (missingCredentials.length) {
  console.error(
    `missing required environment variables: ${missingCredentials.join(", ")}`,
  );
  process.exit(2);
}
const USER = process.env.E2E_USER.trim();
const PASS = process.env.E2E_PASS;
const SHOTS = path.join(__dirname, "screenshots");
const ANSWER_WAIT = Number(process.env.E2E_ANSWER_WAIT || 120000);

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
    rec(name, false, String(err.message).split("\n")[0].slice(0, 160));
    return false;
  }
}

function samplePdf() {
  const fixture = path.join(
    REPO,
    "test",
    "fixtures",
    "ragflow",
    "Doc1.pdf",
  );
  return fs.existsSync(fixture) ? fixture : null;
}

/**
 * antd marks a toggled Button with color-primary + variant-filled; the
 * untoggled state is color-default + variant-outlined. There is no
 * "active"/"selected" class, so matching on those never detected the state.
 */
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
  const got = await isToggleOn(btn);
  if (got !== want) throw new Error(`could not set "${label}" to ${want}`);
}

/** Poll for the rendered answer rather than sleeping a fixed interval. */
async function waitForAnswer(page, timeout = ANSWER_WAIT) {
  const deadline = Date.now() + timeout;
  let text = "";
  while (Date.now() < deadline) {
    text = (await page.textContent("body")) || "";
    if (/Request failed with status code/i.test(text)) {
      throw new Error('UI rendered "Request failed with status code"');
    }
    // The answer block and its action row only appear once streaming ends.
    if (/Answer/.test(text)) return text;
    await page.waitForTimeout(1000);
  }
  throw new Error(`no Answer section rendered within ${timeout}ms`);
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
  await page.waitForTimeout(2500);
  return `HTTP ${res.status()}`;
}

async function ask(page, question, { web }) {
  await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1200);
  await setToggle(page, "Web Search", web);

  const box = await page.$('textarea, [contenteditable="true"]');
  if (!box) throw new Error("composer not found");
  await box.click();
  await box.fill(question);

  const [res] = await Promise.all([
    page.waitForResponse((r) => /ai_search|deep_research/.test(r.url()), {
      timeout: 150000,
    }),
    page.keyboard.press("Enter"),
  ]);

  const url = res.url().replace(WEB, "");
  if (/deep_research/.test(url)) {
    throw new Error(
      `posted to /deep_research/, which does not exist (${res.status()})`,
    );
  }
  if (res.status() !== 200) throw new Error(`${url} -> ${res.status()}`);

  await waitForAnswer(page);
  return `${url} -> ${res.status()}`;
}

(async () => {
  fs.mkdirSync(SHOTS, { recursive: true });
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

  console.log("\n=== AUTH ===");
  await step("login", () => login(page));
  await step("unauthenticated visit redirects to /login", async () => {
    const ctx = await browser.newContext();
    const p2 = await ctx.newPage();
    await p2.goto(`${WEB}/`, { waitUntil: "networkidle" });
    await p2.waitForTimeout(1500);
    const url = p2.url();
    await ctx.close();
    if (!/login/.test(url)) throw new Error(`landed on ${url}`);
    return url;
  });

  console.log("\n=== CHAT: AVAILABLE TOGGLE COMBINATIONS ===");
  await step("Deep Search control is hidden", async () => {
    await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
    const button = await page.$('button:has-text("Deep Search")');
    if (button) throw new Error("Deep Search control is visible");
    return "future workflow remains unavailable";
  });
  await step("web=off", () =>
    ask(page, "What is 2+2? Answer with one number.", { web: false }),
  );
  await step("web=on", () =>
    ask(page, "What is the capital of France? One word.", { web: true }),
  );
  await page.screenshot({ path: path.join(SHOTS, "chat.png"), fullPage: true });

  console.log("\n=== FILE UPLOAD ===");
  await step(
    "Add context remains enabled while Deep Search is hidden",
    async () => {
      await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
      await page.waitForTimeout(1000);
      const input = await page.$('input[type="file"]');
      const disabled = await input.evaluate((el) => el.disabled);
      if (disabled)
        throw new Error("hidden Deep Search state disabled file input");
      return "enabled";
    },
  );

  await step("upload a PDF sends a request", async () => {
    const pdf = samplePdf();
    if (!pdf) throw new Error("required RAGFlow PDF fixture is missing");
    await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);
    const input = await page.$('input[type="file"]');
    // upload() requires an array; a bare File fails before sending a request.
    const [res] = await Promise.all([
      page.waitForResponse(
        (r) => /add_context|start-processing/.test(r.url()),
        {
          timeout: 300000,
        },
      ),
      input.setInputFiles(pdf),
    ]);
    if (res.status() !== 200) throw new Error(`upload -> ${res.status()}`);
    return `${res.url().replace(WEB, "")} -> ${res.status()}`;
  });

  console.log("\n=== ROUTES ===");
  for (const route of ["/", "/graph", "/repository", "/login", "/register"]) {
    await step(`route ${route}`, async () => {
      const res = await page.goto(`${WEB}${route}`, {
        waitUntil: "networkidle",
        timeout: 60000,
      });
      await page.waitForFunction(
        () => document.querySelector("#root")?.children.length > 0,
        { timeout: 20000 },
      );
      const text = ((await page.textContent("body")) || "").trim();
      if (!res.ok()) throw new Error(`HTTP ${res.status()}`);
      if (!text) throw new Error("rendered an empty body");
      return `HTTP ${res.status()}, ${text.length} chars`;
    });
  }

  console.log("\n=== GRAPH VIEWER ===");
  await step("graph renders nodes and links", async () => {
    await page.goto(`${WEB}/graph`, { waitUntil: "networkidle" });
    await page.waitForTimeout(8000);
    const text = (await page.textContent("body")) || "";
    const m = text.match(/Nodes:\s*(\d+)[\s\S]*?Links:\s*(\d+)/);
    await page.screenshot({
      path: path.join(SHOTS, "graph.png"),
      fullPage: true,
    });
    if (!m)
      throw new Error('no "Nodes:/Links:" readout — viewer did not initialise');
    return `${m[1]} nodes, ${m[2]} links`;
  });

  console.log("\n=== DIAGNOSTICS ===");
  const uniq = (a) => [...new Set(a)];
  console.log(`  HTTP >=400 : ${uniq(httpErrors).length}`);
  uniq(httpErrors)
    .slice(0, 10)
    .forEach((e) => console.log(`    ${e}`));
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
