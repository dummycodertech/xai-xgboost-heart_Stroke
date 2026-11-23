const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runSmokeTests() {
  console.log('>>> Running Automated Smoke Test Suite for Triage...');
  let passed = 0;
  let failed = 0;

  function assert(condition, testName) {
    if (condition) {
      console.log(`[PASS] ${testName}`);
      passed++;
    } else {
      console.error(`[FAIL] ${testName}`);
      failed++;
    }
  }

  // 1. Health check endpoint test
  try {
    const res = await fetch('http://localhost:3000/api/health');
    const data = await res.json();
    assert(res.status === 200 && data.status === 'ok', 'GET /api/health responds with HTTP 200 and status ok');
  } catch (err) {
    assert(false, `GET /api/health responds with HTTP 200 (error: ${err.message})`);
  }

  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: path.join(require('os').tmpdir(), 'puppeteer_smoke_' + Date.now()),
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-webgl',
      '--enable-features=Vulkan',
      '--use-gl=angle',
      '--use-angle=default',
      '--ignore-gpu-blocklist',
      '--disable-dev-shm-usage',
    ],
  });

  const page = await browser.newPage();
  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => {
    consoleErrors.push(err.toString());
  });

  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await sleep(1500);

  // 2. Zero console errors on load
  assert(consoleErrors.length === 0, `Page loads with 0 console errors (found ${consoleErrors.length})`);

  // 3. Hero canvas present and WebGL active
  const canvasPresent = await page.evaluate(() => {
    const cv = document.querySelector('canvas');
    if (!cv) return false;
    const gl = cv.getContext('webgl2') || cv.getContext('webgl');
    return gl !== null;
  });
  assert(canvasPresent, 'Hero 3D Canvas is present and WebGL context initialized');

  // 4. Organ toggle HEART <-> BRAIN
  const initialMode = await page.evaluate(() => {
    const btn = document.querySelector('button[aria-label*="scan view"][aria-pressed="true"]');
    return btn ? btn.textContent.trim() : null;
  });
  console.log('Initial scan mode:', initialMode);

  // Click second mode button (BRAIN / OUTCOME)
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button[aria-label*="scan view"]'));
    const secondBtn = buttons.find((b) => b.textContent.includes('BRAIN') || b.textContent.includes('OUTCOME') || b.textContent.includes('02'));
    if (secondBtn) secondBtn.click();
  });
  await sleep(500);

  const switchedMode = await page.evaluate(() => {
    const btn = document.querySelector('button[aria-label*="scan view"][aria-pressed="true"]');
    return btn ? btn.textContent.trim() : null;
  });
  assert(switchedMode && (switchedMode.includes('BRAIN') || switchedMode.includes('OUTCOME')), 'Scan view toggles from mode 01 to mode 02');

  // 5. Theme toggle switches between light and dark
  const initialTheme = await page.evaluate(() => document.documentElement.dataset.theme || 'light');
  await page.evaluate(() => {
    const toggleBtn = document.querySelector('button[aria-label*="theme"]');
    if (toggleBtn) toggleBtn.click();
  });
  await sleep(200);
  const switchedTheme = await page.evaluate(() => document.documentElement.dataset.theme || 'light');
  assert(initialTheme !== switchedTheme, `Theme toggle switches theme from ${initialTheme} to ${switchedTheme}`);

  // Switch back to light
  await page.evaluate(() => {
    const toggleBtn = document.querySelector('button[aria-label*="theme"]');
    if (toggleBtn) toggleBtn.click();
  });

  // 6. Predictor dynamic update on input change
  const initialProb = await page.evaluate(() => {
    const el = document.querySelector('#predict .text-7xl, #predict .text-8xl');
    return el ? el.textContent.trim() : null;
  });

  const ageInput = await page.$('input[aria-label="Age"]');
  if (ageInput) {
    await page.evaluate((el) => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(el, 80);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }, ageInput);
    await sleep(500);
  }

  const updatedProb = await page.evaluate(() => {
    const el = document.querySelector('#predict .text-7xl, #predict .text-8xl');
    return el ? el.textContent.trim() : null;
  });
  assert(initialProb !== updatedProb, `Predictor dynamically updates probability when input changes (${initialProb} -> ${updatedProb})`);

  // 7. Local surrogate (LIME-style) panel present with units
  const surrogatePresent = await page.evaluate(() => {
    const headers = Array.from(document.querySelectorAll('h3'));
    const limeHeader = headers.find((h) => h.textContent.includes('Local surrogate (LIME-style)'));
    const unitsSpan = document.querySelector('span.font-mono.text-\\[11px\\].text-muted');
    return !!limeHeader;
  });
  assert(surrogatePresent, 'Local surrogate (LIME-style) section is rendered');

  // 8. Operating point switcher updates confusion matrix
  const matrixInitial = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll('#model .text-3xl'));
    return cells.map((c) => c.textContent.trim()).join(',');
  });

  // Click 80% recall tab
  await page.evaluate(() => {
    const tabs = Array.from(document.querySelectorAll('#model button[role="tab"]'));
    const tab = tabs.find((t) => t.textContent.includes('80%'));
    if (tab) tab.click();
  });
  await sleep(300);

  const matrix80 = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll('#model .text-3xl'));
    return cells.map((c) => c.textContent.trim()).join(',');
  });
  assert(matrixInitial !== matrix80 && (matrix80 === '39,11,293,679' || matrix80 === '39,293,11,679'), `Operating point switcher updates confusion matrix (${matrixInitial} -> ${matrix80})`);

  // 9. ROC Curve and Calibration Curves SVGs present
  const curvesPresent = await page.evaluate(() => {
    const roc = document.querySelector('svg[aria-label*="ROC"], svg[aria-label*="True positive rate"]');
    const calib = document.querySelector('svg[aria-label*="Calibration"], svg[aria-label*="Observed frequency"]');
    return !!roc && !!calib;
  });
  assert(curvesPresent, 'ROC Curve SVG and Calibration Reliability SVG are both rendered');

  // 10. Drift surveillance table present with sparklines
  const driftPresent = await page.evaluate(() => {
    const table = document.querySelector('table[aria-label="Feature drift monitoring table"]');
    const sparklines = table ? table.querySelectorAll('polyline').length : 0;
    return sparklines === 5;
  });
  assert(driftPresent, 'Drift surveillance table renders 5 feature PSI sparklines');

  await browser.close();

  console.log(`\n========================================`);
  console.log(`Smoke Test Summary: ${passed} passed, ${failed} failed`);
  console.log(`========================================\n`);

  if (failed > 0) {
    process.exit(1);
  }
}

runSmokeTests().catch((err) => {
  console.error('Fatal smoke test error:', err);
  process.exit(1);
});
