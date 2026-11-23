const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

const ARTIFACT_DIR = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f';
const OUT_DIR = path.join(ARTIFACT_DIR, 'step4');
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  console.log('>>> Starting Puppeteer verification for Step 4 (Product Completeness & Evidence)...');

  // Verify /api/health endpoint
  try {
    const res = await fetch('http://localhost:3000/api/health');
    const healthJson = await res.json();
    console.log('/api/health response:', healthJson);
    fs.writeFileSync(path.join(OUT_DIR, 'health_response.json'), JSON.stringify(healthJson, null, 2));
  } catch (err) {
    console.error('Failed to fetch /api/health:', err);
  }

  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: path.join(require('os').tmpdir(), 'puppeteer_step4_' + Date.now()),
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

  // 1. Verify Predictor and Local Surrogate (LIME-style)
  console.log('Verifying Predictor & Local Surrogate...');
  const predictorEl = await page.$('#predict');
  if (predictorEl) {
    // Scroll predictor into view
    await predictorEl.scrollIntoView();
    await sleep(500);

    // Change age input to 74 to trigger dynamic update
    const ageInput = await page.$('input[aria-label="Age"]');
    if (ageInput) {
      await page.evaluate((el) => {
        el.value = '74';
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }, ageInput);
      await sleep(400); // allow debounce
    }

    await predictorEl.screenshot({ path: path.join(OUT_DIR, 'step4_predictor_surrogate.png') });
    console.log('Saved step4_predictor_surrogate.png');
  }

  // 2. Verify Evidence Section & Operating Points Switcher
  console.log('Verifying Operating Points & Confusion Matrix...');
  const modelSection = await page.$('#model');
  if (modelSection) {
    await modelSection.scrollIntoView();
    await sleep(500);

    // Initial state screenshot (Cost-optimal)
    await page.screenshot({ path: path.join(OUT_DIR, 'step4_evidence_confusion_matrix.png') });
    console.log('Saved step4_evidence_confusion_matrix.png');

    // Test clicking the "Target 80% OOF Recall" tab
    const tabs = await page.$$('button[role="tab"]');
    console.log(`Found ${tabs.length} operating point tabs`);
    for (const tab of tabs) {
      const text = await page.evaluate((el) => el.textContent, tab);
      if (text.includes('80%')) {
        console.log('Clicking 80% recall tab:', text);
        await tab.click();
        await sleep(300);
        break;
      }
    }

    // Capture screenshot with 80% recall selected
    await page.screenshot({ path: path.join(OUT_DIR, 'step4_evidence_80pct_recall.png') });

    // Test clicking the "High precision" tab
    for (const tab of tabs) {
      const text = await page.evaluate((el) => el.textContent, tab);
      if (text.includes('High precision')) {
        console.log('Clicking High precision tab:', text);
        await tab.click();
        await sleep(300);
        break;
      }
    }
    await page.screenshot({ path: path.join(OUT_DIR, 'step4_evidence_high_precision.png') });
  }

  // 3. Capture ROC & Calibration Curves
  console.log('Capturing ROC and Calibration curves...');
  const curvesEl = await page.evaluate(() => {
    const el = document.querySelector('svg[aria-label*="ROC curve"]');
    return el ? true : false;
  });
  console.log('ROC curve SVG found:', curvesEl);

  // 4. Capture Drift Surveillance Table
  console.log('Capturing Drift Surveillance table...');
  const driftTable = await page.$('table[aria-label="Feature drift monitoring table"]');
  if (driftTable) {
    await driftTable.scrollIntoView();
    await sleep(300);
    await page.screenshot({ path: path.join(OUT_DIR, 'step4_drift_sparklines.png') });
    console.log('Saved step4_drift_sparklines.png');
  }

  // 5. Capture Full Evidence View
  if (modelSection) {
    await modelSection.scrollIntoView();
    await sleep(500);
    await page.screenshot({ path: path.join(OUT_DIR, 'step4_full_evidence.png') });
    console.log('Saved step4_full_evidence.png');
  }

  console.log('Console errors during run:', consoleErrors.length);
  if (consoleErrors.length > 0) {
    console.error('Console error details:', consoleErrors);
  }

  const results = {
    timestamp: new Date().toISOString(),
    consoleErrorsCount: consoleErrors.length,
    consoleErrors,
    rocCurvePresent: curvesEl,
  };
  fs.writeFileSync(path.join(OUT_DIR, 'step4_results.json'), JSON.stringify(results, null, 2));

  await browser.close();
  console.log('>>> Step 4 Puppeteer verification complete.');
}

main().catch((err) => {
  console.error('Fatal error in step4_verify:', err);
  process.exit(1);
});
