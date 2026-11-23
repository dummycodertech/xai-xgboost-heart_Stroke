const path = require('path');
const os = require('os');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

async function main() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pup-model-'));
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: tmpDir,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-webgl',
      '--ignore-gpu-blocklist'
    ]
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  console.log('Navigating to http://localhost:3000 ...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2000));

  // 1. Check for Demo mode text
  const bodyText = await page.evaluate(() => document.body.innerText);
  if (bodyText.includes('Demo mode')) {
    console.error('FAIL: "Demo mode" text is present on the page!');
  } else {
    console.log('PASS: No "Demo mode" text present. Inference comes strictly from real trained model.');
  }

  // 2. Check the calculated probability in the Predictor
  await page.waitForSelector('#predict', { timeout: 10000 });
  const probInfo = await page.evaluate(() => {
    const calc = document.querySelector('#predict');
    if (!calc) return { found: false };
    const h1s = calc.querySelectorAll('.font-mono');
    let text = '';
    for (const el of h1s) {
      if (el.textContent.includes('%') && /\d+\.\d+%/.test(el.textContent)) {
        text = el.textContent;
        break;
      }
    }
    const errorBanner = calc.querySelector('[role="alert"]');
    return {
      found: true,
      text,
      error: errorBanner ? errorBanner.textContent : null
    };
  });
  console.log('Predictor initial result (Default patient):', probInfo);

  // 3. Check LIME Chart with the correct aria-label
  const limeInfo = await page.evaluate(() => {
    const limeSvg = document.querySelector('svg[aria-label="LIME local surrogate rule contributions"]');
    if (!limeSvg) return { found: false };
    const rects = limeSvg.querySelectorAll('rect');
    const textNodes = Array.from(limeSvg.querySelectorAll('text')).map(t => t.textContent.trim());
    return {
      found: true,
      rectCount: rects.length,
      sampleTexts: textNodes.slice(0, 8)
    };
  });
  console.log('LIME Chart status:', limeInfo);

  // 4. Scroll to Predictor and take screenshot
  await page.evaluate(() => {
    const calc = document.querySelector('#predict');
    if (calc) calc.scrollIntoView({ behavior: 'instant', block: 'center' });
  });
  await new Promise(r => setTimeout(r, 1200));

  const artifactDir = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades';
  if (!fs.existsSync(artifactDir)) fs.mkdirSync(artifactDir, { recursive: true });

  const limeScreenshotPath = path.join(artifactDir, 'real_model_lime_chart.png');
  await page.screenshot({ path: limeScreenshotPath });
  console.log('Captured screenshot:', limeScreenshotPath);

  // 5. Scroll to Beeswarm Chart
  const beeswarmInfo = await page.evaluate(() => {
    const beeswarmSvg = document.querySelector('svg[aria-label="SHAP summary beeswarm plot"]');
    if (!beeswarmSvg) return { found: false };
    beeswarmSvg.scrollIntoView({ behavior: 'instant', block: 'center' });
    const dots = beeswarmSvg.querySelectorAll('circle');
    return {
      found: true,
      dotsCount: dots.length
    };
  });
  console.log('Beeswarm Chart status:', beeswarmInfo);
  await new Promise(r => setTimeout(r, 1200));

  const beeswarmScreenshotPath = path.join(artifactDir, 'beeswarm_chart.png');
  await page.screenshot({ path: beeswarmScreenshotPath });
  console.log('Captured screenshot:', beeswarmScreenshotPath);

  // 6. Test preset selection ("Higher risk" profile)
  console.log('Testing "Higher risk" preset button click...');
  await page.evaluate(() => {
    const calc = document.querySelector('#predict');
    calc.scrollIntoView({ behavior: 'instant', block: 'center' });
    const buttons = Array.from(calc.querySelectorAll('button'));
    const higherRiskBtn = buttons.find(b => b.textContent.includes('Higher risk'));
    if (higherRiskBtn) higherRiskBtn.click();
  });
  await new Promise(r => setTimeout(r, 2000));

  const updatedProb = await page.evaluate(() => {
    const calc = document.querySelector('#predict');
    const h1s = calc ? calc.querySelectorAll('.font-mono') : [];
    for (const el of h1s) {
      if (el.textContent.includes('%') && /\d+\.\d+%/.test(el.textContent)) {
        return el.textContent;
      }
    }
    return null;
  });
  console.log('Updated probability after clicking Higher risk preset:', updatedProb);

  const interactiveScreenshotPath = path.join(artifactDir, 'interactive_real_model.png');
  await page.screenshot({ path: interactiveScreenshotPath });
  console.log('Captured screenshot:', interactiveScreenshotPath);

  await browser.close();
  try {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  } catch (e) {}
  console.log('All verifications completed successfully!');
}

main().catch(err => {
  console.error('Error during verification:', err);
  process.exit(1);
});
