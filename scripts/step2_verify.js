const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

const ARTIFACT_DIR = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f';
const OUT_DIR = path.join(ARTIFACT_DIR, 'step2');
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  console.log('>>> Starting Puppeteer verification for Step 2...');
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: path.join(require('os').tmpdir(), 'puppeteer_step2_' + Date.now()),
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
  const consoleMessages = [];

  page.on('console', (msg) => {
    consoleMessages.push(msg.text());
    if (msg.type() === 'error' || msg.text().includes('GPU stall')) {
      console.log(`[BROWSER ${msg.type().toUpperCase()}]:`, msg.text());
    }
  });

  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  console.log('Navigating to http://localhost:3000 ...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await sleep(2500);

  // Measure FPS at 1440x900
  console.log('Measuring FPS at 1440x900 with upgraded shaders & GLBs...');
  const fpsData = await page.evaluate(async () => {
    return new Promise((resolve) => {
      let frames = 0;
      let start = performance.now();
      function onFrame(now) {
        frames++;
        if (now - start >= 2000) {
          resolve(Math.round((frames / ((now - start) / 1000)) * 10) / 10);
        } else {
          requestAnimationFrame(onFrame);
        }
      }
      requestAnimationFrame(onFrame);
    });
  });
  console.log('Measured FPS:', fpsData);

  // Read telemetry text in the HUD
  const hudTextHeart = await page.evaluate(() => {
    const el = document.querySelector('[role="group"]');
    return el ? el.innerText : '';
  });
  console.log('Heart HUD status:\n', hudTextHeart);

  // Save Step 2 heart screenshot
  console.log('Saving step2_heart_hero.png...');
  await page.screenshot({ path: path.join(OUT_DIR, 'step2_heart_hero.png') });

  // Switch to brain mode
  console.log('Switching to brain mode...');
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const brainBtn = buttons.find((b) => b.textContent && (b.textContent.includes('BRAIN') || b.textContent.includes('02')));
    if (brainBtn) brainBtn.click();
    else document.querySelector('[role="group"]').click();
  });
  await sleep(2000);

  const hudTextBrain = await page.evaluate(() => {
    const el = document.querySelector('[role="group"]');
    return el ? el.innerText : '';
  });
  console.log('Brain HUD status:\n', hudTextBrain);

  console.log('Saving step2_brain_hero.png...');
  await page.screenshot({ path: path.join(OUT_DIR, 'step2_brain_hero.png') });

  await browser.close();
  console.log('Puppeteer test finished!');
}

main().catch((err) => {
  console.error('FAILED:', err);
  process.exit(1);
});
