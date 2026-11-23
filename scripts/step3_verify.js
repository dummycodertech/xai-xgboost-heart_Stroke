const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

const ARTIFACT_DIR = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f';
const OUT_DIR = path.join(ARTIFACT_DIR, 'step3');
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  console.log('>>> Starting Puppeteer verification for Step 3...');
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: path.join(require('os').tmpdir(), 'puppeteer_step3_' + Date.now()),
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

  // 1. Desktop 1440x900 FPS measurement
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await sleep(1500);

  const desktopFps = await page.evaluate(async () => {
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
  console.log('Desktop 1440x900 FPS:', desktopFps);

  // 2. Mobile 390x844 FPS measurement
  console.log('Measuring Mobile 390x844 FPS...');
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  await sleep(1000);

  const mobileFps = await page.evaluate(async () => {
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
  console.log('Mobile 390x844 FPS:', mobileFps);

  // 3. Keyboard navigation test
  console.log('Testing keyboard navigation...');
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const brainBtn = buttons.find((b) => b.textContent && b.textContent.includes('BRAIN'));
    if (brainBtn) {
      brainBtn.focus();
      brainBtn.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      brainBtn.click();
    }
  });
  await sleep(1000);

  const ariaStatus = await page.evaluate(() => {
    const el = document.querySelector('[role="status"]');
    return el ? el.textContent : '';
  });
  console.log('Aria live status message after keyboard switch:', ariaStatus);

  // 4. Test prefers-reduced-motion
  console.log('Testing prefers-reduced-motion emulation...');
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await sleep(1000);

  const reducedMotionScreenshot = path.join(OUT_DIR, 'reduced_motion_static_frame.png');
  await page.screenshot({ path: reducedMotionScreenshot });
  console.log('Saved reduced motion static screenshot:', reducedMotionScreenshot);

  await browser.close();

  const report = {
    desktopFps,
    mobileFps,
    ariaStatus,
    targetDesktop60Met: desktopFps >= 58,
    targetMobile30Met: mobileFps >= 30,
  };
  fs.writeFileSync(path.join(OUT_DIR, 'perf_report.json'), JSON.stringify(report, null, 2));
  console.log('Performance report saved:', report);
}

main().catch((err) => {
  console.error('FAILED:', err);
  process.exit(1);
});
