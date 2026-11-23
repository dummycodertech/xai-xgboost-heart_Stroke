const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

const ARTIFACT_DIR = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f';
const OUT_DIR = path.join(ARTIFACT_DIR, 'step1');
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  console.log('>>> Starting Puppeteer verification for Step 1...');
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: path.join(require('os').tmpdir(), 'puppeteer_step1_' + Date.now()),
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
  const pageErrors = [];
  const failedRequests = [];

  page.on('console', (msg) => {
    const text = msg.text();
    consoleMessages.push({ type: msg.type(), text });
    if (msg.type() === 'error' || msg.type() === 'warning' || text.includes('GPU stall') || text.includes('ReadPixels')) {
      console.log(`[BROWSER ${msg.type().toUpperCase()}]:`, text);
    }
  });

  page.on('pageerror', (err) => {
    pageErrors.push(err.toString());
    console.error('[BROWSER ERROR]:', err);
  });

  page.on('requestfailed', (req) => {
    failedRequests.push({ url: req.url(), error: req.failure() ? req.failure().errorText : 'failed' });
    console.log('[REQUEST FAILED]:', req.url(), req.failure() ? req.failure().errorText : '');
  });

  page.on('response', (res) => {
    if (res.status() >= 400) {
      console.log(`[HTTP ${res.status()}]:`, res.url());
    }
  });

  // Set viewport to 1440x900
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });

  console.log('Navigating to http://localhost:3000 ...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await sleep(1500);

  // Measure FPS over 2 seconds
  console.log('Measuring FPS at 1440x900...');
  const fpsData = await page.evaluate(async () => {
    return new Promise((resolve) => {
      let frames = 0;
      let start = performance.now();
      const frameTimes = [];
      let lastTime = start;

      function onFrame(now) {
        frames++;
        const delta = now - lastTime;
        lastTime = now;
        frameTimes.push(delta);

        if (now - start >= 2000) {
          const elapsed = (now - start) / 1000;
          const fps = frames / elapsed;
          const avgDelta = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
          const maxDelta = Math.max(...frameTimes);
          resolve({ fps: Math.round(fps * 10) / 10, frames, elapsed, avgDelta, maxDelta });
        } else {
          requestAnimationFrame(onFrame);
        }
      }
      requestAnimationFrame(onFrame);
    });
  });
  console.log('Measured FPS:', fpsData);

  // Screenshot 1: Hero idle (dark monitor on 1440x900)
  console.log('Saving 1: 01_hero_idle.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '01_hero_idle.png') });

  // Switch to brain state on normal page
  console.log('Switching to brain state...');
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const brainBtn = buttons.find((b) => b.textContent && (b.textContent.includes('BRAIN') || b.textContent.includes('02')));
    if (brainBtn) {
      brainBtn.click();
    } else {
      const g = document.querySelector('[role="group"]');
      if (g) g.click();
    }
  });
  // Wait for transition to finish completely (1.5s)
  await sleep(1500);

  // Screenshot 4: Brain state
  console.log('Saving 4: 04_brain_state.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '04_brain_state.png') });

  // Now test transitions on /test page where performance.now is slowed by 1/50
  console.log('Navigating to http://localhost:3000/test (slow clock harness)...');
  await page.goto('http://localhost:3000/test', { waitUntil: 'networkidle2' });
  await sleep(1500);

  // In /test, click viewport to trigger transition
  console.log('Triggering transition in /test...');
  await page.evaluate(() => {
    const g = document.querySelector('[role="group"]');
    if (g) g.click();
  });

  // Virtual timeline:
  // Glitch runs 0 -> 110ms virtual. Midpoint is 55ms virtual = 55 * 50 = 2750ms real.
  console.log('Waiting 2700ms real for mid slice-glitch...');
  await sleep(2700);

  // Screenshot 2: Mid slice-glitch
  console.log('Saving 2: 02_mid_slice_glitch.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '02_mid_slice_glitch.png') });

  // ASCII runs 110ms -> 350ms virtual. Midpoint is ~230ms virtual = 230 * 50 = 11500ms real.
  // We already waited 2700ms, so wait another (11500 - 2700) = 8800ms real.
  console.log('Waiting 8800ms real for mid ASCII flash...');
  await sleep(8800);

  // Screenshot 3: Mid ASCII flash
  console.log('Saving 3: 03_mid_ascii_flash.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '03_mid_ascii_flash.png') });

  // Navigate back to main page for predictor and sections screenshots
  console.log('Navigating back to http://localhost:3000 ...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await sleep(1500);

  // Screenshot 5: Predictor light mode
  console.log('Scrolling to #predict for light mode predictor...');
  await page.evaluate(() => {
    document.querySelector('#predict').scrollIntoView();
  });
  await sleep(600);
  console.log('Saving 5: 05_predictor_light.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '05_predictor_light.png') });

  // Screenshot 6: Predictor dark mode
  console.log('Toggling dark theme...');
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button')).find((b) => b.textContent && (b.textContent.includes('Dark') || b.textContent.includes('Light')));
    if (btn) btn.click();
  });
  await sleep(600);
  console.log('Saving 6: 06_predictor_dark.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '06_predictor_dark.png') });

  // Screenshot 7: Model + limits section
  console.log('Scrolling to #model and #limits...');
  await page.evaluate(() => {
    document.querySelector('#model').scrollIntoView();
  });
  await sleep(600);
  console.log('Saving 7: 07_model_limits.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '07_model_limits.png') });

  // Screenshot 8: 390px mobile hero
  console.log('Resizing to 390x844 (mobile)...');
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  // Ensure light theme for mobile hero
  await page.evaluate(() => {
    delete document.documentElement.dataset.theme;
    window.scrollTo(0, 0);
  });
  await sleep(1200);
  console.log('Saving 8: 08_mobile_390px.png...');
  await page.screenshot({ path: path.join(OUT_DIR, '08_mobile_390px.png') });

  console.log('>>> Verification completed!');
  console.log('Total console logs captured:', consoleMessages.length);
  console.log('Page errors:', pageErrors.length);

  // Check if "GPU stall" appears
  const readPixelsStalls = consoleMessages.filter((m) => m.text.includes('GPU stall') || m.text.includes('ReadPixels'));
  console.log('ReadPixels stalls count:', readPixelsStalls.length);

  await browser.close();

  // Write summary report
  const summary = {
    fpsData,
    consoleMessagesCount: consoleMessages.length,
    pageErrors,
    readPixelsStallsCount: readPixelsStalls.length,
  };
  fs.writeFileSync(path.join(OUT_DIR, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log('Summary saved to:', path.join(OUT_DIR, 'summary.json'));
}

main().catch((err) => {
  console.error('FAILED:', err);
  process.exit(1);
});
