const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

(async () => {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webgl', '--ignore-gpu-blocklist']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise((r) => setTimeout(r, 1000));

  // 1. Check initial gradient transforms
  const g1 = await page.evaluate(() => {
    const topG = document.querySelector('#flow-top');
    const waveG = document.querySelector('#flow-wave');
    return {
      top: topG ? topG.getAttribute('gradientTransform') : null,
      wave: waveG ? waveG.getAttribute('gradientTransform') : null,
    };
  });
  console.log('Initial gradient transforms (t=0):', g1);

  // Wait 400ms for gradient animation loop to advance
  await new Promise((r) => setTimeout(r, 400));

  const g2 = await page.evaluate(() => {
    const topG = document.querySelector('#flow-top');
    const waveG = document.querySelector('#flow-wave');
    return {
      top: topG ? topG.getAttribute('gradientTransform') : null,
      wave: waveG ? waveG.getAttribute('gradientTransform') : null,
    };
  });
  console.log('Updated gradient transforms (t=400ms):', g2);

  const topFlowing = g1.top !== g2.top;
  const waveFlowing = g1.wave !== g2.wave;
  console.log('Top contour colors flowing:', topFlowing);
  console.log('Wave colors flowing:', waveFlowing);

  // Capture Dark mode with flowing colors
  await page.screenshot({ path: 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/flowing_colors_dark.png' });
  console.log('Saved flowing_colors_dark.png');

  // Toggle to Light mode
  await page.evaluate(() => {
    const btn = document.querySelector('button[aria-label*="theme"]');
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 500));

  // Capture Light mode with flowing colors
  await page.screenshot({ path: 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/flowing_colors_light.png' });
  console.log('Saved flowing_colors_light.png');

  // Scroll down to predict section
  await page.evaluate(() => {
    const el = document.getElementById('predict');
    if (el) el.scrollIntoView({ behavior: 'instant' });
  });
  await new Promise((r) => setTimeout(r, 500));

  // Capture scrolled view with flowing colors behind content
  await page.screenshot({ path: 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/flowing_colors_scrolled.png' });
  console.log('Saved flowing_colors_scrolled.png');

  await browser.close();
})();
