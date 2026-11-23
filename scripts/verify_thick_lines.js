const path = require('path');
const os = require('os');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pup-thick-'));
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: tmpDir,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webgl', '--ignore-gpu-blocklist']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', err => errors.push(err.message));

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise((r) => setTimeout(r, 1000));

  // 1. Inspect lines and absence of glass-pane elements
  const info = await page.evaluate(() => {
    const topPaths = Array.from(document.querySelectorAll('svg path[stroke*="flow-top"]'));
    const botPaths = Array.from(document.querySelectorAll('svg path[stroke*="flow-bot"]'));
    const wavePaths = Array.from(document.querySelectorAll('svg path[stroke*="flow-wave"]'));

    const glassPanes = document.querySelectorAll('.glass-pane');
    const glassCards = document.querySelectorAll('.glass-card');

    const topStrokes = topPaths.map(p => p.getAttribute('stroke-width'));
    const botStrokes = botPaths.map(p => p.getAttribute('stroke-width'));
    const waveStrokes = wavePaths.map(p => p.getAttribute('stroke-width'));

    return {
      totalLines: topPaths.length + botPaths.length + wavePaths.length,
      topCount: topPaths.length,
      botCount: botPaths.length,
      waveCount: wavePaths.length,
      topStrokes,
      botStrokes,
      waveStrokes,
      glassPanesCount: glassPanes.length,
      glassCardsCount: glassCards.length
    };
  });

  console.log('=== Backdrop Lines & Glass Removal Verification ===');
  console.log(JSON.stringify(info, null, 2));

  // Screenshot 1: Dark Mode Hero (Clean without extra panes, 1.2x thicker glowing lines)
  const darkHeroPath = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/thick_lines_dark.png';
  await page.screenshot({ path: darkHeroPath });
  console.log('Saved:', darkHeroPath);

  // Scroll down to Predictor section
  await page.evaluate(() => {
    const el = document.getElementById('predict');
    if (el) el.scrollIntoView({ behavior: 'instant' });
  });
  await new Promise((r) => setTimeout(r, 500));

  const darkScrolledPath = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/thick_lines_scrolled_dark.png';
  await page.screenshot({ path: darkScrolledPath });
  console.log('Saved:', darkScrolledPath);

  // Toggle to Light Mode
  await page.evaluate(() => {
    const btn = document.querySelector('button[aria-label*="theme"]');
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 500));

  // Screenshot 3: Light Mode Scrolled View
  const lightScrolledPath = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/thick_lines_scrolled_light.png';
  await page.screenshot({ path: lightScrolledPath });
  console.log('Saved:', lightScrolledPath);

  // Scroll back to top for Light Mode Hero
  await page.evaluate(() => {
    window.scrollTo({ top: 0, behavior: 'instant' });
  });
  await new Promise((r) => setTimeout(r, 500));

  // Screenshot 4: Light Mode Hero
  const lightHeroPath = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/thick_lines_light.png';
  await page.screenshot({ path: lightHeroPath });
  console.log('Saved:', lightHeroPath);

  console.log('Errors logged:', errors);
  await browser.close();
})();
