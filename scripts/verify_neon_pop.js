const path = require('path');
const os = require('os');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pup-neon-'));
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: tmpDir,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webgl', '--ignore-gpu-blocklist']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  // Collect console errors
  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', err => errors.push(err.message));

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise((r) => setTimeout(r, 1200));

  // 1. Inspect line count and structure
  const stats = await page.evaluate(() => {
    const haloTop = document.querySelectorAll('.neon-halo path[stroke*="flow-top"]');
    const coreTop = document.querySelectorAll('.neon-core path[stroke*="flow-top"]');
    const haloBot = document.querySelectorAll('.neon-halo path[stroke*="flow-bot"]');
    const coreBot = document.querySelectorAll('.neon-core path[stroke*="flow-bot"]');
    const haloWave = document.querySelectorAll('.neon-halo path[stroke*="flow-wave"]');
    const coreWave = document.querySelectorAll('.neon-core path[stroke*="flow-wave"]');

    const topGrad = document.querySelector('#flow-top');
    const botGrad = document.querySelector('#flow-bot');
    const waveGrad = document.querySelector('#flow-wave');

    const computedBody = window.getComputedStyle(document.body);
    const flowCrimson = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-crimson').trim();
    const flowOrange = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-orange').trim();
    const flowGold = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-gold').trim();
    const flowDarkRed = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-dark-red').trim();
    const flowOpacity = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-opacity').trim();
    const flowGlowOpacity = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-glow-opacity').trim();

    return {
      haloTopCount: haloTop.length,
      coreTopCount: coreTop.length,
      haloBotCount: haloBot.length,
      coreBotCount: coreBot.length,
      haloWaveCount: haloWave.length,
      coreWaveCount: coreWave.length,
      totalLines: haloTop.length + haloBot.length + haloWave.length,
      totalPaths: haloTop.length + coreTop.length + haloBot.length + coreBot.length + haloWave.length + coreWave.length,
      topStops: topGrad ? topGrad.querySelectorAll('stop').length : 0,
      botStops: botGrad ? botGrad.querySelectorAll('stop').length : 0,
      waveStops: waveGrad ? waveGrad.querySelectorAll('stop').length : 0,
      colors: { flowCrimson, flowOrange, flowGold, flowDarkRed, flowOpacity, flowGlowOpacity }
    };
  });

  console.log('=== Neon Backdrop Line & Color Stats ===');
  console.log(JSON.stringify(stats, null, 2));

  // 2. Check Gradient Animation
  const g1 = await page.evaluate(() => {
    const topG = document.querySelector('#flow-top');
    const waveG = document.querySelector('#flow-wave');
    return {
      top: topG ? topG.getAttribute('gradientTransform') : null,
      wave: waveG ? waveG.getAttribute('gradientTransform') : null,
    };
  });

  await new Promise((r) => setTimeout(r, 400));

  const g2 = await page.evaluate(() => {
    const topG = document.querySelector('#flow-top');
    const waveG = document.querySelector('#flow-wave');
    return {
      top: topG ? topG.getAttribute('gradientTransform') : null,
      wave: waveG ? waveG.getAttribute('gradientTransform') : null,
    };
  });

  console.log('Top gradient rotation flowing:', g1.top !== g2.top, `(${g1.top} -> ${g2.top})`);
  console.log('Wave gradient translation flowing:', g1.wave !== g2.wave, `(${g1.wave} -> ${g2.wave})`);

  // Move mouse around to trigger cursor interaction
  await page.mouse.move(720, 450);
  await new Promise((r) => setTimeout(r, 100));
  await page.mouse.move(1060, 260);
  await new Promise((r) => setTimeout(r, 300));

  // Screenshot 1: Dark Mode Neon Pop
  const darkPath = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/neon_pop_dark.png';
  await page.screenshot({ path: darkPath });
  console.log('Saved:', darkPath);

  // Toggle to Light Mode
  await page.evaluate(() => {
    const btn = document.querySelector('button[aria-label*="theme"]');
    if (btn) btn.click();
  });
  await new Promise((r) => setTimeout(r, 400));

  const lightColors = await page.evaluate(() => {
    const flowCrimson = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-crimson').trim();
    const flowOrange = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-orange').trim();
    const flowGold = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-gold').trim();
    const flowDarkRed = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-dark-red').trim();
    const flowOpacity = window.getComputedStyle(document.documentElement).getPropertyValue('--flow-opacity').trim();
    return { flowCrimson, flowOrange, flowGold, flowDarkRed, flowOpacity };
  });
  console.log('Light Mode Theme Tokens:', lightColors);

  // Screenshot 2: Light Mode Neon Pop
  const lightPath = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/neon_pop_light.png';
  await page.screenshot({ path: lightPath });
  console.log('Saved:', lightPath);

  // Scroll down to show dense neon lines weaving behind content sections
  await page.evaluate(() => {
    const el = document.getElementById('predict');
    if (el) el.scrollIntoView({ behavior: 'instant' });
  });
  await new Promise((r) => setTimeout(r, 500));

  // Screenshot 3: Scrolled View with dense neon lines
  const scrolledPath = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/neon_pop_scrolled.png';
  await page.screenshot({ path: scrolledPath });
  console.log('Saved:', scrolledPath);

  console.log('Errors logged:', errors);
  await browser.close();
})();
