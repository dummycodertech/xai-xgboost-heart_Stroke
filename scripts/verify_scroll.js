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

  // 1. Check above the fold
  const info1 = await page.evaluate(() => {
    const wm = document.getElementById('hero-wordmark');
    const navA = document.querySelector('header a[href="#top"]');
    const r = wm.getBoundingClientRect();
    return {
      top: r.top,
      bottom: r.bottom,
      navOpacity: window.getComputedStyle(navA).opacity
    };
  });
  console.log('Above fold:', info1);

  // 2. Scroll predict section into view + 100px
  await page.evaluate(() => {
    const el = document.getElementById('predict');
    if (el) el.scrollIntoView({ behavior: 'instant' });
    window.scrollBy(0, 100);
  });
  await new Promise((r) => setTimeout(r, 600));

  // 3. Check after scroll
  const info2 = await page.evaluate(() => {
    const wm = document.getElementById('hero-wordmark');
    const navA = document.querySelector('header a[href="#top"]');
    const r = wm.getBoundingClientRect();
    return {
      top: r.top,
      bottom: r.bottom,
      navOpacity: window.getComputedStyle(navA).opacity
    };
  });
  console.log('Scrolled to predict:', info2);

  await page.screenshot({ path: 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/scrolled_nav_brand.png' });
  console.log('Saved scrolled_nav_brand.png');

  await browser.close();
})();
