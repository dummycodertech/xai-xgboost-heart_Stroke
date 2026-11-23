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

  // 1. Get initial path `d` of first top contour and first wave
  const initialD = await page.evaluate(() => {
    const pTop = document.querySelector('path.contour-top');
    const pWave = document.querySelector('path.flowing-wave');
    return {
      top: pTop ? pTop.getAttribute('d')?.slice(0, 50) : null,
      wave: pWave ? pWave.getAttribute('d')?.slice(0, 50) : null,
    };
  });
  console.log('Initial paths:', initialD);

  // 2. Move mouse to (1060, 260) - near the primary top-right contour cluster
  await page.mouse.move(1060, 260);
  await new Promise((r) => setTimeout(r, 300));

  // 3. Check updated path `d`
  const updatedD = await page.evaluate(() => {
    const pTop = document.querySelector('path.contour-top');
    const pWave = document.querySelector('path.flowing-wave');
    return {
      top: pTop ? pTop.getAttribute('d')?.slice(0, 50) : null,
      wave: pWave ? pWave.getAttribute('d')?.slice(0, 50) : null,
    };
  });
  console.log('Updated paths after mouse hover near top cluster:', updatedD);

  const isTopChanged = initialD.top !== updatedD.top;
  const isWaveChanged = initialD.wave !== updatedD.wave;
  console.log('Contours reacted to cursor:', isTopChanged);
  console.log('Waves reacted to cursor:', isWaveChanged);

  await page.screenshot({ path: 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades/cursor_interactive_backdrop.png' });
  console.log('Saved cursor_interactive_backdrop.png');

  await browser.close();
})();
