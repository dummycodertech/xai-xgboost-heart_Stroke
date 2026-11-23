const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 2000));
  await page.screenshot({ path: 'check_heart.png' });
  console.log('Saved check_heart.png');

  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('OUTCOME') || b.textContent.includes('BRAIN'));
    if (btn) btn.click();
  });
  console.log('Clicked outcome button');
  await new Promise(r => setTimeout(r, 3000));
  await page.screenshot({ path: 'check_brain.png' });
  console.log('Saved check_brain.png');

  await browser.close();
})();
