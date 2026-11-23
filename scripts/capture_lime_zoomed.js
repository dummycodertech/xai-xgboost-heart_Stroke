const path = require('path');
const os = require('os');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

async function main() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pup-lime-'));
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: tmpDir,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webgl', '--ignore-gpu-blocklist']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 2000));

  await page.evaluate(() => {
    const limeSvg = document.querySelector('svg[aria-label="LIME local surrogate rule contributions"]');
    if (limeSvg) limeSvg.scrollIntoView({ behavior: 'instant', block: 'center' });
  });
  await new Promise(r => setTimeout(r, 800));

  const artifactDir = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f/upgrades';
  const limeClosePath = path.join(artifactDir, 'lime_chart_zoomed.png');
  await page.screenshot({ path: limeClosePath });
  console.log('Saved:', limeClosePath);

  // Also toggle light theme to verify both dark and light modes
  await page.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Light');
    if (btn) btn.click();
  });
  await new Promise(r => setTimeout(r, 800));

  const limeLightPath = path.join(artifactDir, 'lime_chart_light.png');
  await page.screenshot({ path: limeLightPath });
  console.log('Saved:', limeLightPath);

  await page.evaluate(() => {
    const beeswarmSvg = document.querySelector('svg[aria-label="SHAP summary beeswarm plot"]');
    if (beeswarmSvg) beeswarmSvg.scrollIntoView({ behavior: 'instant', block: 'center' });
  });
  await new Promise(r => setTimeout(r, 800));

  const beeswarmLightPath = path.join(artifactDir, 'beeswarm_chart_light.png');
  await page.screenshot({ path: beeswarmLightPath });
  console.log('Saved:', beeswarmLightPath);

  await browser.close();
  try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch (e) {}
}

main().catch(console.error);
