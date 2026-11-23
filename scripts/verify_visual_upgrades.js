const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../web/node_modules/puppeteer-core'));

const ARTIFACT_DIR = 'C:/Users/kyaba/.gemini/antigravity/brain/686249d3-dee4-413f-9c91-6e85c65b3e6f';
const OUT_DIR = path.join(ARTIFACT_DIR, 'upgrades');
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  console.log('>>> Running visual upgrade verification...');
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: 'new',
    userDataDir: path.join(require('os').tmpdir(), 'puppeteer_upgrades_' + Date.now()),
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
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });

  // Navigate to fresh page (no localStorage yet)
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  await sleep(1500);

  // 1. Check default theme
  const theme = await page.evaluate(() => document.documentElement.dataset.theme);
  console.log('Default theme attribute:', theme);

  // 2. Capture Heart screenshot with centered, larger driver blocks
  const heroEl = await page.$('#top');
  if (heroEl) {
    await heroEl.screenshot({ path: path.join(OUT_DIR, 'heart_centered_large_blocks.png') });
    console.log('Saved heart_centered_large_blocks.png');
  }

  // 3. Switch to Brain (Outcome) mode
  console.log('Switching to Brain mode...');
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button[aria-label*="scan view"]'));
    const brainBtn = buttons.find((b) => b.textContent.includes('OUTCOME') || b.textContent.includes('BRAIN') || b.textContent.includes('02'));
    if (brainBtn) brainBtn.click();
  });
  await sleep(1800); // allow transition and model rendering

  // 4. Capture Brain screenshot to inspect opacity and sharpness
  if (heroEl) {
    await heroEl.screenshot({ path: path.join(OUT_DIR, 'brain_sharp_opaque_centered_blocks.png') });
    console.log('Saved brain_sharp_opaque_centered_blocks.png');
  }

  // 5. Test theme toggle to light
  console.log('Testing theme toggle...');
  await page.evaluate(() => {
    const toggleBtn = document.querySelector('button[aria-label*="theme"]');
    if (toggleBtn) toggleBtn.click();
  });
  await sleep(500);
  const themeAfterToggle = await page.evaluate(() => document.documentElement.dataset.theme);
  console.log('Theme after toggling once:', themeAfterToggle || 'light (default without attr)');

  await page.screenshot({ path: path.join(OUT_DIR, 'light_theme_toggled.png') });
  console.log('Saved light_theme_toggled.png');

  await browser.close();
  console.log('>>> Verification completed successfully.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
