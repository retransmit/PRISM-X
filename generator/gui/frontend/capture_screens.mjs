import puppeteer from 'puppeteer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const screenshotsDir = path.join(__dirname, '..', 'screenshots');
if (!fs.existsSync(screenshotsDir)) {
  fs.mkdirSync(screenshotsDir, { recursive: true });
}

async function run() {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // 1. Desktop 1440x900 - Studio View (Visualizer)
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://127.0.0.1:8000/#studio', { waitUntil: 'networkidle0' });
  // Wait for D3 graph to settle
  await new Promise(r => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(screenshotsDir, '1_screen_studio_visualizer_desktop.png') });
  console.log('Captured: 1_screen_studio_visualizer_desktop.png');

  // Also write legacy filename for backwards compatibility
  await page.screenshot({ path: path.join(screenshotsDir, '1_screen_generator_desktop.png') });

  // 2. Desktop 1440x900 - Switch to Transaction Ledger
  // Click the "Transaction Ledger" button
  const ledgerBtn = await page.$('button:has(svg.lucide-table)');
  if (ledgerBtn) {
    await ledgerBtn.click();
    await page.waitForSelector('table', { timeout: 5000 });
    await new Promise(r => setTimeout(r, 600));
    await page.screenshot({ path: path.join(screenshotsDir, '2_screen_studio_ledger_desktop.png') });
    await page.screenshot({ path: path.join(screenshotsDir, '2_screen_transactions_desktop.png') });
    console.log('Captured: 2_screen_studio_ledger_desktop.png');
  }

  // 2b. Desktop 1440x900 - Transaction Drawer open
  // Click the first row in the table
  const firstRow = await page.$('tbody tr:first-child');
  if (firstRow) {
    await firstRow.click();
    await page.waitForSelector('.fixed.inset-y-0.right-0', { timeout: 5000 });
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: path.join(screenshotsDir, '2b_screen_transaction_drawer.png') });
    console.log('Captured: 2b_screen_transaction_drawer.png');
    // Close drawer
    await page.keyboard.press('Escape');
  }

  // 4. Desktop 1440x900 - Benchmarks Tab
  await page.goto('http://127.0.0.1:8000/#benchmark', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: path.join(screenshotsDir, '4_screen_benchmark_desktop.png') });
  console.log('Captured: 4_screen_benchmark_desktop.png');

  // 5. Laptop 1280x800 - Studio View
  await page.setViewport({ width: 1280, height: 800 });
  await page.goto('http://127.0.0.1:8000/#studio', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(screenshotsDir, 'responsive_laptop_1280.png') });
  console.log('Captured: responsive_laptop_1280.png');

  // 6. Tablet 768x1024 - Studio View
  await page.setViewport({ width: 768, height: 1024 });
  await page.goto('http://127.0.0.1:8000/#studio', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(screenshotsDir, 'responsive_tablet_768.png') });
  console.log('Captured: responsive_tablet_768.png');

  // 7. Mobile 390x844 - Studio View
  await page.setViewport({ width: 390, height: 844 });
  await page.goto('http://127.0.0.1:8000/#studio', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(screenshotsDir, 'responsive_mobile_390.png') });
  console.log('Captured: responsive_mobile_390.png');

  await browser.close();
  console.log('All screenshots captured successfully.');
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
