// Mobile-drawer geometry check, 390x844 (iPhone 14).
// Asserts (a) the "ISO Field 39" badge does not overlap the verdict heading,
// and (b) Escape closes the drawer. Exits non-zero on failure.
import puppeteer from 'puppeteer';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const report = [];

const OVERLAP = (a, b) =>
  !(a.x + a.width < b.x || b.x + b.width < a.x || a.y + a.height < b.y || b.y + b.height < a.y);

async function run() {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844 });

  await page.goto('http://127.0.0.1:8000/#studio', { waitUntil: 'networkidle0' });
  const ledgerBtn = await page.$('button:has(svg.lucide-table)');
  if (!ledgerBtn) {
    console.log(JSON.stringify({ error: 'ledger button not found' }));
    await browser.close();
    process.exit(2);
  }
  await ledgerBtn.click();
  await page.waitForSelector('tbody tr', { timeout: 10000 });
  await sleep(600);

  const rows = await page.$$('tbody tr');
  let opened = false;
  for (const row of rows) {
    const clickable = await row.evaluate((el) => el.onclick !== null || /cursor-pointer/.test(el.className));
    if (!clickable) continue;
    await row.click();
    try {
      await page.waitForSelector('.fixed.inset-y-0.right-0', { timeout: 4000 });
      opened = true;
      break;
    } catch { /* try next row */ }
  }
  if (!opened) {
    console.log(JSON.stringify({ error: 'no row opened the drawer', rows: rows.length }));
    await browser.close();
    process.exit(3);
  }
  await sleep(1200); // let the detail fetch resolve

  const measured = await page.evaluate(() => {
    const panel = document.querySelector('.fixed.inset-y-0.right-0');
    const all = Array.from(panel.querySelectorAll('*'));
    const leaf = (re) => all.find((n) => n.children.length === 0 && re.test(n.textContent || ''));
    const verdict = leaf(/LEGITIMATE CARDHOLDER SPEND|FRAUDULENT TRANSACTION/);
    const badge = leaf(/ISO Field 39/);
    // DOMRect does not survive CDP serialization (it becomes {}), so copy the fields out.
    const box = (n) => {
      if (!n) return null;
      const r = n.getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height, top: r.top, bottom: r.bottom };
    };
    return {
      verdictText: verdict ? verdict.textContent.trim() : null,
      badgeText: badge ? badge.textContent.trim() : null,
      verdict: box(verdict),
      badge: box(badge),
      viewport: { w: window.innerWidth, h: window.innerHeight },
    };
  });

  report.push(measured);

  if (!measured.verdict || !measured.badge) {
    report.push({ overlap: null, note: 'element(s) missing' });
    console.log(JSON.stringify(report, null, 2));
    await page.screenshot({ path: '../screenshots/mobile_drawer.png' });
    await browser.close();
    process.exit(4);
  }

  report.push({ overlap: OVERLAP(measured.verdict, measured.badge) });
  await page.screenshot({ path: '../screenshots/mobile_drawer.png' });

  await page.keyboard.press('Escape');
  await sleep(800);
  const closed = (await page.$('.fixed.inset-y-0.right-0')) === null;
  report.push({ escapeClosedDrawer: closed });
  await page.screenshot({ path: '../screenshots/mobile_drawer_closed.png' });

  console.log(JSON.stringify(report, null, 2));
  await browser.close();
  const overlap = report.some((r) => r.overlap === true);
  process.exit(overlap || !closed ? 1 : 0);
}

run().catch((err) => {
  console.log(JSON.stringify(report, null, 2));
  console.error(err.message);
  process.exit(5);
});
