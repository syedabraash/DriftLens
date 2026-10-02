/* Local UI checks. This script never processes clips or changes role assignments. */
const fs = require('fs');
const path = require('path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.DRIFTLENS_PLAYWRIGHT_PATH || path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const { chromium } = playwright;

const root = path.resolve(__dirname, '..');
const screenshots = path.join(root, 'outputs', 'screenshots');
const exportsDir = path.join(screenshots, 'exports');
fs.mkdirSync(exportsDir, { recursive: true });
const errors = [];
const checks = {};

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function noExceptions(page) {
  assert((await page.locator('[data-testid="stException"]').count()) === 0, 'Streamlit exception shown');
  assert(errors.length === 0, `JavaScript page errors: ${errors.join('; ')}`);
}

async function downloadExport(page, label, kind) {
  const button = page.getByRole('button', { name: label, exact: true }).first();
  await button.scrollIntoViewIfNeeded();
  const pending = page.waitForEvent('download', { timeout: 20000 });
  await button.click();
  const download = await pending;
  const filename = path.basename(download.suggestedFilename());
  const target = path.join(exportsDir, filename);
  await download.saveAs(target);
  assert((await download.failure()) === null, `${label} download failed`);
  const size = fs.statSync(target).size;
  assert(size > 20, `${label} download is empty`);
  if (kind === 'csv') {
    const contents = fs.readFileSync(target, 'utf8');
    assert(contents.split(/\r?\n/).length > 2, `${label} has no data rows`);
  } else if (kind === 'json') {
    assert(typeof JSON.parse(fs.readFileSync(target, 'utf8')) === 'object', 'Invalid downloaded JSON');
  } else if (kind === 'video') {
    const contents = fs.readFileSync(target);
    assert(contents.subarray(4, 8).toString() === 'ftyp', 'Downloaded video is not an MP4');
  }
  return { filename, bytes: size, verified: true };
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.DRIFTLENS_CHROME_PATH || path.join(process.env.ProgramFiles || 'C:/Program Files', 'Google/Chrome/Application/chrome.exe'),
    headless: true,
    args: ['--disable-gpu', '--no-first-run', '--autoplay-policy=no-user-gesture-required'],
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1080 }, acceptDownloads: true });
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8510', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.getByRole('tab', { name: 'Shot review', exact: true }).waitFor({ timeout: 40000 });
    await noExceptions(page);
    checks.loaded = true;
    await page.screenshot({ path: path.join(screenshots, 'overview.png') });

    const reviewTab = page.getByRole('tab', { name: 'Shot review', exact: true });
    await reviewTab.click();
    const reviewPanel = page.getByRole('tabpanel', {name: 'Shot review'});
    const resultSelect = reviewPanel.getByRole('combobox', {name: 'Tracking result', exact: true});
    await resultSelect.scrollIntoViewIfNeeded();
    const originalSelection = await resultSelect.inputValue();
    await reviewPanel.locator('[data-testid="stSelectbox"]').first().getByRole('button', {name:'Open',exact:true}).click();
    const options = page.getByRole('option');
    const count = await options.count();
    assert(count > 0, 'Tracking result selector has no results');
    if (count > 1) {
      await options.nth(1).click();
      await page.waitForTimeout(600);
      checks.clipSelectorChanged = (await resultSelect.inputValue()) !== originalSelection;
      assert(checks.clipSelectorChanged, 'Tracking result selector did not change');
      await reviewPanel.locator('[data-testid="stSelectbox"]').first().getByRole('button', {name:'Open',exact:true}).click();
      await page.getByRole('option').first().click();
      await page.waitForTimeout(1500);
    } else {
      await options.first().click();
      checks.clipSelectorChanged = 'Only one completed result is available';
    }
    await noExceptions(page);
    const video = reviewPanel.locator('video').first();
    await video.waitFor({ state: 'visible', timeout: 20000 });
    await video.scrollIntoViewIfNeeded();
    await page.waitForFunction(() => {
      const video = document.querySelector('[role="tabpanel"]:not([hidden]) video');
      return video && video.readyState >= 2 && Number.isFinite(video.duration) && video.duration > 0;
    }, null, { timeout: 20000 });
    checks.videoMetadata = await video.evaluate(element => ({ readyState: element.readyState, duration: element.duration, width: element.videoWidth, height: element.videoHeight }));
    assert(checks.videoMetadata.width > 0 && checks.videoMetadata.height > 0, 'Replay has no decoded video dimensions');
    await video.evaluate(async element => { element.currentTime = 0; await element.play(); });
    await page.waitForTimeout(850);
    checks.playbackAdvanced = await video.evaluate(element => element.currentTime > .2 && !element.error);
    assert(checks.playbackAdvanced, 'Replay playback did not advance');
    await video.evaluate(element => element.pause());
    const charts = reviewPanel.locator('[data-testid="stVegaLiteChart"], [data-testid="stAltairChart"]');
    checks.reviewCharts = await charts.count();
    assert(checks.reviewCharts >= 2, 'Assigned role review lacks both charts');
    assert((await charts.locator('canvas, svg.marks').count()) >= 2, 'Charts have no rendered canvas or SVG');
    await page.getByRole('heading',{name:'Annotated replay',exact:true}).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'review.png') });
    checks.downloads = [];
    for (const [label, kind] of [['Observations CSV', 'csv'], ['Frame metrics CSV', 'csv'], ['Run summary JSON', 'json'], ['Annotated replay', 'video']]) {
      checks.downloads.push(await downloadExport(page, label, kind));
      await noExceptions(page);
    }

    const library = page.locator('[data-testid="stExpander"]').filter({has: page.getByText('The shot library', {exact:true})});
    await library.locator('summary').click();
    await page.waitForFunction(() => {
      const images = [...document.querySelectorAll('[data-testid="stImage"] img')];
      return images.length >= 12 && images.every(image=>image.naturalWidth > 0);
    }, null, {timeout:20000});
    checks.libraryThumbnails = await library.locator('img').count();
    await library.screenshot({path:path.join(screenshots,'library.png')});
    await library.locator('summary').click();

    await page.getByRole('tab', { name: 'Process a clip', exact: true }).click();
    await page.getByText('Process a continuous shot', { exact: true }).waitFor();
    assert(await page.getByText('Local source video', { exact: true }).isVisible(), 'Process source selector is absent');
    assert(await page.getByRole('button', { name: 'Process clip locally', exact: true }).isVisible(), 'Process button is absent');
    checks.processTab = true;
    await noExceptions(page);
    await page.getByText('Process a continuous shot', { exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'process.png') });

    await page.getByRole('tab', { name: 'Evidence', exact: true }).click();
    await page.getByText('Measured evidence, visible limitations', { exact: true }).waitFor();
    assert(await page.getByText('Reference annotation evaluation', { exact: true }).isVisible(), 'Reference evaluation heading is absent');
    checks.evidenceTab = true;
    await noExceptions(page);
    await page.getByText('Measured evidence, visible limitations', { exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(screenshots, 'evidence.png') });
    checks.downloads.push(await downloadExport(page, 'Download saved evaluation', 'json'));
    checks.downloads.push(await downloadExport(page, 'Read the results report · docs/RESULTS.md', 'text'));

    await page.getByRole('tab', {name:'Shot review',exact:true}).click();
    await reviewPanel.locator('[data-testid="stSelectbox"]').first().getByRole('button',{name:'Open',exact:true}).click();
    await page.getByRole('option').filter({hasText:'clip07 ·'}).click();
    await page.getByRole('tab', {name:'Evidence',exact:true}).click();
    const comparisonExpander=page.locator('[data-testid="stExpander"]').filter({hasText:'Inspect botsort comparison'});
    await comparisonExpander.locator('summary').first().click();
    const comparisonVideo=comparisonExpander.locator('video');
    await comparisonVideo.scrollIntoViewIfNeeded();
    await page.waitForFunction(() => [...document.querySelectorAll('video')].some(video=>video.readyState >= 2 && Math.abs(video.duration-4.3) < .05),null,{timeout:20000});
    checks.comparisonReplay = await comparisonVideo.evaluate(video=>({duration:video.duration,width:video.videoWidth,height:video.videoHeight}));
    assert(checks.comparisonReplay.width === 1280, 'BoTSORT comparison replay did not decode');
    await comparisonExpander.screenshot({path:path.join(screenshots,'tracker_comparison.png')});
    await noExceptions(page);
    await page.getByRole('tab', { name: 'Shot review', exact: true }).click();
    await reviewPanel.locator('[data-testid="stSelectbox"]').first().getByRole('button',{name:'Open',exact:true}).click();
    await page.getByRole('option').first().click();
    await noExceptions(page);
    checks.reviewReturn = true;
    checks.noExceptions = true;
    checks.note = 'Verified real saved replay playback, charts, navigation, selection and exports. Numeric accuracy is separately measured against sparse visual reference labels.';
    fs.writeFileSync(path.join(screenshots, 'browser_qa.json'), JSON.stringify(checks, null, 2));
    const errorFile=path.join(screenshots,'browser_qa_error.txt');
    if(fs.existsSync(errorFile)) fs.unlinkSync(errorFile);
    process.stdout.write(JSON.stringify(checks, null, 2) + '\n');
  } finally {
    await browser.close();
  }
})().catch(error => {
  fs.writeFileSync(path.join(screenshots, 'browser_qa_error.txt'), error.stack || String(error));
  process.stderr.write((error.stack || String(error)) + '\n');
  process.exitCode = 1;
});
