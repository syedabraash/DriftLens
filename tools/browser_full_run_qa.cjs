/* UI verification only. Never submit a valid role correction or process footage. */
const fs = require('fs');
const path = require('path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(process.env.DRIFTLENS_PLAYWRIGHT_PATH || path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const { chromium } = playwright;
const root = path.resolve(process.env.DRIFTLENS_PROJECT_ROOT || 'G:/DriftLens');
const output = path.join(root, 'outputs', 'screenshots', 'full_run');
const downloads = path.join(output, 'exports');
const checks = { project: root, checked_at: new Date().toISOString() };
const errors = [];

function assert(condition, message) { if (!condition) throw new Error(message); }

async function noExceptions(page) {
  assert(await page.locator('[data-testid="stException"]').count() === 0, 'Streamlit exception shown');
  assert(errors.length === 0, 'JavaScript page errors: ' + errors.join('; '));
}

async function saveDownload(page, name, kind) {
  const button = page.getByRole('button', { name, exact: true });
  await button.scrollIntoViewIfNeeded();
  const pending = page.waitForEvent('download', { timeout: 20000 });
  await button.click();
  const download = await pending;
  const target = path.join(downloads, path.basename(download.suggestedFilename()));
  await download.saveAs(target);
  assert(await download.failure() === null, name + ' download failed');
  const bytes = fs.readFileSync(target);
  assert(bytes.length > 20, name + ' download is empty');
  let value;
  if (kind === 'json') value = JSON.parse(bytes.toString('utf8'));
  if (kind === 'csv') value = bytes.toString('utf8').trim().split(/\r?\n/);
  if (kind === 'video') assert(bytes.subarray(4, 8).toString() === 'ftyp', 'Replay export is not MP4');
  return { filename: path.basename(target), bytes: bytes.length, value };
}

async function jumpTo(page, panel, shotId, expected) {
  const playerStart = Math.floor(expected);
  const widget = panel.locator('[data-testid="stSelectbox"]').filter({ has: page.getByText('Jump to camera shot', { exact: true }) });
  await widget.getByRole('button', { name: 'Open', exact: true }).click();
  const option = page.getByRole('option').filter({ hasText: shotId });
  // Labels are descriptions, so match the source shot's ordinal when the ID is absent.
  if (await option.count()) await option.first().click();
  else {
    const ordinal = Number(shotId.replace(/\D/g, ''));
    await page.getByRole('option').nth(ordinal).click();
  }
  const video = panel.locator('video').first();
  await page.waitForFunction(expected => {
    const video = document.querySelector('[role="tabpanel"]:not([hidden]) video');
    return video && video.readyState >= 2 && Math.abs(video.currentTime - expected) < .3;
  }, playerStart, { timeout: 20000 });
  await video.evaluate(element => element.pause());
  const actual = await video.evaluate(element => element.currentTime);
  assert(Math.abs(actual - playerStart) < .3, `${shotId} jumped to ${actual}, expected native start ${playerStart}`);
  return { shot_id: shotId, requested_playback_boundary_seconds: expected, expected_native_seek_seconds: playerStart, actual_seconds: actual, pre_roll_seconds: expected - playerStart };
}

(async () => {
  fs.mkdirSync(downloads, { recursive: true });
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
    const tab = page.getByRole('tab', { name: 'Complete run', exact: true });
    await tab.waitFor({ timeout: 40000 });
    assert(await tab.getAttribute('aria-selected') === 'true', 'Complete run is not the default tab');
    const panel = page.getByRole('tabpanel', { name: 'Complete run', exact: true });
    await panel.getByText('The complete tandem run', { exact: true }).waitFor();
    checks.default_complete_run = true;
    await panel.getByText('26.8 seconds', { exact: true }).waitFor({ timeout: 20000 });
    checks.prominent_duration_seconds = 26.8;
    const video = panel.locator('video').first();
    await video.scrollIntoViewIfNeeded();
    await page.waitForFunction(() => {
      const video = document.querySelector('[role="tabpanel"]:not([hidden]) video');
      return video && video.readyState >= 2 && Number.isFinite(video.duration) && video.videoWidth > 0;
    }, null, { timeout: 25000 });
    checks.video = await video.evaluate(element => ({ duration_seconds: element.duration, width: element.videoWidth, height: element.videoHeight }));
    assert(Math.abs(checks.video.duration_seconds - 26.8) < .05, 'Compiled replay is not 26.8 seconds');
    await video.evaluate(async element => { element.currentTime = 0; await element.play(); });
    await page.waitForTimeout(950);
    checks.playback_advanced_seconds = await video.evaluate(element => element.currentTime);
    assert(checks.playback_advanced_seconds > .2, 'Complete replay playback did not advance');
    await video.evaluate(element => element.pause());
    await page.screenshot({ path: path.join(output, 'complete_replay.png') });
    const charts = panel.locator('[data-testid="stVegaLiteChart"], [data-testid="stAltairChart"]');
    checks.rendered_charts = await charts.count();
    assert(checks.rendered_charts === 2, 'Complete run charts are missing');
    await charts.first().scrollIntoViewIfNeeded();
    assert(await charts.locator('canvas, svg.marks').count() >= 2, 'Charts have no rendered canvas or SVG');
    await page.screenshot({ path: path.join(output, 'complete_timeline.png') });
    checks.shot_jumps = [await jumpTo(page, panel, 'shot02', 8.2), await jumpTo(page, panel, 'shot04', 17.6)];
    await page.screenshot({ path: path.join(output, 'shot04_jump.png') });

    const timeline = await saveDownload(page, 'Complete timeline CSV', 'csv');
    assert(timeline.value.length - 1 === 268, 'Timeline export does not contain 268 data rows');
    assert(timeline.value[0].includes('playback_time') && timeline.value[0].includes('source_time'), 'Timestamp mapping columns missing');
    const summary = await saveDownload(page, 'Complete run summary', 'json');
    assert(summary.value.frame_count === 268 && Math.abs(summary.value.duration_seconds - 26.8) < .05, 'Downloaded summary disagrees with complete replay');
    const shots = await saveDownload(page, 'Camera shots JSON', 'json');
    const shotValues = Array.isArray(shots.value) ? shots.value : shots.value.shots;
    assert(shotValues.length === 4, 'Shot export does not contain four camera shots');
    const replay = await saveDownload(page, 'Complete annotated replay', 'video');
    checks.exports = [timeline, summary, shots, replay].map(item => ({ filename: item.filename, bytes: item.bytes }));
    checks.timeline_rows = timeline.value.length - 1;
    checks.exported_shots = shotValues.length;

    const roles = panel.locator('[data-testid="stExpander"]').filter({ hasText: 'Review roles for a camera shot' });
    await roles.locator('summary').first().click();
    const lead = roles.getByRole('combobox', { name: 'Lead track ID', exact: true });
    const chaseWidget = roles.locator('[data-testid="stSelectbox"]').filter({ has: page.getByText('Chase track ID', { exact: true }) });
    const leadValue = await lead.inputValue();
    if (leadValue === 'Choose an ID') {
      const leadWidget = roles.locator('[data-testid="stSelectbox"]').filter({ has: page.getByText('Lead track ID', { exact: true }) });
      await leadWidget.getByRole('button', { name: 'Open', exact: true }).click();
      await page.getByRole('option').nth(1).click();
    }
    const sameId = await lead.inputValue();
    await chaseWidget.getByRole('button', { name: 'Open', exact: true }).click();
    await page.getByRole('option', { name: sameId, exact: true }).click();
    await roles.getByRole('button', { name: 'Apply shot roles and rebuild run', exact: true }).click();
    await panel.getByText('Choose two different observed track IDs for this camera shot.', { exact: true }).waitFor();
    checks.invalid_same_id_rejected = true;
    await noExceptions(page);
    checks.no_streamlit_or_javascript_exceptions = true;
    checks.no_valid_role_mutation = true;
    fs.writeFileSync(path.join(output, 'browser_full_run_qa.json'), JSON.stringify(checks, null, 2) + '\n');
    process.stdout.write(JSON.stringify(checks, null, 2) + '\n');
  } finally {
    await browser.close();
  }
})().catch(error => {
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, 'browser_full_run_qa_error.txt'), error.stack || String(error));
  process.stderr.write((error.stack || String(error)) + '\n');
  process.exitCode = 1;
});
