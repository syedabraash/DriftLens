/* Real local browser validation, including an actual private raw-video upload. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { chromium } = require(process.env.DRIFTLENS_PLAYWRIGHT_PATH || 'C:/Users/m10ah/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = process.env.DRIFTLENS_PROJECT_ROOT || 'G:/DriftLens';
const output = path.join(root, 'outputs/screenshots/improvements');
const checks = { checked_at: new Date().toISOString(), uploaded_test_scope: 'Private two second source excerpt; not an independent or personal-footage accuracy benchmark' };
const errors = [];
function assert(value, message) { if (!value) throw new Error(message); }
function digest(file) { return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }
async function clean(page) {
  assert(await page.locator('[data-testid="stException"]').count() === 0, 'Streamlit exception');
  assert(errors.length === 0, errors.join(';'));
}
async function choose(page, panel, label, index) {
  const widget = panel.locator('[data-testid="stSelectbox"]').filter({ has: page.getByText(label, { exact: true }) });
  await widget.getByRole('button', { name: 'Open', exact: true }).click();
  await page.getByRole('option').nth(index).click();
}
async function download(page, panel, label, suffix) {
  const promise = page.waitForEvent('download', { timeout: 20000 });
  await panel.getByRole('button', { name: label, exact: true }).click();
  const file = await promise;
  const destination = path.join(output, path.basename(file.suggestedFilename()));
  await file.saveAs(destination);
  const content = fs.readFileSync(destination);
  assert(content.length > 40, 'Empty report download');
  if (suffix === 'json') JSON.parse(content.toString());
  return { bytes: content.length, file: path.basename(destination) };
}
(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--disable-gpu', '--no-first-run'] });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1080 }, acceptDownloads: true });
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8510/', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.getByRole('tab', { name: 'Complete run', exact: true }).waitFor({ timeout: 60000 });
    let panel = page.getByRole('tabpanel', { name: 'Complete run', exact: true });
    await panel.getByText('Measured run conclusions', { exact: true }).waitFor();
    checks.readable_conclusions = true;
    checks.report_downloads = [await download(page, panel, 'Download readable report', 'txt'), await download(page, panel, 'Download analysis JSON', 'json')];
    await choose(page, panel, 'Jump to a review event', 2);
    await panel.getByText('Inspect the exact annotated event frame', { exact: true }).waitFor();
    assert(await panel.locator('[data-testid="stImage"]').count() >= 1, 'Exact event image missing');
    checks.precise_event_still = true;
    await clean(page);
    await page.screenshot({ path: path.join(output, 'run_analysis.png'), fullPage: true });

    await page.getByRole('tab', { name: 'Evidence', exact: true }).click();
    panel = page.getByRole('tabpanel', { name: 'Evidence', exact: true });
    await panel.getByText('Fine tuning experiment', { exact: true }).waitFor();
    checks.finetuning_evidence_visible = true;
    const reviewPath = path.join(root, 'outputs/annotation_review/queue.json');
    const before = digest(reviewPath);
    const review = panel.getByText('Review reference labels for future training', { exact: true }).locator('xpath=ancestor::details[1]');
    await review.locator('summary').click();
    await review.getByText('Original source frame', { exact: true }).waitFor();
    assert(await review.locator('[data-testid="stImage"]').count() === 2, 'Original and boxed reference views missing');
    await review.getByRole('button', { name: 'Save my frame review', exact: true }).click();
    await review.locator('[data-testid="stAlert"]').waitFor();
    assert(digest(reviewPath) === before, 'Incomplete review mutated stored labels');
    checks.incomplete_human_review_rejected_without_mutation = true;
    await clean(page);

    await page.getByRole('tab', { name: 'Process a clip', exact: true }).click();
    panel = page.getByRole('tabpanel', { name: 'Process a clip', exact: true });
    await panel.getByText('Upload my video', { exact: true }).click();
    await panel.locator('input[type="file"]').setInputFiles(path.join(root, 'outputs/upload_validation/raw_upload_test.mp4'));
    await panel.getByText('Ready: raw_upload_test.mp4', { exact: true }).waitFor({ timeout: 30000 });
    checks.real_video_upload_accepted = true;
    const name = 'qa_upload_' + Date.now();
    await panel.getByRole('textbox', { name: 'Result name', exact: true }).fill(name);
    const end = panel.getByRole('spinbutton', { name: 'End seconds', exact: true });
    await end.fill('2.0');
    await panel.getByRole('button', { name: 'Process clip locally', exact: true }).click();
    await page.getByText(/Analyzed 20 sampled frames/).waitFor({ timeout: 180000 });
    await page.getByRole('tab', { name: 'Shot review', exact: true }).click();
    panel = page.getByRole('tabpanel', { name: 'Shot review', exact: true });
    await panel.getByText('Measured run conclusions', { exact: true }).waitFor();
    await clean(page);
    const directory = path.join(root, 'outputs/runs', name);
    const summary = JSON.parse(fs.readFileSync(path.join(directory, 'summary.json')));
    const analysis = JSON.parse(fs.readFileSync(path.join(directory, 'analysis.json')));
    assert(summary.status === 'complete' && summary.frame_count === 20, 'Upload did not yield a complete20frame analysis');
    assert(summary.uploaded_source && summary.uploaded_source.source_type === 'user_upload', 'Upload provenance missing');
    assert(analysis.sample_count === 20 && analysis.accepted_paired_samples === 0, 'Unreviewed roles must not become paired measurements');
    assert(fs.statSync(path.join(directory, 'annotated.mp4')).size > 10000, 'Upload replay missing');
    assert(fs.readFileSync(path.join(directory, 'report.txt'), 'utf8').length > 100, 'Upload written report missing');
    const player = panel.locator('video').first();
    await player.waitFor();
    await player.evaluate(async video => { await video.play(); });
    await page.waitForTimeout(800);
    assert(await player.evaluate(video => video.currentTime > 0 && video.duration >= 1.9 && !video.error), 'Uploaded replay did not actually play');
    checks.upload_video_playback = true;
    checks.upload_analysis = { result: name, sampled_frames: summary.frame_count, tracked_ids: summary.track_ids, automatic_report: true, replay: true, unreviewed_roles_preserved: true };
    await page.screenshot({ path: path.join(output, 'own_clip_results.png'), fullPage: true });
    checks.no_streamlit_or_javascript_exceptions = true;
    fs.writeFileSync(path.join(root, 'outputs/browser_improvements_qa.json'), JSON.stringify(checks, null, 2) + '\n');
    process.stdout.write(JSON.stringify(checks, null, 2) + '\n');
  } finally { await browser.close(); }
})().catch(error => {
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, 'error.txt'), error.stack || String(error));
  process.stderr.write(String(error.stack || error));
  process.exitCode = 1;
});
