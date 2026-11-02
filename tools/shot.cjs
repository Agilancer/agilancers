// Headless screenshot / probe harness (uses the preinstalled Playwright Chromium with WebGPU via SwiftShader).
// usage: node tools/shot.cjs <page-path-with-query> <out.png> [width height] [--eval-out=file.json] [--timeout=ms]
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
(async () => {
  const { startServer } = await import(path.join(__dirname, 'serve.mjs'));
  const port = 8000 + Math.floor(Math.random() * 900);
  const server = await startServer(port);
  const args = process.argv.slice(2);
  const flags = Object.fromEntries(args.filter(a => a.startsWith('--')).map(a => a.slice(2).split('=')));
  const pos = args.filter(a => !a.startsWith('--'));
  const [page, out, w = '1280', h = '800'] = pos;
  const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist'] });
  const ctx = await browser.newContext({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 });
  const pg = await ctx.newPage();
  // Map CDN three.js imports to the local node_modules copy so tests work offline.
  await pg.route(/cdn\.jsdelivr\.net\/npm\/three@[^/]+\/(.*)$/, async (route) => {
    const m = route.request().url().match(/three@[^/]+\/(.*)$/);
    const f = path.join(__dirname, '..', 'node_modules', 'three', m[1]);
    if (fs.existsSync(f)) return route.fulfill({ path: f, contentType: 'text/javascript' });
    return route.abort();
  });
  const logs = [];
  pg.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  pg.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await pg.goto(`http://localhost:${port}/${page}`);
  const timeout = +(flags.timeout || 180000);
  try {
    await pg.waitForFunction(() => window.__done === true, null, { timeout, polling: 250 });
  } catch (e) { logs.push('[harness] timeout waiting for __done'); }
  const result = await pg.evaluate(() => window.__result ?? null);
  if (flags['eval-out']) fs.writeFileSync(flags['eval-out'], JSON.stringify(result, null, 1));
  // Pages may also hand back a list of named PNG data-URLs (multi-view sheets).
  // Pages may expose named views: window.__views = [...], window.__setView(name) -> Promise.
  const views = await pg.evaluate(() => window.__views ?? null);
  if (views && out) {
    for (const name of views) {
      await pg.evaluate((n) => window.__setView(n), name);
      await pg.waitForTimeout(+(flags.settle || 150));
      await pg.screenshot({ path: out.replace(/\.png$/, `_${name}.png`) });
    }
  }
  if (out && !flags.noshot) await pg.screenshot({ path: out });
  console.log(logs.slice(-40).join('\n'));
  await browser.close();
  server.close();
})();
