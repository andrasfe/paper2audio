/* Service worker: full offline support.
 *
 * On install, precaches the app shell plus every asset referenced by
 * site.json (chapter audio, figure images, figure-explanation audio).
 * Fetches are served cache-first. Range requests (audio seeking) are
 * satisfied by slicing the cached body and answering 206 — required on
 * iOS Safari, which refuses media that ignores its Range headers.
 */
const VERSION = 'p2a-v7';
const CORE = ['./', 'index.html', 'site.json', 'manifest.webmanifest', 'icon.png'];

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(VERSION);
    await cache.addAll(CORE);
    const site = await (await cache.match('site.json')).json();
    const urls = new Set();
    (site.chapters || []).forEach((c) => urls.add(c.audio));
    Object.values(site.figures || {}).forEach((f) => {
      urls.add(f.src);
      if (f.audio) urls.add(f.audio);
    });
    await cache.addAll([...urls]);
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) {
      if (k !== VERSION) await caches.delete(k);
    }
    await self.clients.claim();
  })());
});

self.addEventListener('message', (e) => {
  if (e.data === 'status') {
    e.source.postMessage({ type: 'offline-ready', version: VERSION });
  }
});

self.addEventListener('fetch', (e) => {
  if (e.request.method === 'GET') e.respondWith(handle(e.request));
});

async function handle(req) {
  const cache = await caches.open(VERSION);
  const cached = await cache.match(req, { ignoreSearch: true });
  if (!cached) {
    try {
      const resp = await fetch(req);
      if (resp.ok && new URL(req.url).origin === location.origin) {
        cache.put(req, resp.clone());
      }
      return resp;
    } catch (err) {
      return new Response('offline and not cached', { status: 503 });
    }
  }
  const range = req.headers.get('range');
  if (!range) return cached;

  const buf = await cached.arrayBuffer();
  const m = /bytes=(\d*)-(\d*)/.exec(range);
  const start = m && m[1] ? parseInt(m[1], 10) : 0;
  let end = m && m[2] ? parseInt(m[2], 10) : buf.byteLength - 1;
  end = Math.min(end, buf.byteLength - 1);
  return new Response(buf.slice(start, end + 1), {
    status: 206,
    headers: {
      'Content-Type': cached.headers.get('Content-Type') || 'application/octet-stream',
      'Content-Range': `bytes ${start}-${end}/${buf.byteLength}`,
      'Content-Length': String(end - start + 1),
      'Accept-Ranges': 'bytes',
    },
  });
}
