// Service Worker — Memory Dashboard PWA
const CACHE_NAME = 'memory-dash-v3';
const ASSETS = [
  '/',
  '/static/style.css',
  '/static/app.js',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  // Only cache GET requests for static assets
  if (request.method !== 'GET') return;
  // Skip API calls — always go to network
  if (request.url.includes('/api/') || request.url.includes('/auth/')) return;

  event.respondWith(
    caches.match(request).then((cached) => {
      // Network-first for HTML, cache-first for assets
      if (request.headers.get('accept')?.includes('text/html')) {
        return fetch(request).then((resp) => {
          if (resp.ok) {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          }
          return resp;
        }).catch(() => cached || new Response('Offline', { status: 503 }));
      }
      // Cache-first for static assets
      if (cached) return cached;
      return fetch(request).then((resp) => {
        if (resp.ok) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
        }
        return resp;
      });
    })
  );
});
