// Wires the in-memory MockBackend into the API client's transport. Loaded only when
// VITE_MOCK=1 via a dynamic import, so none of this ships in the production bundle.
import { setTransport, type EventStream, type UploadProgress } from '../api/client';
import { MockBackend, type MockResponse } from './backend';

const STORAGE_KEY = 'zllf-mock-state-v1';
const STEP_MS = 700;

function toResponse(r: MockResponse): Response {
  if (r.status === 204 || r.body === undefined) return new Response(null, { status: r.status });
  if (r.contentType) {
    return new Response(r.body as string, {
      status: r.status,
      headers: { 'content-type': r.contentType },
    });
  }
  return new Response(JSON.stringify(r.body), {
    status: r.status,
    headers: { 'content-type': 'application/json' },
  });
}

function apiPath(url: string): string | null {
  const u = new URL(url, window.location.origin);
  return u.pathname.startsWith('/api/') ? u.pathname.slice(4) + u.search : null;
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** EventSource look-alike fed by the mock backend. */
class MockEventSource implements EventStream {
  onmessage: ((ev: MessageEvent<string>) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onopen: ((ev: Event) => void) | null = null;
  private unsub: (() => void) | null = null;
  private closed = false;

  constructor(backend: MockBackend, runId: string) {
    // Deliver asynchronously, like a real connection, after handlers are attached.
    setTimeout(() => {
      if (this.closed) return;
      this.onopen?.(new Event('open'));
      this.unsub = backend.subscribe(runId, (ev) => {
        this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(ev) }));
      });
    }, 0);
  }

  close(): void {
    this.closed = true;
    this.unsub?.();
  }
}

function loadBackend(): MockBackend {
  try {
    const saved = window.sessionStorage.getItem(STORAGE_KEY);
    if (saved) return MockBackend.fromJSON(saved);
  } catch {
    // storage unavailable or corrupt — start fresh
  }
  return new MockBackend();
}

function save(backend: MockBackend): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, backend.toJSON());
  } catch {
    // ignore
  }
}

export function installMock(): MockBackend {
  const backend = loadBackend();
  const realFetch = window.fetch.bind(window);

  setTransport({
    fetch: async (input, init) => {
      const path = apiPath(input);
      if (path === null) return realFetch(input, init);
      await delay(80 + Math.random() * 120);
      let body: unknown = init?.body;
      if (typeof body === 'string') {
        try {
          body = JSON.parse(body);
        } catch {
          // leave as string
        }
      }
      const res = backend.handle(init?.method ?? 'GET', path, body);
      save(backend);
      return toResponse(res);
    },
    upload: async (url: string, fd: FormData, onProgress?: UploadProgress) => {
      const path = apiPath(url) ?? url;
      const total = fd
        .getAll('files[]')
        .reduce<number>((s, f) => s + (f instanceof File ? f.size : 0), 0);
      for (let i = 1; i <= 5; i++) {
        await delay(120);
        onProgress?.((total * i) / 5, total);
      }
      const res = backend.handle('POST', path, fd);
      save(backend);
      return toResponse(res);
    },
    eventSource: (url: string) => {
      const m = url.match(/\/runs\/([^/]+)\/events/);
      return new MockEventSource(backend, decodeURIComponent(m?.[1] ?? ''));
    },
  });

  window.setInterval(() => {
    backend.step();
    save(backend);
  }, STEP_MS);

  // Handy for demos: `zllfMock.reset()` in the console.
  (window as unknown as { zllfMock: unknown }).zllfMock = {
    backend,
    reset: () => {
      window.sessionStorage.removeItem(STORAGE_KEY);
      window.location.reload();
    },
  };
  console.info('[ZLLF] Mock mode: using an in-memory fake backend. No data leaves the browser.');
  return backend;
}
