import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  api,
  ApiError,
  batchUploads,
  resetTransport,
  setTransport,
  subscribeRunEvents,
  uploadDocuments,
  type EventStream,
} from './client';

function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => resetTransport());

describe('request', () => {
  it('sends JSON with credentials to the /api base path', async () => {
    const fetch = vi.fn(async () => jsonResponse(200, { id: 'u1', username: 'a', role: 'admin' }));
    setTransport({ fetch });
    const user = await api.login('a', 'secret123');
    expect(user.username).toBe('a');
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/auth/login');
    expect(init.method).toBe('POST');
    expect(init.credentials).toBe('same-origin');
    expect(new Headers(init.headers).get('content-type')).toBe('application/json');
    expect(JSON.parse(init.body as string)).toEqual({ username: 'a', password: 'secret123' });
  });

  it('encodes path parameters', async () => {
    const fetch = vi.fn(async (_url: string) => jsonResponse(200, {}));
    setTransport({ fetch });
    await api.getRun('a/b c');
    expect(fetch.mock.calls[0]![0]).toBe('/api/runs/a%2Fb%20c');
  });

  it('returns undefined for 204', async () => {
    setTransport({ fetch: async () => new Response(null, { status: 204 }) });
    await expect(api.deleteWorkspace('w1')).resolves.toBeUndefined();
  });

  it('maps the {error:{code,message}} shape to ApiError', async () => {
    setTransport({
      fetch: async () =>
        jsonResponse(409, { error: { code: 'E_RUN_ACTIVE', message: 'A run is already going.' } }),
    });
    const err = await api.startRun('w1', { mode: 'A' }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({
      status: 409,
      code: 'E_RUN_ACTIVE',
      message: 'A run is already going.',
    });
  });

  it('falls back to a helpful message for non-JSON errors', async () => {
    setTransport({ fetch: async () => new Response('<html>Bad gateway</html>', { status: 502 }) });
    const err = (await api.me().catch((e: unknown) => e)) as ApiError;
    expect(err.code).toBe('HTTP_502');
    expect(err.message).toMatch(/data is safe/i);
  });

  it('turns network failures into E_NETWORK with a user-facing message', async () => {
    setTransport({
      fetch: async () => {
        throw new TypeError('Failed to fetch');
      },
    });
    const err = (await api.health().catch((e: unknown) => e)) as ApiError;
    expect(err.code).toBe('E_NETWORK');
    expect(err.message).toMatch(/could not reach/i);
  });
});

describe('uploads', () => {
  const file = (name: string, size: number) => new File([new Uint8Array(size)], name);

  it('batches by file count and size', () => {
    const items = Array.from({ length: 5 }, (_, i) => ({
      file: file(`f${i}`, 10),
      relativePath: `f${i}`,
    }));
    expect(batchUploads(items, 2).map((b) => b.length)).toEqual([2, 2, 1]);
    expect(batchUploads(items, 100, 25).map((b) => b.length)).toEqual([2, 2, 1]);
  });

  it('sends files[] and paths[] and reports overall progress', async () => {
    const forms: FormData[] = [];
    const upload = vi.fn(
      async (_url: string, fd: FormData, onProgress?: (l: number, t: number) => void) => {
        forms.push(fd);
        onProgress?.(5, 10);
        const names = fd.getAll('paths[]').map(String);
        return jsonResponse(
          201,
          names.map((n) => ({ id: n, name: n })),
        );
      },
    );
    setTransport({ upload });
    const progress: number[] = [];
    const items = [
      { file: file('a.pdf', 10), relativePath: 'manuals/a.pdf' },
      { file: file('b.txt', 10), relativePath: 'manuals/sub/b.txt' },
    ];
    const docs = await uploadDocuments('w1', items, (loaded, total) => {
      expect(total).toBe(20);
      progress.push(loaded);
    });
    expect(docs.map((d) => d.name)).toEqual(['manuals/a.pdf', 'manuals/sub/b.txt']);
    expect(upload.mock.calls[0]![0]).toBe('/api/workspaces/w1/documents');
    expect(forms[0]!.getAll('paths[]')).toEqual(['manuals/a.pdf', 'manuals/sub/b.txt']);
    expect(forms[0]!.getAll('files[]')).toHaveLength(2);
    expect(progress.at(-1)).toBe(20);
  });

  it('surfaces upload errors as ApiError', async () => {
    setTransport({
      upload: async () =>
        jsonResponse(413, { error: { code: 'E_TOO_LARGE', message: 'Too big.' } }),
    });
    await expect(
      uploadDocuments('w1', [{ file: file('a', 1), relativePath: 'a' }]),
    ).rejects.toMatchObject({ code: 'E_TOO_LARGE', message: 'Too big.' });
  });
});

describe('subscribeRunEvents', () => {
  it('parses SSE frames, ignores junk and closes on unsubscribe', () => {
    let stream: EventStream | null = null;
    const close = vi.fn();
    const eventSource = vi.fn((url: string) => {
      expect(url).toBe('/api/runs/r1/events');
      stream = { onmessage: null, onerror: null, onopen: null, close };
      return stream;
    });
    setTransport({ eventSource });
    const got: string[] = [];
    const unsub = subscribeRunEvents('r1', { onEvent: (e) => got.push(e.type) });
    const s = stream as unknown as EventStream;
    s.onmessage?.(
      new MessageEvent('message', {
        data: '{"ts":1,"run_id":"r1","type":"stage_started","stage":"ingest"}',
      }),
    );
    s.onmessage?.(new MessageEvent('message', { data: 'not json' }));
    s.onmessage?.(new MessageEvent('message', { data: '{"no_type":true}' }));
    expect(got).toEqual(['stage_started']);
    unsub();
    expect(close).toHaveBeenCalled();
  });
});
