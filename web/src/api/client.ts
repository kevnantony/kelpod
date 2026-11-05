// Thin, typed wrapper around the ZLLF HTTP API (docs/API.md).
// Auth is the `zllf_session` cookie, so every request is sent with credentials.
import type {
  Adapter,
  ChatRequest,
  ChatResponse,
  CreateWorkspaceBody,
  Deployment,
  DocumentInfo,
  GoldItem,
  HardwareProfile,
  Health,
  ModelInfo,
  RedactionSummary,
  Run,
  RunEvent,
  RunSummary,
  Scorecard,
  StartRunBody,
  Trial,
  User,
  Workspace,
} from './types';

export const API_BASE = '/api';

/** Error thrown for every failed call. `message` is always user-facing text. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError;
}

// ------------------------------------------------------------------ transport
// The transport is swappable so mock mode (VITE_MOCK=1) and tests can plug in a fake backend
// without patching globals.

/** The backend rejects state-changing requests without this header (403 E_CSRF). */
export const CSRF_HEADER = 'X-ZLLF-CSRF';

export type UploadProgress = (loadedBytes: number, totalBytes: number) => void;

/** Minimal subset of EventSource we rely on. */
export type EventStream = {
  onmessage: ((ev: MessageEvent<string>) => void) | null;
  onerror: ((ev: Event) => void) | null;
  onopen?: ((ev: Event) => void) | null;
  close(): void;
};

export type Transport = {
  fetch: (input: string, init?: RequestInit) => Promise<Response>;
  upload: (url: string, body: FormData, onProgress?: UploadProgress) => Promise<Response>;
  eventSource: (url: string) => EventStream;
};

function xhrUpload(url: string, body: FormData, onProgress?: UploadProgress): Promise<Response> {
  // fetch() has no upload progress, so uploads use XMLHttpRequest.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.withCredentials = true;
    xhr.setRequestHeader(CSRF_HEADER, '1');
    xhr.responseType = 'text';
    xhr.upload.onprogress = (ev) => {
      if (onProgress && ev.lengthComputable) onProgress(ev.loaded, ev.total);
    };
    xhr.onload = () => {
      const headers = new Headers();
      const ct = xhr.getResponseHeader('content-type');
      if (ct) headers.set('content-type', ct);
      const status = xhr.status;
      // Response() rejects bodies on null-body statuses.
      const nullBody = status === 204 || status === 205 || status === 304;
      resolve(new Response(nullBody ? null : xhr.responseText, { status, headers }));
    };
    xhr.onerror = () => reject(new TypeError('Network request failed'));
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'));
    xhr.send(body);
  });
}

const browserTransport: Transport = {
  fetch: (input, init) => globalThis.fetch(input, init),
  upload: xhrUpload,
  eventSource: (url) => new EventSource(url, { withCredentials: true }),
};

let transport: Transport = browserTransport;

export function setTransport(t: Partial<Transport>): void {
  transport = { ...transport, ...t };
}

export function resetTransport(): void {
  transport = browserTransport;
}

// ------------------------------------------------------------------ core request

const NETWORK_ERROR_MESSAGE =
  'Could not reach the ZLLF appliance. Check that the service is running, then try again. ' +
  'Your data has not left this machine.';

function fallbackMessage(status: number): string {
  if (status === 401) return 'Your session has ended. Please sign in again.';
  if (status === 403) return 'You do not have permission to do this. Ask an administrator.';
  if (status === 404) return 'This item was not found. It may have been deleted.';
  if (status === 409) return 'This action is not possible right now. Refresh and try again.';
  if (status === 413) return 'The upload is too large. Try fewer or smaller files at once.';
  if (status >= 500)
    return 'The appliance hit an internal problem. Your data is safe. Try again, and if it keeps happening, ask an administrator to check the logs.';
  return `The request failed (HTTP ${status}). Try again.`;
}

/** Turn a non-OK Response into an ApiError, using the `{error:{code,message}}` body if present. */
export async function toApiError(res: Response): Promise<ApiError> {
  let code = `HTTP_${res.status}`;
  let message = fallbackMessage(res.status);
  try {
    const text = await res.text();
    if (text) {
      const body: unknown = JSON.parse(text);
      if (body && typeof body === 'object' && 'error' in body) {
        const err = (body as { error: unknown }).error;
        if (err && typeof err === 'object') {
          const e = err as { code?: unknown; message?: unknown };
          if (typeof e.code === 'string') code = e.code;
          if (typeof e.message === 'string' && e.message) message = e.message;
        }
      }
    }
  } catch {
    // Non-JSON error body (proxy page etc.) — keep the fallback message.
  }
  return new ApiError(res.status, code, message);
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set('Accept', 'application/json');
  if (method !== 'GET' && method !== 'HEAD') headers.set(CSRF_HEADER, '1');
  let payload: BodyInit | undefined;
  if (body instanceof FormData) {
    payload = body;
  } else if (body !== undefined) {
    headers.set('Content-Type', 'application/json');
    payload = JSON.stringify(body);
  }
  let res: Response;
  try {
    res = await transport.fetch(`${API_BASE}${path}`, {
      ...init,
      method,
      headers,
      body: payload,
      credentials: 'same-origin',
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new ApiError(0, 'E_NETWORK', NETWORK_ERROR_MESSAGE);
  }
  return parse<T>(res);
}

const get = <T>(p: string) => request<T>('GET', p);
const post = <T>(p: string, b?: unknown) => request<T>('POST', p, b);
const del = (p: string) => request<void>('DELETE', p);
const enc = encodeURIComponent;

// ------------------------------------------------------------------ uploads

export type UploadItem = { file: File; relativePath: string };

/** Split files into batches so a big folder becomes several moderately sized requests. */
export function batchUploads(
  items: UploadItem[],
  maxFiles = 25,
  maxBytes = 64 * 1024 * 1024,
): UploadItem[][] {
  const batches: UploadItem[][] = [];
  let cur: UploadItem[] = [];
  let bytes = 0;
  for (const it of items) {
    if (cur.length > 0 && (cur.length >= maxFiles || bytes + it.file.size > maxBytes)) {
      batches.push(cur);
      cur = [];
      bytes = 0;
    }
    cur.push(it);
    bytes += it.file.size;
  }
  if (cur.length) batches.push(cur);
  return batches;
}

/**
 * Upload files (optionally from folders) in batches. `paths[]` carries each file's relative
 * path so the backend can keep the folder structure. Progress is reported over all bytes.
 */
export async function uploadDocuments(
  workspaceId: string,
  items: UploadItem[],
  onProgress?: UploadProgress,
): Promise<DocumentInfo[]> {
  const total = items.reduce((s, i) => s + i.file.size, 0);
  let done = 0;
  const out: DocumentInfo[] = [];
  for (const batch of batchUploads(items)) {
    const fd = new FormData();
    for (const it of batch) {
      fd.append('files[]', it.file, it.file.name);
      fd.append('paths[]', it.relativePath || it.file.name);
    }
    const batchBytes = batch.reduce((s, i) => s + i.file.size, 0);
    let res: Response;
    try {
      res = await transport.upload(
        `${API_BASE}/workspaces/${enc(workspaceId)}/documents`,
        fd,
        (loaded) => onProgress?.(done + Math.min(loaded, batchBytes), total),
      );
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      throw new ApiError(0, 'E_NETWORK', NETWORK_ERROR_MESSAGE);
    }
    out.push(...(await parse<DocumentInfo[]>(res)));
    done += batchBytes;
    onProgress?.(done, total);
  }
  return out;
}

// ------------------------------------------------------------------ events (SSE)

export type EventHandlers = {
  onEvent: (ev: RunEvent) => void;
  onError?: (ev: Event) => void;
  onOpen?: () => void;
};

/** Subscribe to `/api/runs/{id}/events`. Past events are replayed first. Returns unsubscribe. */
export function subscribeRunEvents(runId: string, handlers: EventHandlers): () => void {
  const es = transport.eventSource(`${API_BASE}/runs/${enc(runId)}/events`);
  es.onmessage = (msg) => {
    try {
      const data: unknown = JSON.parse(msg.data);
      if (data && typeof data === 'object' && 'type' in data) handlers.onEvent(data as RunEvent);
    } catch {
      // Ignore malformed frames (e.g. keep-alive comments).
    }
  };
  es.onerror = (ev) => handlers.onError?.(ev);
  if (handlers.onOpen) es.onopen = () => handlers.onOpen?.();
  return () => es.close();
}

// ------------------------------------------------------------------ file downloads

/** Fetch a file (with the session cookie) and hand it to the browser as a download. */
export async function downloadFile(url: string, filename: string): Promise<void> {
  let res: Response;
  try {
    res = await transport.fetch(url, { credentials: 'same-origin' });
  } catch {
    throw new ApiError(0, 'E_NETWORK', NETWORK_ERROR_MESSAGE);
  }
  if (!res.ok) throw await toApiError(res);
  const blob = await res.blob();
  const href = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

// ------------------------------------------------------------------ endpoints

export const api = {
  // auth & setup
  login: (username: string, password: string) => post<User>('/auth/login', { username, password }),
  logout: () => post<void>('/auth/logout'),
  me: () => get<User>('/me'),
  setupStatus: () => get<{ needs_setup: boolean }>('/setup'),
  setup: (username: string, password: string) => post<User>('/setup', { username, password }),

  // workspaces
  listWorkspaces: () => get<Workspace[]>('/workspaces'),
  createWorkspace: (body: CreateWorkspaceBody) => post<Workspace>('/workspaces', body),
  getWorkspace: (id: string) => get<Workspace>(`/workspaces/${enc(id)}`),
  deleteWorkspace: (id: string) => del(`/workspaces/${enc(id)}`),

  // documents
  listDocuments: (wsId: string) => get<DocumentInfo[]>(`/workspaces/${enc(wsId)}/documents`),
  deleteDocument: (wsId: string, docId: string) =>
    del(`/workspaces/${enc(wsId)}/documents/${enc(docId)}`),
  uploadDocuments,

  // models
  listModels: () => get<ModelInfo[]>('/models'),

  // runs
  startRun: (wsId: string, body: StartRunBody) => post<Run>(`/workspaces/${enc(wsId)}/runs`, body),
  listRuns: (wsId: string) => get<RunSummary[]>(`/workspaces/${enc(wsId)}/runs`),
  getRun: (id: string) => get<Run>(`/runs/${enc(id)}`),
  getRedactions: (id: string) => get<RedactionSummary>(`/runs/${enc(id)}/redactions`),
  approveRun: (id: string) => post<Run>(`/runs/${enc(id)}/approve`),
  cancelRun: (id: string) => post<Run>(`/runs/${enc(id)}/cancel`),
  resumeRun: (id: string) => post<Run>(`/runs/${enc(id)}/resume`),
  getTrials: (id: string) => get<Trial[]>(`/runs/${enc(id)}/trials`),
  getScorecard: (id: string) => get<Scorecard>(`/runs/${enc(id)}/scorecard`),
  reportUrl: (id: string, kind: 'pdf' | 'json') => `${API_BASE}/runs/${enc(id)}/report.${kind}`,

  // gold set
  getGold: (wsId: string) => get<GoldItem[]>(`/workspaces/${enc(wsId)}/gold`),
  putGold: (wsId: string, items: GoldItem[]) =>
    post<GoldItem[]>(`/workspaces/${enc(wsId)}/gold`, items),

  // adapters & deployments
  listAdapters: () => get<Adapter[]>('/adapters'),
  deployAdapter: (id: string) => post<Deployment>(`/adapters/${enc(id)}/deploy`),
  stopDeployment: (id: string) => post<Deployment>(`/deployments/${enc(id)}/stop`),
  listDeployments: () => get<Deployment[]>('/deployments'),

  // playground
  chat: (body: ChatRequest) => post<ChatResponse>('/playground/chat', body),

  // system
  health: () => get<Health>('/system/health'),
  hardware: () => get<HardwareProfile>('/system/hardware'),
};

export type Api = typeof api;
