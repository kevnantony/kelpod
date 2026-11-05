// Collect files from drag-and-drop (including whole folders) or from <input type=file>.
import type { UploadItem } from '../api/client';

// Minimal typings for the File and Directory Entries API (not in lib.dom for all TS versions).
type FsEntry = {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  fullPath: string;
};
type FsFileEntry = FsEntry & { file(ok: (f: File) => void, err?: (e: unknown) => void): void };
type FsDirEntry = FsEntry & {
  createReader(): { readEntries(ok: (e: FsEntry[]) => void, err?: (e: unknown) => void): void };
};

const IGNORED = /(^|\/)(\.DS_Store|Thumbs\.db|desktop\.ini|\._.*)$/i;

function readFile(entry: FsFileEntry): Promise<File> {
  return new Promise((res, rej) => entry.file(res, rej));
}

async function readAllEntries(dir: FsDirEntry): Promise<FsEntry[]> {
  const reader = dir.createReader();
  const all: FsEntry[] = [];
  // readEntries returns results in chunks (≈100 in Chrome) until it returns an empty list.
  for (;;) {
    const chunk = await new Promise<FsEntry[]>((res, rej) => reader.readEntries(res, rej));
    if (chunk.length === 0) break;
    all.push(...chunk);
  }
  return all;
}

async function walk(entry: FsEntry, out: UploadItem[]): Promise<void> {
  if (entry.isFile) {
    const file = await readFile(entry as FsFileEntry);
    const rel = entry.fullPath.replace(/^\//, '') || file.name;
    if (!IGNORED.test(rel)) out.push({ file, relativePath: rel });
  } else if (entry.isDirectory) {
    for (const child of await readAllEntries(entry as FsDirEntry)) await walk(child, out);
  }
}

/** Files from a drop event, descending into folders where the browser supports it. */
export async function filesFromDataTransfer(dt: DataTransfer): Promise<UploadItem[]> {
  const out: UploadItem[] = [];
  const items = Array.from(dt.items ?? []);
  const entries = items
    .filter((i) => i.kind === 'file')
    .map((i) =>
      'webkitGetAsEntry' in i ? (i.webkitGetAsEntry() as unknown as FsEntry | null) : null,
    );
  if (entries.length && entries.every((e) => e !== null)) {
    for (const e of entries) await walk(e as FsEntry, out);
    return out;
  }
  return filesFromList(dt.files);
}

/** Files from <input type="file"> (with or without webkitdirectory). */
export function filesFromList(list: FileList | null | undefined): UploadItem[] {
  return Array.from(list ?? [])
    .map((file) => ({ file, relativePath: file.webkitRelativePath || file.name }))
    .filter((i) => !IGNORED.test(i.relativePath));
}
