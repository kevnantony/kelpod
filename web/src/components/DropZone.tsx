import { useRef, useState, type DragEvent } from 'react';
import type { UploadItem } from '../api/client';
import { filesFromDataTransfer, filesFromList } from '../lib/files';
import { cx } from '../lib/cx';
import { Button } from './ui';

/** Drop area that accepts files and whole folders, plus buttons for both pickers. */
export function DropZone({
  onFiles,
  disabled,
}: {
  onFiles: (items: UploadItem[]) => void;
  disabled?: boolean;
}) {
  const [over, setOver] = useState(false);
  const [reading, setReading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    if (disabled) return;
    setReading(true);
    try {
      const items = await filesFromDataTransfer(e.dataTransfer);
      if (items.length) onFiles(items);
    } finally {
      setReading(false);
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => void onDrop(e)}
      className={cx(
        'flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors',
        over ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 bg-white',
        disabled && 'opacity-60',
      )}
    >
      <svg
        viewBox="0 0 24 24"
        className="h-10 w-10 text-slate-400"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        aria-hidden="true"
      >
        <path
          d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <p className="mt-2 font-medium text-slate-900">
        {reading ? 'Reading folder…' : 'Drag files or folders here'}
      </p>
      <p className="mt-1 text-sm text-slate-600">
        PDF, Word, text, Markdown, HTML, XML, CSV and scanned images. Folders keep their structure.
      </p>
      <div className="mt-4 flex flex-wrap justify-center gap-2">
        <Button variant="secondary" disabled={disabled} onClick={() => fileInput.current?.click()}>
          Choose files
        </Button>
        <Button
          variant="secondary"
          disabled={disabled}
          onClick={() => folderInput.current?.click()}
        >
          Choose a folder
        </Button>
      </div>
      <input
        ref={fileInput}
        type="file"
        multiple
        hidden
        aria-hidden="true"
        tabIndex={-1}
        data-testid="file-input"
        onChange={(e) => {
          const items = filesFromList(e.target.files);
          e.target.value = '';
          if (items.length) onFiles(items);
        }}
      />
      <input
        ref={(el) => {
          folderInput.current = el;
          // React does not know these non-standard attributes; set them directly.
          if (el) {
            el.setAttribute('webkitdirectory', '');
            el.setAttribute('directory', '');
          }
        }}
        type="file"
        multiple
        hidden
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const items = filesFromList(e.target.files);
          e.target.value = '';
          if (items.length) onFiles(items);
        }}
      />
    </div>
  );
}
