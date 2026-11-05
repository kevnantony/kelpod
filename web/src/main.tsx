import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './index.css';

async function start() {
  // Mock mode: swap in the in-memory backend. The `if` is resolved at build time, so the mock
  // chunk is not part of a normal production build.
  if (import.meta.env.VITE_MOCK === '1') {
    const { installMock } = await import('./mock/install');
    installMock();
  }
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void start();
