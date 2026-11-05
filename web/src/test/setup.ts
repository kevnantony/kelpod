import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => cleanup());

// jsdom does not implement window.scrollTo (used by ScrollToTop).
window.scrollTo = (() => {}) as typeof window.scrollTo;
