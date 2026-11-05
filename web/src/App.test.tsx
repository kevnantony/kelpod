// End-to-end-ish smoke test: the real router + pages against the in-memory mock backend.
import { QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vitest';
import { resetTransport, setTransport } from './api/client';
import { AppRoutes } from './App';
import { makeQueryClient } from './lib/queryClient';
import { MockBackend } from './mock/backend';

function wire(backend: MockBackend) {
  setTransport({
    fetch: async (url, init) => {
      let body: unknown = init?.body;
      if (typeof body === 'string') body = JSON.parse(body);
      const res = backend.handle(init?.method ?? 'GET', url.replace(/^\/api/, ''), body);
      return res.status === 204
        ? new Response(null, { status: 204 })
        : new Response(JSON.stringify(res.body), { status: res.status });
    },
  });
}

function renderAt(path: string) {
  return render(
    <QueryClientProvider client={makeQueryClient()}>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => resetTransport());

describe('App', () => {
  it('walks through first-run setup into an empty workspace list, then creates a workspace', async () => {
    wire(new MockBackend());
    renderAt('/');
    expect(await screen.findByText('Welcome to ZLLF')).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.type(screen.getByLabelText('Repeat password'), 'password123');
    await userEvent.click(screen.getByRole('button', { name: 'Create administrator' }));
    expect(await screen.findByText('No workspaces yet')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Create your first workspace' }));
    await userEvent.type(screen.getByLabelText('Name'), 'Plant manuals');
    // wait for models to load into the picker
    await screen.findByRole('option', { name: 'Qwen/Qwen2.5-0.5B-Instruct' });
    await userEvent.click(screen.getByRole('button', { name: 'Create workspace' }));
    expect(
      await screen.findByRole('heading', { level: 1, name: /Plant manuals/ }),
    ).toBeInTheDocument();
    expect(screen.getByText('Drag files or folders here')).toBeInTheDocument();
  });

  it('redirects to login when setup is done but there is no session', async () => {
    const b = new MockBackend();
    b.handle('POST', '/setup', { username: 'admin', password: 'password123' });
    wire(b);
    renderAt('/models');
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText('User name'), 'admin');
    await userEvent.type(screen.getByLabelText('Password'), 'wrong-password');
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(
      await screen.findByText('User name or password is wrong. Try again.'),
    ).toBeInTheDocument();
  });
});
