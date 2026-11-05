import { useEffect, useState, type ReactNode } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, Navigate, NavLink, Outlet, useLocation } from 'react-router-dom';
import { api, isApiError } from '../api/client';
import { qk, useMe, useSetupStatus } from '../api/hooks';
import { Button, ErrorBox, Loading } from './ui';
import { cx } from '../lib/cx';

/** Redirects to /setup (first run) or /login when there is no session. */
export function RequireAuth({ children }: { children: ReactNode }) {
  const me = useMe();
  const unauth =
    me.data === null || (me.isError && isApiError(me.error) && me.error.status === 401);
  const setup = useSetupStatus();
  const location = useLocation();

  if (me.isPending) return <Loading label="Connecting to the appliance…" />;
  if (unauth) {
    if (setup.isPending) return <Loading label="Connecting to the appliance…" />;
    if (setup.data?.needs_setup) return <Navigate to="/setup" replace />;
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  if (me.isError)
    return (
      <div className="mx-auto max-w-lg p-6">
        <ErrorBox
          title="Cannot reach the appliance"
          error={me.error}
          onRetry={() => void me.refetch()}
        />
      </div>
    );
  return <>{children}</>;
}

const NAV = [
  { to: '/', label: 'Workspaces', end: true },
  { to: '/models', label: 'Models' },
  { to: '/playground', label: 'Try it' },
  { to: '/admin', label: 'System', adminOnly: true },
];

/** Scroll to the top when the page (path) changes, like a normal navigation. */
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo?.(0, 0);
  }, [pathname]);
  return null;
}

export function AppShell() {
  const me = useMe();
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const logout = useMutation({
    mutationFn: api.logout,
    onSettled: () => {
      qc.clear();
      qc.setQueryData(qk.me, null);
    },
  });
  const user = me.data;
  const items = NAV.filter((n) => !n.adminOnly || user?.role === 'admin');

  return (
    <div className="min-h-screen">
      <ScrollToTop />
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4">
          <Link to="/" className="flex items-center gap-2 font-bold text-slate-900">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span>
              ZLLF
              <span className="hidden font-normal text-slate-600 sm:inline"> · LoRA Factory</span>
            </span>
          </Link>
          <nav aria-label="Main" className="ml-4 hidden gap-1 md:flex">
            {items.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  cx(
                    'rounded-md px-3 py-2 text-sm font-medium',
                    isActive ? 'bg-indigo-50 text-indigo-800' : 'text-slate-700 hover:bg-slate-100',
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto hidden items-center gap-3 md:flex">
            <span className="text-sm text-slate-600" title="Signed in">
              {user?.username}
              {user?.role === 'admin' && ' (admin)'}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => logout.mutate()}
              loading={logout.isPending}
            >
              Sign out
            </Button>
          </div>
          <button
            type="button"
            className="ml-auto rounded-md p-2 text-slate-700 hover:bg-slate-100 md:hidden"
            aria-expanded={open}
            aria-controls="mobile-nav"
            aria-label={open ? 'Close menu' : 'Open menu'}
            onClick={() => setOpen((o) => !o)}
          >
            <svg
              viewBox="0 0 24 24"
              className="h-6 w-6"
              aria-hidden="true"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
            </svg>
          </button>
        </div>
        {open && (
          <nav
            id="mobile-nav"
            aria-label="Main"
            className="border-t border-slate-200 px-4 py-2 md:hidden"
          >
            {items.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  cx(
                    'block rounded-md px-3 py-2 text-sm font-medium',
                    isActive ? 'bg-indigo-50 text-indigo-800' : 'text-slate-700',
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
            <div className="mt-2 flex items-center justify-between border-t border-slate-200 pt-2">
              <span className="text-sm text-slate-600">{user?.username}</span>
              <Button variant="ghost" size="sm" onClick={() => logout.mutate()}>
                Sign out
              </Button>
            </div>
          </nav>
        )}
      </header>
      <main id="main" className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
      <footer className="mx-auto max-w-6xl px-4 pb-6 text-xs text-slate-500">
        Runs entirely on this appliance. No data leaves this machine.
      </footer>
    </div>
  );
}
