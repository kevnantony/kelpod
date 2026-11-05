import { useState, type FormEvent } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { qk, useSetupStatus } from '../api/hooks';
import { Button, ErrorBox, TextField } from '../components/ui';
import { AuthLayout } from './AuthLayout';

export function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const qc = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const setup = useSetupStatus();
  const from = (location.state as { from?: string } | null)?.from ?? '/';

  const login = useMutation({
    mutationFn: () => api.login(username.trim(), password),
    onSuccess: (user) => {
      qc.setQueryData(qk.me, user);
      navigate(from, { replace: true });
    },
  });

  if (setup.data?.needs_setup) return <Navigate to="/setup" replace />;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    login.mutate();
  };

  return (
    <AuthLayout title="Sign in" subtitle="Use your local appliance account.">
      <form onSubmit={submit} className="space-y-4" noValidate>
        <TextField
          label="User name"
          autoComplete="username"
          required
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <TextField
          label="Password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {login.isError && <ErrorBox title="Could not sign in" error={login.error} />}
        <Button
          type="submit"
          className="w-full"
          loading={login.isPending}
          disabled={!username || !password}
        >
          Sign in
        </Button>
        <p className="text-xs text-slate-600">
          Forgot your password? Ask your administrator to reset it.
        </p>
      </form>
    </AuthLayout>
  );
}
