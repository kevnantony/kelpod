import { useState, type FormEvent } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Navigate, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { qk, useSetupStatus } from '../api/hooks';
import { Button, ErrorBox, Loading, TextField } from '../components/ui';
import { AuthLayout } from './AuthLayout';

export function SetupPage() {
  const status = useSetupStatus();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [touched, setTouched] = useState(false);

  const pwError = touched && password.length < 8 ? 'Use at least 8 characters.' : null;
  const confirmError = touched && confirm !== password ? 'The two passwords do not match.' : null;

  const create = useMutation({
    mutationFn: async () => {
      await api.setup(username.trim(), password);
      // Sign in straight away so the admin lands in the app.
      return api.login(username.trim(), password);
    },
    onSuccess: (user) => {
      qc.setQueryData(qk.me, user);
      qc.setQueryData(qk.setup, { needs_setup: false });
      navigate('/', { replace: true });
    },
  });

  if (status.isPending) return <Loading label="Connecting to the appliance…" />;
  if (status.isError)
    return (
      <AuthLayout title="Welcome">
        <ErrorBox
          title="Cannot reach the appliance"
          error={status.error}
          onRetry={() => void status.refetch()}
        />
      </AuthLayout>
    );
  // Setup already done: go to the app if we just signed in, otherwise to the login page.
  if (!status.data.needs_setup)
    return <Navigate to={qc.getQueryData(qk.me) ? '/' : '/login'} replace />;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (password.length < 8 || confirm !== password || !username.trim()) return;
    create.mutate();
  };

  return (
    <AuthLayout
      title="Welcome to ZLLF"
      subtitle="First-time setup: create the administrator account for this appliance."
    >
      <form onSubmit={submit} className="space-y-4" noValidate>
        <TextField
          label="Administrator user name"
          autoComplete="username"
          required
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <TextField
          label="Password"
          type="password"
          autoComplete="new-password"
          required
          hint="At least 8 characters. Stored only on this appliance."
          error={pwError}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <TextField
          label="Repeat password"
          type="password"
          autoComplete="new-password"
          required
          error={confirmError}
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
        />
        {create.isError && <ErrorBox title="Setup did not complete" error={create.error} />}
        <Button type="submit" className="w-full" loading={create.isPending}>
          Create administrator
        </Button>
      </form>
    </AuthLayout>
  );
}
