import { QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { makeQueryClient } from './lib/queryClient';
import { AppShell, RequireAuth } from './components/AppShell';
import { AdminPage } from './pages/AdminPage';
import { CorpusPage } from './pages/CorpusPage';
import { GoalPage } from './pages/GoalPage';
import { LoginPage } from './pages/LoginPage';
import { ModelsPage } from './pages/ModelsPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { PlaygroundPage } from './pages/PlaygroundPage';
import { RedactionPage } from './pages/RedactionPage';
import { RunPage } from './pages/RunPage';
import { RunsPage } from './pages/RunsPage';
import { ScorecardPage } from './pages/ScorecardPage';
import { SetupPage } from './pages/SetupPage';
import { WorkspaceLayout } from './pages/WorkspaceLayout';
import { WorkspacesPage } from './pages/WorkspacesPage';

const queryClient = makeQueryClient();

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/setup" element={<SetupPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        <Route index element={<WorkspacesPage />} />
        <Route path="workspaces/:wsId" element={<WorkspaceLayout />}>
          <Route index element={<CorpusPage />} />
          <Route path="goal" element={<GoalPage />} />
          <Route path="runs" element={<RunsPage />} />
        </Route>
        <Route path="runs/:runId" element={<RunPage />} />
        <Route path="runs/:runId/review" element={<RedactionPage />} />
        <Route path="runs/:runId/scorecard" element={<ScorecardPage />} />
        <Route path="playground" element={<PlaygroundPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="admin" element={<AdminPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
