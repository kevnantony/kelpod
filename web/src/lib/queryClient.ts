import { MutationCache, QueryCache, QueryClient } from '@tanstack/react-query';
import { isApiError } from '../api/client';
import { qk } from '../api/hooks';

export function makeQueryClient(): QueryClient {
  // A 401 anywhere means the session ended: drop the cached user so RequireAuth redirects.
  const on401 = (e: unknown) => {
    if (isApiError(e) && e.status === 401) qc.setQueryData(qk.me, null);
  };
  const qc: QueryClient = new QueryClient({
    queryCache: new QueryCache({ onError: on401 }),
    mutationCache: new MutationCache({ onError: on401 }),
    defaultOptions: {
      queries: {
        retry: (count, e) => !(isApiError(e) && e.status >= 400 && e.status < 500) && count < 2,
        refetchOnWindowFocus: true,
      },
    },
  });
  return qc;
}
