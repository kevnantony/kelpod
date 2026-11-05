import { Link } from 'react-router-dom';
import { EmptyState } from '../components/ui';

export function NotFoundPage() {
  return (
    <EmptyState
      title="Page not found"
      action={
        <Link to="/" className="font-medium text-indigo-700 underline">
          Go to your workspaces
        </Link>
      }
    >
      The address may be mistyped, or the item was deleted.
    </EmptyState>
  );
}
