import type { Deployment } from '../api/types';

/** Build the chat-completions URL from the deployment's endpoint (base URL or full path). */
export function chatCompletionsUrl(endpoint: string): string {
  const e = endpoint.replace(/\/+$/, '');
  if (e.endsWith('/chat/completions')) return e;
  if (e.endsWith('/v1')) return `${e}/chat/completions`;
  return `${e}/v1/chat/completions`;
}

export function curlExample(d: Deployment): string {
  return [
    `curl ${chatCompletionsUrl(d.endpoint)} \\`,
    `  -H 'Content-Type: application/json' \\`,
    `  -d '{`,
    `    "model": "${d.model_name}",`,
    `    "messages": [{"role": "user", "content": "What is the reset procedure after fault code E-4711?"}]`,
    `  }'`,
  ].join('\n');
}

export function pythonExample(d: Deployment): string {
  const base = chatCompletionsUrl(d.endpoint).replace(/\/chat\/completions$/, '');
  return [
    'from openai import OpenAI',
    '',
    `client = OpenAI(base_url="${base}", api_key="not-needed")`,
    'reply = client.chat.completions.create(',
    `    model="${d.model_name}",`,
    '    messages=[{"role": "user", "content": "Hello"}],',
    ')',
    'print(reply.choices[0].message.content)',
  ].join('\n');
}
