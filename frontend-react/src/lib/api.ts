const API = '/api';

function headers(): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  const token = localStorage.getItem('cliniq_token');
  if (token) h['Authorization'] = `Bearer ${token}`;
  return h;
}

export async function apiFetch<T>(url: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(url, { ...opts, headers: { ...headers(), ...opts.headers } });
  if (res.status === 401) {
    localStorage.removeItem('cliniq_token');
    window.location.reload();
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

export async function apiFetchStream(
  url: string,
  body: unknown,
  onToken: (token: string, intent: string) => void,
  onDone: (intent: string) => void,
  onError: (err: string) => void,
) {
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify(body),
    });
    if (res.status === 401) {
      localStorage.removeItem('cliniq_token');
      window.location.reload();
      return;
    }
    const reader = res.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.token) onToken(data.token, data.intent);
          if (data.done) onDone(data.intent);
        } catch {}
      }
    }
  } catch (e) {
    onError(e instanceof Error ? e.message : 'Stream error');
  }
}
