import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuthStore } from '../store/authStore';
import { apiFetch, apiFetchStream } from '../lib/api';
import type { Patient, PatientDetail, ChatMessage, ActivityEntry } from '../types';
import { marked } from 'marked';

export default function Dashboard() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [patient, setPatient] = useState<PatientDetail | null>(null);
  const [search, setSearch] = useState('');
  const [semantic, setSemantic] = useState(false);
  const [tab, setTab] = useState('overview');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);

  const loadPatients = useCallback(async (q = '') => {
    setLoading(true);
    try {
      if (q && semantic) {
        const data = await apiFetch<{ results: Patient[] }>(`/api/patients/search?q=${encodeURIComponent(q)}&limit=30`);
        setPatients(data.results);
      } else {
        const params = new URLSearchParams({ limit: '100', sort_by: 'patient_name', order: 'asc' });
        if (q) params.set('q', q);
        const data = await apiFetch<{ patients: Patient[] }>(`/api/patients?${params}`);
        setPatients(data.patients);
      }
    } catch {} finally { setLoading(false); }
  }, [semantic]);

  useEffect(() => { loadPatients(search); }, [search, semantic, loadPatients]);

  const selectPatient = async (id: string) => {
    setSelectedId(id);
    setMessages([]);
    setStreamText('');
    setPatient(null);
    try {
      const data = await apiFetch<PatientDetail>(`/api/patients/${id}`);
      setPatient(data);
      setTab('overview');
      apiFetch<{ history: ActivityEntry[] }>(`/api/patients/${id}/activity?limit=50`)
        .then((act) => setActivity(act.history || []))
        .catch(() => {});
    } catch (e) {
      console.error('Failed to load patient:', e);
    }
  };

  const sendChat = async () => {
    if (!chatInput.trim() || !selectedId) return;
    const msg = chatInput.trim();
    setChatInput('');
    setMessages((prev) => [...prev, { role: 'user', content: msg }]);
    setStreaming(true);
    setStreamText('');
    let full = '';
    await apiFetchStream(
      `/api/patients/${selectedId}/query/stream`,
      { message: msg, session_id: `sess-${Date.now()}` },
      (token) => { full += token; setStreamText(full); },
      (intent) => {
        setMessages((prev) => [...prev, { role: 'assistant', content: full }]);
        setStreamText('');
        setStreaming(false);
        if (intent === 'write' || intent === 'mixed' && selectedId) {
          selectPatient(selectedId);
        }
      },
      (err) => {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${err}` }]);
        setStreamText('');
        setStreaming(false);
      },
    );
  };

  const tabs = ['overview', 'conditions', 'medications', 'observations', 'encounters', 'activity'];

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header user={user} onLogout={logout} />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          patients={patients}
          selectedId={selectedId}
          search={search}
          semantic={semantic}
          loading={loading}
          onSearch={setSearch}
          onSemantic={setSemantic}
          onSelect={selectPatient}
        />
        <main className="flex-1 flex flex-col overflow-hidden">
          {!patient ? (
            <EmptyState />
          ) : (
            <div className="flex-1 flex flex-col overflow-hidden">
              <PatientHeader patient={patient} tab={tab} onTabChange={setTab} />
              <div className="flex-1 flex overflow-hidden">
                <div className="flex-1 overflow-y-auto scrollbar-thin p-6">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={tab}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.2 }}
                    >
                      {tab === 'overview' && <OverviewTab patient={patient} />}
                      {tab === 'conditions' && <ListTab title="Conditions" items={patient.conditions} />}
                      {tab === 'medications' && <ListTab title="Medications" items={patient.medications} />}
                      {tab === 'observations' && <ListTab title="Observations" items={patient.observations} />}
                      {tab === 'encounters' && <ListTab title="Encounters" items={patient.encounters} />}
                      {tab === 'activity' && <ActivityTab activity={activity} />}
                    </motion.div>
                  </AnimatePresence>
                </div>
                <ChatPanel
                  messages={messages}
                  streaming={streaming}
                  streamText={streamText}
                  input={chatInput}
                  onInputChange={setChatInput}
                  onSend={sendChat}
                />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function Header({ user, onLogout }: { user: { email: string; name: string } | null; onLogout: () => void }) {
  return (
    <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-center gap-4 shrink-0 z-10">
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
        </div>
        <span className="text-lg font-bold text-slate-900 tracking-tight">ClinIQ</span>
      </div>
      <div className="flex-1" />
      <span className="text-sm text-slate-500">{user?.email}</span>
      <button onClick={onLogout} className="text-sm text-slate-400 hover:text-red-500 transition-colors">
        Sign out
      </button>
    </header>
  );
}

function Sidebar({
  patients, selectedId, search, semantic, loading, onSearch, onSemantic, onSelect,
}: {
  patients: Patient[];
  selectedId: string | null;
  search: string;
  semantic: boolean;
  loading: boolean;
  onSearch: (v: string) => void;
  onSemantic: (v: boolean) => void;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className="w-80 shrink-0 bg-white border-r border-slate-200 flex flex-col overflow-hidden">
      <div className="p-4 border-b border-slate-100">
        <div className="relative">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => onSearch(e.target.value)}
            placeholder="Search patients..."
            className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
          />
        </div>
        <label className="flex items-center gap-2 mt-2 text-xs text-slate-500 cursor-pointer">
          <input type="checkbox" checked={semantic} onChange={(e) => onSemantic(e.target.checked)} className="accent-brand-600 rounded" />
          Semantic search
        </label>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {loading ? (
          Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="px-4 py-3 border-b border-slate-50">
              <div className="skeleton h-4 w-3/4 rounded mb-2" />
              <div className="skeleton h-3 w-1/2 rounded" />
            </div>
          ))
        ) : patients.length === 0 ? (
          <p className="px-4 py-8 text-sm text-slate-400 text-center">No patients found</p>
        ) : (
          <AnimatePresence>
            {patients.map((p, i) => (
              <motion.button
                key={p.patient_id}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.02, duration: 0.2 }}
                onClick={() => onSelect(p.patient_id)}
                className={`w-full text-left px-4 py-3 border-b border-slate-50 transition-colors hover:bg-brand-50 ${
                  selectedId === p.patient_id ? 'bg-brand-50 border-l-2 border-l-brand-600' : ''
                }`}
              >
                <p className="text-sm font-medium text-slate-800 truncate">{p.patient_name}</p>
                <p className="text-xs text-slate-400 mt-0.5">
                  {p.last_visit_date || '—'} · {p.chunk_count} records
                </p>
                {p.score != null && (
                  <span className="inline-block mt-1 text-[10px] bg-brand-100 text-brand-700 px-1.5 py-0.5 rounded-full font-medium">
                    {(p.score * 100).toFixed(0)}%
                  </span>
                )}
              </motion.button>
            ))}
          </AnimatePresence>
        )}
      </div>
    </aside>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-slate-400 gap-3">
      <motion.div
        animate={{ y: [0, -8, 0] }}
        transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
      >
        <svg className="w-16 h-16 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      </motion.div>
      <p className="text-sm font-medium">Select a patient from the sidebar</p>
      <p className="text-xs">Use the search bar to find patients by name</p>
    </div>
  );
}

function PatientHeader({ patient, tab, onTabChange }: { patient: PatientDetail; tab: string; onTabChange: (t: string) => void }) {
  const name = patient.demographics?.[0]?.text?.split(',')[0]?.replace('Patient: ', '') || patient.patient_id;
  const tabs = ['overview', 'conditions', 'medications', 'observations', 'encounters', 'activity'];
  return (
    <div className="bg-white border-b border-slate-200 px-6 py-4 shrink-0">
      <h2 className="text-lg font-bold text-slate-900">{name}</h2>
      <p className="text-sm text-slate-500 mt-0.5">{patient.total} records · ID: {patient.patient_id.slice(0, 8)}…</p>
      <div className="flex gap-1 mt-4 bg-slate-100 rounded-lg p-1">
        {tabs.map((t) => (
          <button
            key={t}
            onClick={() => onTabChange(t)}
            className={`flex-1 py-1.5 text-xs font-medium rounded-md capitalize transition-all ${
              tab === t ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}

function OverviewTab({ patient }: { patient: PatientDetail }) {
  return (
    <div className="space-y-6">
      <Section title="Demographics">
        {patient.demographics.map((d, i) => <p key={i} className="text-sm text-slate-700 py-0.5">{d.text}</p>)}
      </Section>
      <Section title="Recent Encounters">
        {patient.encounters.slice(0, 5).map((e, i) => (
          <p key={i} className="text-sm text-slate-600 py-0.5">{e.text}</p>
        ))}
      </Section>
      <Section title="Recent Observations">
        {patient.observations.slice(0, 5).map((o, i) => (
          <p key={i} className="text-sm text-slate-600 py-0.5">{o.text}</p>
        ))}
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">{title}</h3>
      <div className="bg-white border border-slate-200 rounded-xl p-4">{children}</div>
    </div>
  );
}

function ListTab({ title, items }: { title: string; items: { text: string; metadata?: Record<string, string> }[] }) {
  if (!items.length) return <p className="text-sm text-slate-400">No {title.toLowerCase()} recorded.</p>;
  return (
    <div className="space-y-3">
      {items.map((item, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.03 }}
          className="bg-white border border-slate-200 rounded-xl p-4 text-sm text-slate-700"
        >
          {item.text}
          {item.metadata?.updated_at && (
            <p className="text-xs text-brand-500 mt-1">Updated {item.metadata.updated_at.slice(0, 10)}</p>
          )}
        </motion.div>
      ))}
    </div>
  );
}

function ActivityTab({ activity }: { activity: ActivityEntry[] }) {
  if (!activity.length) return <p className="text-sm text-slate-400">No activity recorded.</p>;
  return (
    <div className="space-y-2">
      {activity.map((a, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.02 }}
          className="flex items-start gap-3 py-2"
        >
          <div className="w-2 h-2 rounded-full bg-brand-400 mt-2 shrink-0" />
          <div>
            <p className="text-sm text-slate-700">{a.detail}</p>
            <p className="text-xs text-slate-400 mt-0.5">
              {a.action} · {new Date(a.timestamp).toLocaleString()}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function ChatPanel({
  messages, streaming, streamText, input, onInputChange, onSend,
}: {
  messages: ChatMessage[];
  streaming: boolean;
  streamText: string;
  input: string;
  onInputChange: (v: string) => void;
  onSend: () => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, streamText]);

  return (
    <div className="w-96 shrink-0 border-l border-slate-200 bg-white flex flex-col">
      <div className="px-4 py-3 border-b border-slate-100">
        <h3 className="text-sm font-semibold text-slate-700">Chat</h3>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-3">
        {messages.length === 0 && !streaming && (
          <p className="text-sm text-slate-400 text-center mt-8">Ask a clinical question about this patient.</p>
        )}
        <AnimatePresence>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.2 }}
              className={`max-w-[85%] ${m.role === 'user' ? 'ml-auto' : ''}`}
            >
              <div
                className={`px-4 py-2.5 text-sm leading-relaxed ${
                  m.role === 'user'
                    ? 'bg-brand-600 text-white rounded-2xl rounded-br-md'
                    : 'bg-slate-100 text-slate-800 rounded-2xl rounded-bl-md'
                }`}
                dangerouslySetInnerHTML={{ __html: marked.parse(m.content) as string }}
              />
            </motion.div>
          ))}
        </AnimatePresence>
        {streaming && streamText && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-[85%]"
          >
            <div
              className="bg-slate-100 text-slate-800 rounded-2xl rounded-bl-md px-4 py-2.5 text-sm leading-relaxed"
              dangerouslySetInnerHTML={{ __html: marked.parse(streamText) as string }}
            />
          </motion.div>
        )}
        {streaming && !streamText && (
          <div className="flex gap-1.5 px-4 py-2.5">
            <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        )}
        <div ref={endRef} />
      </div>
      <div className="p-4 border-t border-slate-100">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onSend()}
            placeholder="Ask a clinical question..."
            className="flex-1 px-3 py-2 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
          />
          <motion.button
            onClick={onSend}
            disabled={!input.trim() || streaming}
            whileTap={{ scale: 0.95 }}
            className="px-4 py-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-40 text-white rounded-xl transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </motion.button>
        </div>
      </div>
    </div>
  );
}
