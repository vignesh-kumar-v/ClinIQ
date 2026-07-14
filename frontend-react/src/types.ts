export interface Patient {
  patient_id: string;
  patient_name: string;
  last_visit_date: string;
  chunk_count: number;
  conditions_summary: string;
  score?: number;
}

export interface PatientDetail {
  patient_id: string;
  total: number;
  demographics: Chunk[];
  conditions: Chunk[];
  medications: Chunk[];
  observations: Chunk[];
  notes: Chunk[];
  encounters: Chunk[];
}

export interface Chunk {
  text: string;
  metadata: Record<string, string>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface QueryResponse {
  answer: string;
  intent: string;
  sources: string[];
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  email: string;
  name: string;
}

export interface ActivityEntry {
  action: string;
  detail: string;
  timestamp: string;
}
