import { create } from 'zustand';
import type { AuthResponse } from '../types';

interface AuthState {
  token: string | null;
  user: AuthResponse | null;
  setAuth: (data: AuthResponse) => void;
  logout: () => void;
  isAuthenticated: () => boolean;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: localStorage.getItem('cliniq_token'),
  user: null,
  setAuth: (data) => {
    localStorage.setItem('cliniq_token', data.access_token);
    set({ token: data.access_token, user: data });
  },
  logout: () => {
    localStorage.removeItem('cliniq_token');
    set({ token: null, user: null });
  },
  isAuthenticated: () => !!get().token,
}));
