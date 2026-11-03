import type { SessionRecord } from './types';

const STORAGE_KEY = 'viz-agent:sessions:v1';
export const MAX_SESSIONS = 15;

function isSession(value: unknown): value is SessionRecord {
  if (!value || typeof value !== 'object') return false;
  const record = value as Partial<SessionRecord>;
  return typeof record.id === 'string'
    && typeof record.title === 'string'
    && typeof record.datasetId === 'string'
    && typeof record.datasetName === 'string'
    && Array.isArray(record.messages)
    && typeof record.updatedAt === 'number';
}

export function loadSessions(): SessionRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(isSession)
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_SESSIONS);
  } catch {
    return [];
  }
}

export function persistSessions(sessions: SessionRecord[]): void {
  if (typeof localStorage === 'undefined') return;
  let list = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_SESSIONS);
  while (list.length > 0) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
      return;
    } catch {
      // 存储配额不足时逐步丢弃最早的会话
      list = list.slice(0, -1);
    }
  }
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // 忽略清理失败
  }
}
