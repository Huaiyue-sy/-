import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EChartsOption } from 'echarts';
import ChartView, { type ChartHandle } from './ChartView';
import type { ChatMessage, ChatResponse, DatasetMeta, HealthInfo, SessionRecord, VisionResponse } from './types';
import { loadSessions, persistSessions } from './sessions';

const API = '/api';

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function Icon({ name }: { name: 'spark' | 'database' | 'upload' | 'chart' | 'send' | 'download' | 'plus' | 'attach' | 'chevron' }) {
  const common = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  switch (name) {
    case 'spark':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" />
          <path {...common} d="M18.5 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2z" />
        </svg>
      );
    case 'database':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <ellipse {...common} cx="12" cy="5.5" rx="7.5" ry="3" />
          <path {...common} d="M4.5 5.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6" />
          <path {...common} d="M4.5 11.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6" />
        </svg>
      );
    case 'upload':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
          <path {...common} d="M4.5 15.5v3A1.5 1.5 0 006 20h12a1.5 1.5 0 001.5-1.5v-3" />
        </svg>
      );
    case 'chart':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M4 19.5h16" />
          <path {...common} d="M7 16V9m5 7V5m5 11v-5" />
        </svg>
      );
    case 'send':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M4 12l16-7-5 7 5 7-16-7z" />
        </svg>
      );
    case 'download':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M12 4v11m0 0l-4-4m4 4l4-4" />
          <path {...common} d="M5 19.5h14" />
        </svg>
      );
    case 'plus':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M12 5v14M5 12h14" />
        </svg>
      );
    case 'attach':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M20 11.5l-7.8 7.8a5 5 0 01-7.1-7.1l8.2-8.2a3.4 3.4 0 014.8 4.8l-8.2 8.2a1.8 1.8 0 01-2.6-2.6l7.5-7.5" />
        </svg>
      );
    case 'chevron':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path {...common} d="M6 9.5l6 6 6-6" />
        </svg>
      );
  }
}

function PlannerBadge({ planner, kind = 'chart' }: { planner?: 'llm'; kind?: 'chart' | 'chat' | 'vision' }) {
  if (kind === 'vision') {
    return (
      <span className="badge llm">
        YOLO · 缺陷检测{planner ? ' + 在线模型' : ''}
      </span>
    );
  }
  if (!planner) return null;
  return (
    <span className="badge llm">
      {kind === 'chat' ? '在线模型 · 回答' : '在线模型 · 图表'}
    </span>
  );
}

function formatCount(n: number): string {
  return n.toLocaleString('zh-CN');
}

function makeTitle(text: string): string {
  const clean = text.replace(/\n?\[图片\].*$/, '').replace(/\s+/g, ' ').trim();
  if (!clean) return '图片缺陷检测';
  return clean.length > 28 ? `${clean.slice(0, 28)}…` : clean;
}

function formatRelativeTime(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return '刚刚';
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} 天前`;
  return new Date(timestamp).toLocaleDateString('zh-CN');
}

interface HistoryListProps {
  sessions: SessionRecord[];
  activeId: string;
  onOpen: (record: SessionRecord) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
  compact?: boolean;
}

function HistoryList({ sessions, activeId, onOpen, onDelete, onClear, compact = false }: HistoryListProps) {
  return (
    <section className={`history-section ${compact ? 'compact' : ''}`}>
      <div className="history-head">
        <span>历史会话</span>
        {sessions.length > 0 && (
          <button type="button" className="history-clear" onClick={onClear}>清空</button>
        )}
      </div>
      {sessions.length === 0 ? (
        <p className="history-empty">对话结束后会自动保存在这里</p>
      ) : (
        <ul className="history-list">
          {sessions.map((session) => (
            <li key={session.id} className={session.id === activeId ? 'active' : ''}>
              <button type="button" className="history-open" onClick={() => onOpen(session)}>
                <span className="history-title">{session.title}</span>
                <span className="history-meta">
                  {session.datasetName}
                  {session.vision ? ' · 缺陷检测' : ''}
                  {' · '}
                  {formatRelativeTime(session.updatedAt)}
                </span>
              </button>
              <button
                type="button"
                className="history-delete"
                onClick={() => onDelete(session.id)}
                aria-label={`删除会话「${session.title}」`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

interface DataPickerProps {
  groups: [string, DatasetMeta[]][];
  active: DatasetMeta | undefined;
  activeId: string;
  open: boolean;
  onToggle: () => void;
  onSelect: (id: string) => void;
  onUpload: (files: FileList | null) => void;
  uploading: boolean;
}

function DataPicker({ groups, active, activeId, open, onToggle, onSelect, onUpload, uploading }: DataPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div className={`data-picker ${open ? 'open' : ''}`}>
      <button
        type="button"
        className="data-picker-head"
        onClick={onToggle}
        aria-expanded={open}
      >
        <span className="dataset-icon"><Icon name="database" /></span>
        <span className="data-picker-current">
          <strong>{active?.name ?? '选择数据集'}</strong>
          <em>
            {active
              ? `${formatCount(active.rowCount)} 行 · ${active.columns.length} 列`
              : '展开后选择内置数据或上传 CSV'}
          </em>
        </span>
        <span className="data-picker-toggle">{open ? '收起' : '切换数据'}</span>
        <Icon name="chevron" />
      </button>

      {open && (
        <div className="data-picker-body">
          {groups.map(([group, items]) => (
            <div className="data-picker-group" key={group}>
              <div className="home-group-title">{group}</div>
              <div className="data-picker-items">
                {items.map((dataset) => (
                  <button
                    type="button"
                    key={dataset.id}
                    className={`data-pill ${dataset.id === activeId ? 'active' : ''}`}
                    onClick={() => onSelect(dataset.id)}
                  >
                    <span className="data-pill-name">
                      {dataset.name}
                      {dataset.sample && <em>抽样</em>}
                    </span>
                    <span className="data-pill-meta">
                      {formatCount(dataset.rowCount)} 行 · {dataset.columns.length} 列
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ))}

          <div
            className={`upload-zone ${uploading ? 'busy' : ''}`}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              onUpload(event.dataTransfer.files);
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              hidden
              onChange={(event) => {
                const files = event.target.files;
                event.target.value = '';
                if (files?.length) onUpload(files);
              }}
            />
            <Icon name="upload" />
            <div>
              <strong>{uploading ? '正在解析…' : '上传 CSV'}</strong>
              <span>也可以拖拽文件到这里</span>
            </div>
            <button type="button" onClick={() => inputRef.current?.click()} disabled={uploading}>
              选择文件
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

interface VisionModeBarProps {
  detail: string;
  actionLabel: string;
  onAction: () => void;
}

function VisionModeBar({ detail, actionLabel, onAction }: VisionModeBarProps) {
  return (
    <div className="data-picker vision-mode">
      <div className="data-picker-head">
        <span className="dataset-icon"><Icon name="attach" /></span>
        <span className="data-picker-current">
          <strong>表面缺陷检测模式</strong>
          <em>{detail}</em>
        </span>
        <button type="button" className="vision-exit" onClick={onAction}>{actionLabel}</button>
      </div>
    </div>
  );
}

async function postJson(path: string, body: unknown): Promise<Response> {
  const init: RequestInit = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
  try {
    return await fetch(`${API}${path}`, init);
  } catch {
    await new Promise((resolve) => setTimeout(resolve, 800));
    return fetch(`${API}${path}`, init);
  }
}

export default function App() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [activeId, setActiveId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chart, setChart] = useState<ChatResponse | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [view, setView] = useState<'home' | 'workspace'>('home');
  const [vision, setVision] = useState<VisionResponse | null>(null);
  const [visionLoading, setVisionLoading] = useState(false);
  const [pendingImage, setPendingImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState('');
  const [sessions, setSessions] = useState<SessionRecord[]>(() => loadSessions());
  const [currentSessionId, setCurrentSessionId] = useState('');
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const currentSessionRef = useRef('');
  const chartRef = useRef<ChartHandle>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  const homeImageRef = useRef<HTMLInputElement>(null);
  const composerImageRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef('');
  const autoAsked = useRef(false);
  const sessionRef = useRef(0);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  useEffect(() => {
    persistSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    const id = currentSessionId;
    if (!id) return;
    if (messages.length === 0 && !chart && !vision) return;
    const dataset = datasets.find((d) => d.id === activeId);
    const now = Date.now();
    setSessions((prev) => {
      const existing = prev.find((session) => session.id === id);
      const firstQuestion = messages.find((message) => message.role === 'user')?.text ?? '';
      const record: SessionRecord = {
        id,
        title: existing?.title ?? makeTitle(firstQuestion),
        datasetId: activeId || existing?.datasetId || '',
        datasetName: dataset?.name ?? existing?.datasetName ?? '未知数据集',
        messages,
        chart,
        vision,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
      };
      return [record, ...prev.filter((session) => session.id !== id)];
    });
  }, [messages, chart, vision, currentSessionId, activeId, datasets]);

  const active = datasets.find((d) => d.id === activeId);
  const visionMode = Boolean(pendingImage || vision);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then((data: HealthInfo) => setHealth(data))
      .catch(() => setHealth(null));
    fetch(`${API}/datasets`)
      .then((r) => r.json())
      .then((data: { datasets: DatasetMeta[] }) => {
        setDatasets(data.datasets);
        setActiveId((prev) => prev || data.datasets[0]?.id || '');
      })
      .catch(() => setError('无法连接后端服务，请确认 npm run dev 已启动。'));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading, visionLoading]);

  const groups = useMemo(() => {
    const map = new Map<string, DatasetMeta[]>();
    for (const dataset of datasets) {
      if (!map.has(dataset.group)) map.set(dataset.group, []);
      map.get(dataset.group)!.push(dataset);
    }
    return [...map.entries()];
  }, [datasets]);

  const clearImage = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = '';
    setPendingImage(null);
    setImagePreview('');
  };

  const pickImage = (file: File | null) => {
    if (!file) {
      clearImage();
      return;
    }
    const isImage = file.type.startsWith('image/') || /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name);
    if (!isImage) {
      setError('只支持图片文件（JPG / PNG / BMP / WebP）。');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError('图片不能超过 20MB。');
      return;
    }
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = URL.createObjectURL(file);
    setPendingImage(file);
    setImagePreview(previewUrlRef.current);
    setError('');
  };

  const handleImageFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) pickImage(file);
    if (homeImageRef.current) homeImageRef.current.value = '';
    if (composerImageRef.current) composerImageRef.current.value = '';
  };

  const ensureSession = useCallback(() => {
    if (currentSessionRef.current) return currentSessionRef.current;
    const id = uid();
    currentSessionRef.current = id;
    setCurrentSessionId(id);
    return id;
  }, []);

  const beginSession = () => {
    const id = uid();
    currentSessionRef.current = id;
    setCurrentSessionId(id);
  };

  const detachSession = () => {
    currentSessionRef.current = '';
    setCurrentSessionId('');
  };

  const selectDataset = (id: string) => {
    sessionRef.current += 1;
    messagesRef.current = [];
    detachSession();
    setDataPanelOpen(false);
    setActiveId(id);
    setMessages([]);
    setChart(null);
    setError('');
    setLoading(false);
    setVision(null);
    setVisionLoading(false);
    clearImage();
  };

  const openSession = (record: SessionRecord) => {
    sessionRef.current += 1;
    setDataPanelOpen(false);
    clearImage();
    window.history.replaceState(null, '', window.location.pathname);
    currentSessionRef.current = record.id;
    setCurrentSessionId(record.id);
    const dataset = datasets.find((item) => item.id === record.datasetId);
    setActiveId(dataset ? record.datasetId : '');
    messagesRef.current = record.messages;
    setMessages(record.messages);
    setChart(record.chart);
    setVision(record.vision);
    setError(dataset ? '' : '该会话使用的数据集已失效，请在左侧重新选择数据。');
    setInput('');
    setLoading(false);
    setVisionLoading(false);
    setView('workspace');
  };

  const deleteSession = (id: string) => {
    setSessions((prev) => prev.filter((session) => session.id !== id));
    if (currentSessionRef.current !== id) return;
    detachSession();
    sessionRef.current += 1;
    messagesRef.current = [];
    setMessages([]);
    setChart(null);
    setVision(null);
    setLoading(false);
    setVisionLoading(false);
    setError('');
    setInput('');
    setView('home');
  };

  const clearHistory = () => {
    if (sessions.length === 0) return;
    if (!window.confirm('确定清空全部历史会话吗？此操作无法撤销。')) return;
    setSessions([]);
    detachSession();
    sessionRef.current += 1;
    messagesRef.current = [];
    setMessages([]);
    setChart(null);
    setVision(null);
    setLoading(false);
    setVisionLoading(false);
    setError('');
    setInput('');
    setView('home');
  };

  const ask = useCallback(async (question: string, datasetId: string = activeId) => {
    const q = question.trim();
    if (!q || !datasetId) return;
    ensureSession();
    const session = sessionRef.current;
    const history = messagesRef.current
      .filter((message) => (message.role === 'user' || Boolean(message.planner)) && !message.text.startsWith('回答失败'))
      .slice(-8)
      .map((message) => ({ role: message.role, content: message.text }));
    setMessages((prev) => [...prev, { id: uid(), role: 'user', text: q }]);
    setInput('');
    setLoading(true);
    setError('');
    try {
      const response = await postJson('/chat', {
        datasetId,
        question: q,
        history,
      });
      const data = await response.json();
      if (session !== sessionRef.current) return;
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const result = data as ChatResponse;
      if (result.kind === 'chat') {
        setMessages((prev) => [...prev, {
          id: uid(),
          role: 'assistant',
          kind: 'chat',
          text: result.answer,
          planner: result.planner,
          warning: result.warning,
        }]);
        return;
      }
      if (!result.spec || !result.option) throw new Error('返回的图表数据不完整');
      setChart(result);
      setVision(null);
      setMessages((prev) => [...prev, {
        id: uid(),
        role: 'assistant',
        kind: 'chart',
        text: result.answer,
        planner: result.planner,
        warning: result.warning,
        summary: result.summary,
      }]);
    } catch (err) {
      if (session !== sessionRef.current) return;
      setMessages((prev) => [...prev, {
        id: uid(),
        role: 'assistant',
        text: `回答失败：${err instanceof Error ? err.message : String(err)}`,
      }]);
    } finally {
      if (session === sessionRef.current) setLoading(false);
    }
  }, [activeId, ensureSession]);

  const startConversation = useCallback((datasetId: string, question: string) => {
    const q = question.trim();
    if (!datasetId || !q) return;
    sessionRef.current += 1;
    messagesRef.current = [];
    beginSession();
    setDataPanelOpen(false);
    setActiveId(datasetId);
    setMessages([]);
    setChart(null);
    setError('');
    setInput('');
    setLoading(false);
    setVision(null);
    setVisionLoading(false);
    clearImage();
    setView('workspace');
    void ask(q, datasetId);
  }, [ask]);

  const runVision = useCallback(async (question: string, file: File, reset: boolean) => {
    const q = question.trim() || '检测图片中的缺陷';
    sessionRef.current += 1;
    const session = sessionRef.current;
    if (reset) {
      beginSession();
      messagesRef.current = [];
      setMessages([]);
      setChart(null);
      setInput('');
      setView('workspace');
    } else {
      ensureSession();
    }
    setVision(null);
    setError('');
    setMessages((prev) => [...prev, {
      id: uid(),
      role: 'user',
      text: `${q}\n[图片] ${file.name}`,
    }]);
    setVisionLoading(true);
    try {
      const form = new FormData();
      form.append('question', q);
      form.append('file', file);
      const response = await fetch(`${API}/vision/detect`, { method: 'POST', body: form });
      const data = await response.json();
      if (session !== sessionRef.current) return;
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const result = data as VisionResponse;
      setVision(result);
      setMessages((prev) => [...prev, {
        id: uid(),
        role: 'assistant',
        kind: 'vision',
        text: result.answer,
        planner: result.planner,
        warning: result.warning,
        summary: result.summary,
      }]);
    } catch (err) {
      if (session !== sessionRef.current) return;
      setMessages((prev) => [...prev, {
        id: uid(),
        role: 'assistant',
        text: `缺陷检测失败：${err instanceof Error ? err.message : String(err)}`,
      }]);
    } finally {
      if (session === sessionRef.current) setVisionLoading(false);
      clearImage();
      setInput('');
    }
  }, []);

  const newConversation = () => {
    sessionRef.current += 1;
    messagesRef.current = [];
    detachSession();
    setDataPanelOpen(false);
    setMessages([]);
    setChart(null);
    setError('');
    setInput('');
    setLoading(false);
    setVision(null);
    setVisionLoading(false);
    clearImage();
    window.history.replaceState(null, '', window.location.pathname);
    setView('home');
  };

  useEffect(() => {
    const question = new URLSearchParams(window.location.search).get('q');
    if (question && activeId && !autoAsked.current) {
      autoAsked.current = true;
      setView('workspace');
      void ask(question, activeId);
    }
  }, [activeId, ask]);

  const handleFiles = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('只支持 .csv 文件。');
      return;
    }
    setUploading(true);
    setError('');
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API}/datasets/upload`, { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const meta = data as DatasetMeta;
      setDatasets((prev) => [meta, ...prev.filter((d) => d.id !== meta.id)]);
      selectDataset(meta.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const exportPng = () => {
    const instance = chartRef.current?.getInstance();
    if (!instance || !chart) return;
    const url = instance.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#ffffff' });
    const link = document.createElement('a');
    link.href = url;
    link.download = `${(chart.spec?.title || 'chart').replace(/[\\/:*?"<>|]/g, '_')}.png`;
    link.click();
  };

  const exportVision = () => {
    if (!vision) return;
    const link = document.createElement('a');
    link.href = vision.image;
    link.download = 'yolo-defects.jpg';
    link.click();
  };

  if (view === 'home') {
    return (
      <div className="home">
        <header className="home-top">
          <div className="brand home-brand">
            <span className="brand-mark"><Icon name="spark" /></span>
            <div>
              <strong>工业数据分析智能体</strong>
              <span>Visualization Agent</span>
            </div>
          </div>
          <div className="home-status-group">
            <div className="home-status">
              <span className={`status-dot ${health?.llm.configured ? 'online' : 'missing'}`} />
              {health?.llm.configured ? `在线模型 · ${health.llm.model}` : '未配置在线模型'}
            </div>
            <div className="home-status">
              <span className={`status-dot ${health?.vision?.configured ? 'online' : 'missing'}`} />
              {health?.vision?.configured ? 'YOLO 已就绪' : 'YOLO 未就绪'}
            </div>
          </div>
        </header>

        <main className="home-main">
          <section className="home-hero">
            <span className="hero-mark"><Icon name="spark" /></span>
            <h1>问数据，也查缺陷</h1>
            <p>选择数据集做分析与可视化，或附加一张金属表面图片，让 YOLO 检测缺陷位置并解读结果。</p>
          </section>

          {error && <div className="error-banner">{error}</div>}

          <section className="home-console">
            {pendingImage ? (
              <VisionModeBar
                detail={`已附加 ${pendingImage.name} · 本次分析不读取数据集`}
                actionLabel="移除图片"
                onAction={clearImage}
              />
            ) : (
              <DataPicker
                groups={groups}
                active={active}
                activeId={activeId}
                open={dataPanelOpen}
                onToggle={() => setDataPanelOpen((value) => !value)}
                onSelect={selectDataset}
                onUpload={handleFiles}
                uploading={uploading}
              />
            )}

            <div className="home-console-body">
              <div className="home-tools">
                <input
                  ref={homeImageRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(event) => handleImageFiles(event.target.files)}
                />
                {pendingImage ? (
                  <div className="home-image-chip">
                    <img src={imagePreview} alt={pendingImage.name} />
                    <span>{pendingImage.name}</span>
                    <button type="button" onClick={() => homeImageRef.current?.click()} aria-label="更换图片">
                      更换
                    </button>
                    <button type="button" onClick={clearImage} aria-label="移除图片">×</button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="home-attach"
                    onClick={() => homeImageRef.current?.click()}
                    disabled={visionLoading}
                  >
                    <Icon name="attach" /> 附加图片做 YOLO 缺陷检测
                  </button>
                )}
              </div>

              <form
                className="home-composer"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (pendingImage) {
                    void runVision(input, pendingImage, true);
                  } else {
                    startConversation(activeId, input);
                  }
                }}
              >
                <input
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={pendingImage
                    ? '可选：例如 这张图主要是什么缺陷，该如何处理？'
                    : active
                      ? `例如：${active.suggestedQuestions[0] ?? '画出数据的主要分布'}`
                      : '选择数据或附加图片后输入问题'}
                  disabled={loading || visionLoading || (!active && !pendingImage)}
                  autoFocus
                />
                <button
                  type="submit"
                  disabled={loading || visionLoading || (pendingImage ? false : (!activeId || !input.trim()))}
                >
                  <Icon name="send" />
                  {pendingImage ? '开始检测' : '开始分析'}
                </button>
              </form>

              {active && !pendingImage && active.suggestedQuestions.length > 0 && (
                <div className="home-suggestions">
                  {active.suggestedQuestions.map((question) => (
                    <button
                      type="button"
                      key={question}
                      onClick={() => startConversation(activeId, question)}
                      disabled={loading || visionLoading}
                    >
                      {question}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </section>

          <section className="home-block">
            <div className="home-block-head">
              <h2>历史会话</h2>
              <span>
                {sessions.length > 0
                  ? `${sessions.length} 条记录 · 仅保存在本机浏览器`
                  : '对话结束后会自动保存在这里'}
              </span>
            </div>
            <HistoryList
              sessions={sessions}
              activeId={currentSessionId}
              onOpen={openSession}
              onDelete={deleteSession}
              onClear={clearHistory}
              compact
            />
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark"><Icon name="spark" /></span>
          <div>
            <strong>工业数据分析智能体</strong>
            <span>Visualization Agent</span>
          </div>
        </div>

        <HistoryList
          sessions={sessions}
          activeId={currentSessionId}
          onOpen={openSession}
          onDelete={deleteSession}
          onClear={clearHistory}
        />

        <div className="sidebar-foot">
          <span className={`status-dot ${health?.llm.configured ? 'online' : 'missing'}`} />
          <div>
            <strong>{health?.llm.configured ? '在线模型已接入' : '未配置在线模型'}</strong>
            <span>{health?.llm.configured ? health.llm.model : '请配置 VIZ_LLM_API_KEY / DEEPSEEK_API_KEY'}</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{pendingImage ? '表面缺陷检测' : vision ? '表面缺陷检测结果' : active?.name ?? '选择数据集'}</h1>
            <p>
              {pendingImage
                ? '已附加图片，将调用 YOLO 检测表面缺陷；本次分析不读取数据集。'
                : vision
                  ? `YOLO 检测到 ${vision.detections.length} 个目标，阈值 ≥ ${vision.conf}；本次会话不依赖数据集。`
                  : active?.description ?? '在对话上方展开数据面板，选择内置数据或上传 CSV。'}
            </p>
          </div>
          <div className="topbar-actions">
            <button type="button" className="ghost-button" onClick={newConversation}>
              <Icon name="plus" /> 新对话
            </button>
          </div>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <section className="workspace">
          <div className="chat-panel">
            {pendingImage ? (
              <VisionModeBar
                detail={`已附加 ${pendingImage.name} · 本次分析不读取数据集`}
                actionLabel="移除图片"
                onAction={clearImage}
              />
            ) : vision ? (
              <VisionModeBar
                detail={`YOLO 检测结果 · ${vision.detections.length} 个目标 · 阈值 ≥ ${vision.conf}`}
                actionLabel="退出检测"
                onAction={() => setVision(null)}
              />
            ) : (
              <DataPicker
                groups={groups}
                active={active}
                activeId={activeId}
                open={dataPanelOpen}
                onToggle={() => setDataPanelOpen((value) => !value)}
                onSelect={selectDataset}
                onUpload={handleFiles}
                uploading={uploading}
              />
            )}

            <div className="messages" ref={scrollRef}>
              {messages.length === 0 && (
                <div className="hero">
                  <span className="hero-mark"><Icon name={pendingImage ? 'attach' : 'chart'} /></span>
                  <h2>{pendingImage ? '检测表面缺陷' : '问数据，也聊思路'}</h2>
                  <p>
                    {pendingImage
                      ? '将调用 YOLO 定位缺陷并解读结果，本次不使用数据集。'
                      : '展开上方数据面板选择数据集，然后输入问题即可。'}
                  </p>
                  {active && !pendingImage && (
                    <div className="hero-facts">
                      <span>{formatCount(active.rowCount)} 行</span>
                      <span>{active.columns.length} 列</span>
                      <span>{active.source === 'upload' ? '上传数据' : '内置数据'}</span>
                    </div>
                  )}
                </div>
              )}

              {messages.map((message) => (
                <div className={`message ${message.role}`} key={message.id}>
                  <div className="avatar">
                    {message.role === 'user' ? '你' : <Icon name="spark" />}
                  </div>
                  <div className="bubble-wrap">
                    <div className="bubble">
                      {message.role === 'assistant' && <PlannerBadge planner={message.planner} kind={message.kind} />}
                      <p>{message.text}</p>
                      {message.summary && <div className="message-summary">{message.summary}</div>}
                      {message.warning && <div className="message-warning">{message.warning}</div>}
                    </div>
                  </div>
                </div>
              ))}

              {(loading || visionLoading) && (
                <div className="message assistant">
                  <div className="avatar"><Icon name="spark" /></div>
                  <div className="bubble thinking">
                    <span /><span /><span />
                    <em>{visionLoading ? 'YOLO 正在检测图片…' : '正在理解问题…'}</em>
                  </div>
                </div>
              )}
            </div>

            {active && !visionMode && active.suggestedQuestions.length > 0 && (
              <div className="suggestions">
                {active.suggestedQuestions.map((question) => (
                  <button
                    type="button"
                    key={question}
                    onClick={() => void ask(question)}
                    disabled={loading || visionLoading}
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}

            <form
              className="composer-form"
              onSubmit={(event) => {
                event.preventDefault();
                if (pendingImage) {
                  void runVision(input, pendingImage, false);
                } else {
                  void ask(input);
                }
              }}
            >
              <input
                ref={composerImageRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(event) => handleImageFiles(event.target.files)}
              />
              {pendingImage && (
                <div className="composer-attachment">
                  <img src={imagePreview} alt={pendingImage.name} />
                  <span>{pendingImage.name}</span>
                  <button type="button" onClick={clearImage} aria-label="移除图片">×</button>
                </div>
              )}
              <div className="composer">
                <button
                  type="button"
                  className="attach-button"
                  onClick={() => composerImageRef.current?.click()}
                  disabled={loading || visionLoading}
                  aria-label="附加图片"
                >
                  <Icon name="attach" />
                </button>
                <input
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={pendingImage
                    ? '可选：例如 这些缺陷该如何处理？'
                    : active
                      ? `针对「${active.name}」提问，例如：${active.suggestedQuestions[0] ?? '画出数据的分布'}`
                      : '请先选择数据集或附加图片'}
                  disabled={loading || visionLoading || (!active && !pendingImage)}
                />
                <button
                  type="submit"
                  className="send-button"
                  disabled={loading || visionLoading || (!pendingImage && (!active || !input.trim()))}
                  aria-label="发送"
                >
                  <Icon name="send" />
                </button>
              </div>
            </form>
          </div>

          <div className="chart-panel">
            <div className="chart-head">
              <div>
                <div className="chart-eyebrow">
                  <span>{vision ? '缺陷检测' : '图表'}</span>
                  {vision
                    ? <span className="badge llm">YOLO · {vision.detections.length} 个目标</span>
                    : chart && <PlannerBadge planner={chart.planner} />}
                </div>
                <h2>{vision ? 'YOLO 缺陷检测结果' : chart?.spec?.title ?? '等待生成'}</h2>
                {vision && <p>{vision.summary}</p>}
                {!vision && chart && <p>{chart.summary}</p>}
              </div>
              {vision && (
                <button type="button" className="ghost-button" onClick={exportVision}>
                  <Icon name="download" /> 标注图
                </button>
              )}
              {!vision && chart && (
                <button type="button" className="ghost-button" onClick={exportPng}>
                  <Icon name="download" /> PNG
                </button>
              )}
            </div>

            <div className="chart-body">
              {vision ? (
                <div className="vision-body">
                  <img src={vision.image} alt="YOLO 缺陷检测标注结果" />
                </div>
              ) : chart?.option ? (
                <ChartView
                  ref={chartRef}
                  option={chart.option as EChartsOption}
                />
              ) : (
                <div className="chart-empty">
                  <span className="empty-mark"><Icon name="chart" /></span>
                  <strong>图表与检测结果会出现在这里</strong>
                  <p>模型判断需要可视化时会自动出图；附加缺陷图片后会展示 YOLO 标注框。</p>
                </div>
              )}
            </div>

            {vision && (
              <div className="chart-foot">
                <span>模型 {vision.model}</span>
                <span>阈值 ≥ {vision.conf}</span>
                <span>{vision.width} × {vision.height}</span>
                <span>{vision.detections.length} 个目标</span>
                {vision.counts.map((count) => (
                  <span key={count.className}>{count.label} ×{count.count}</span>
                ))}
              </div>
            )}

            {!vision && chart?.spec && (
              <div className="chart-foot">
                <span>使用 {formatCount(chart.usedRows ?? 0)} 行</span>
                <span>{chart.spec.x ? `x = ${chart.spec.x}` : '无 x 字段'}</span>
                <span>
                  {Array.isArray(chart.spec.y) ? `y = ${chart.spec.y.join(' / ')}` : (chart.spec.y ? `y = ${chart.spec.y}` : '按记录数统计')}
                </span>
                {chart.spec.series && <span>分组 = {chart.spec.series}</span>}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
