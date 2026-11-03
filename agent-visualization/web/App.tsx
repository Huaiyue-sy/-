import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EChartsOption } from 'echarts';
import ChartView, { type ChartHandle } from './ChartView';
import type { ChatMessage, ChatResponse, DatasetMeta, HealthInfo } from './types';

const API = '/api';

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function Icon({ name }: { name: 'spark' | 'database' | 'upload' | 'chart' | 'send' | 'download' | 'plus' }) {
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
  }
}

function PlannerBadge({ planner, kind = 'chart' }: { planner?: 'llm'; kind?: 'chart' | 'chat' }) {
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
  const fileRef = useRef<HTMLInputElement>(null);
  const chartRef = useRef<ChartHandle>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  const autoAsked = useRef(false);
  const sessionRef = useRef(0);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const active = datasets.find((d) => d.id === activeId);

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
  }, [messages, loading]);

  const groups = useMemo(() => {
    const map = new Map<string, DatasetMeta[]>();
    for (const dataset of datasets) {
      if (!map.has(dataset.group)) map.set(dataset.group, []);
      map.get(dataset.group)!.push(dataset);
    }
    return [...map.entries()];
  }, [datasets]);

  const selectDataset = (id: string) => {
    sessionRef.current += 1;
    messagesRef.current = [];
    setActiveId(id);
    setMessages([]);
    setChart(null);
    setError('');
    setLoading(false);
  };

  const ask = useCallback(async (question: string, datasetId: string = activeId) => {
    const q = question.trim();
    if (!q || !datasetId) return;
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
  }, [activeId]);

  const startConversation = useCallback((datasetId: string, question: string) => {
    const q = question.trim();
    if (!datasetId || !q) return;
    sessionRef.current += 1;
    messagesRef.current = [];
    setActiveId(datasetId);
    setMessages([]);
    setChart(null);
    setError('');
    setInput('');
    setLoading(false);
    setView('workspace');
    void ask(q, datasetId);
  }, [ask]);

  const newConversation = () => {
    sessionRef.current += 1;
    messagesRef.current = [];
    setMessages([]);
    setChart(null);
    setError('');
    setInput('');
    setLoading(false);
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
      if (fileRef.current) fileRef.current.value = '';
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
          <div className="home-status">
            <span className={`status-dot ${health?.llm.configured ? 'online' : 'missing'}`} />
            {health?.llm.configured ? `在线模型 · ${health.llm.model}` : '未配置在线模型'}
          </div>
        </header>

        <main className="home-main">
          <section className="home-hero">
            <span className="hero-mark"><Icon name="spark" /></span>
            <h1>选好数据，直接提问</h1>
            <p>选择一份数据集，用自然语言描述你想看的内容；智能体会判断该出图还是用文字回答。</p>
          </section>

          {error && <div className="error-banner">{error}</div>}

          <section className="home-block">
            <div className="home-block-head">
              <h2>1. 选择数据</h2>
              <span>{datasets.length > 0 ? `${datasets.length} 个可用数据集` : '加载中…'}</span>
            </div>

            {groups.map(([group, items]) => (
              <div className="home-group" key={group}>
                <div className="home-group-title">{group}</div>
                <div className="home-datasets">
                  {items.map((dataset) => (
                    <button
                      type="button"
                      key={dataset.id}
                      className={`dataset-card ${dataset.id === activeId ? 'active' : ''}`}
                      onClick={() => selectDataset(dataset.id)}
                    >
                      <span className="dataset-icon"><Icon name="database" /></span>
                      <span className="dataset-card-body">
                        <span className="dataset-card-name">
                          {dataset.name}
                          {dataset.sample && <em>抽样</em>}
                        </span>
                        <span className="dataset-card-meta">
                          {formatCount(dataset.rowCount)} 行 · {dataset.columns.length} 列
                        </span>
                        {dataset.description && <span className="dataset-card-desc">{dataset.description}</span>}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ))}

            <div
              className={`home-upload ${uploading ? 'busy' : ''}`}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                void handleFiles(event.dataTransfer.files);
              }}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".csv,text/csv"
                hidden
                onChange={(event) => void handleFiles(event.target.files)}
              />
              <Icon name="upload" />
              <div>
                <strong>{uploading ? '正在解析…' : '上传自己的 CSV'}</strong>
                <span>拖拽文件到这里，或选择本地文件</span>
              </div>
              <button type="button" onClick={() => fileRef.current?.click()} disabled={uploading}>
                选择文件
              </button>
            </div>
          </section>

          <section className="home-block">
            <div className="home-block-head">
              <h2>2. 输入问题</h2>
              <span>{active ? `${active.name} · ${formatCount(active.rowCount)} 行` : '请先选择数据'}</span>
            </div>
            <form
              className="home-composer"
              onSubmit={(event) => {
                event.preventDefault();
                startConversation(activeId, input);
              }}
            >
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={active
                  ? `例如：${active.suggestedQuestions[0] ?? '画出数据的主要分布'}`
                  : '选择数据后，在这里输入问题'}
                disabled={!active || loading}
                autoFocus
              />
              <button type="submit" disabled={!activeId || !input.trim() || loading}>
                <Icon name="send" />
                开始分析
              </button>
            </form>
            {active && active.suggestedQuestions.length > 0 && (
              <div className="home-suggestions">
                {active.suggestedQuestions.map((question) => (
                  <button
                    type="button"
                    key={question}
                    onClick={() => startConversation(activeId, question)}
                    disabled={loading}
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}
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

        <div className="sidebar-scroll">
          {groups.map(([group, items]) => (
            <section className="dataset-group" key={group}>
              <div className="group-title">{group}</div>
              {items.map((dataset) => (
                <button
                  type="button"
                  key={dataset.id}
                  className={`dataset-item ${dataset.id === activeId ? 'active' : ''}`}
                  onClick={() => selectDataset(dataset.id)}
                >
                  <span className="dataset-icon"><Icon name="database" /></span>
                  <span className="dataset-main">
                    <span className="dataset-name">
                      {dataset.name}
                      {dataset.sample && <em>抽样</em>}
                    </span>
                    <span className="dataset-meta">
                      {formatCount(dataset.rowCount)} 行 · {dataset.columns.length} 列
                    </span>
                  </span>
                </button>
              ))}
            </section>
          ))}

          <div
            className={`upload-zone ${uploading ? 'busy' : ''}`}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              void handleFiles(event.dataTransfer.files);
            }}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              hidden
              onChange={(event) => void handleFiles(event.target.files)}
            />
            <Icon name="upload" />
            <div>
              <strong>{uploading ? '正在解析…' : '上传 CSV'}</strong>
              <span>也可以拖拽文件到这里</span>
            </div>
            <button type="button" onClick={() => fileRef.current?.click()} disabled={uploading}>
              选择文件
            </button>
          </div>
        </div>

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
            <h1>{active?.name ?? '选择数据集'}</h1>
            <p>{active?.description ?? '从左侧选择内置数据，或上传一份 CSV。'}</p>
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
            <div className="messages" ref={scrollRef}>
              {messages.length === 0 && (
                <div className="hero">
                  <span className="hero-mark"><Icon name="chart" /></span>
                  <h2>问数据，也聊思路</h2>
                  <p>在左侧选择数据集后，输入问题即可。</p>
                  {active && (
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

              {loading && (
                <div className="message assistant">
                  <div className="avatar"><Icon name="spark" /></div>
                  <div className="bubble thinking">
                    <span /><span /><span />
                    <em>正在理解问题…</em>
                  </div>
                </div>
              )}
            </div>

            {active && active.suggestedQuestions.length > 0 && (
              <div className="suggestions">
                {active.suggestedQuestions.map((question) => (
                  <button type="button" key={question} onClick={() => void ask(question)} disabled={loading}>
                    {question}
                  </button>
                ))}
              </div>
            )}

            <form className="composer" onSubmit={(event) => { event.preventDefault(); void ask(input); }}>
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={active ? `针对「${active.name}」提问，例如：${active.suggestedQuestions[0] ?? '画出数据的分布'}` : '请先选择数据集'}
                disabled={!active || loading}
              />
              <button type="submit" disabled={!active || loading || !input.trim()} aria-label="发送">
                <Icon name="send" />
              </button>
            </form>
          </div>

          <div className="chart-panel">
            <div className="chart-head">
              <div>
                <div className="chart-eyebrow">
                  <span>图表</span>
                  {chart && <PlannerBadge planner={chart.planner} />}
                </div>
                <h2>{chart?.spec?.title ?? '等待生成'}</h2>
                {chart && <p>{chart.summary}</p>}
              </div>
              {chart && (
                <button type="button" className="ghost-button" onClick={exportPng}>
                  <Icon name="download" /> PNG
                </button>
              )}
            </div>

            <div className="chart-body">
              {chart?.option ? (
                <ChartView
                  ref={chartRef}
                  option={chart.option as EChartsOption}
                />
              ) : (
                <div className="chart-empty">
                  <span className="empty-mark"><Icon name="chart" /></span>
                  <strong>图表会出现在这里</strong>
                  <p>模型判断需要可视化时会自动出图；纯文字回答只看左侧对话即可。</p>
                </div>
              )}
            </div>

            {chart?.spec && (
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
