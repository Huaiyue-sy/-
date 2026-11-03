import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
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

function PlannerBadge({ planner }: { planner?: 'llm' | 'rules' }) {
  if (!planner) return null;
  return (
    <span className={`badge ${planner}`}>
      {planner === 'llm' ? 'LLM 规划' : '规则引擎'}
    </span>
  );
}

function formatCount(n: number): string {
  return n.toLocaleString('zh-CN');
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
  const fileRef = useRef<HTMLInputElement>(null);
  const chartRef = useRef<ReactECharts>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoAsked = useRef(false);

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
    setActiveId(id);
    setMessages([]);
    setChart(null);
    setError('');
  };

  const ask = useCallback(async (question: string) => {
    const q = question.trim();
    if (!q || !activeId || loading) return;
    setMessages((prev) => [...prev, { id: uid(), role: 'user', text: q }]);
    setInput('');
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datasetId: activeId, question: q }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const result = data as ChatResponse;
      setChart(result);
      setMessages((prev) => [...prev, {
        id: uid(),
        role: 'assistant',
        text: result.answer,
        planner: result.planner,
        warning: result.warning,
        summary: result.summary,
      }]);
    } catch (err) {
      setMessages((prev) => [...prev, {
        id: uid(),
        role: 'assistant',
        text: `生成失败：${err instanceof Error ? err.message : String(err)}`,
      }]);
    } finally {
      setLoading(false);
    }
  }, [activeId, loading]);

  useEffect(() => {
    const question = new URLSearchParams(window.location.search).get('q');
    if (question && activeId && !autoAsked.current) {
      autoAsked.current = true;
      void ask(question);
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
    const instance = chartRef.current?.getEchartsInstance();
    if (!instance || !chart) return;
    const url = instance.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#07111e' });
    const link = document.createElement('a');
    link.href = url;
    link.download = `${(chart.spec.title || 'chart').replace(/[\\/:*?"<>|]/g, '_')}.png`;
    link.click();
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark"><Icon name="spark" /></span>
          <div>
            <strong>工业数据可视化</strong>
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
          <span className={`status-dot ${health?.llm.configured ? 'online' : 'offline'}`} />
          <div>
            <strong>{health?.llm.configured ? 'LLM 已配置' : '规则引擎'}</strong>
            <span>{health?.llm.configured ? `${health.llm.model} · 失败自动降级` : '完全离线运行'}</span>
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
            <button type="button" className="ghost-button" onClick={() => { setMessages([]); setChart(null); setError(''); }}>
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
                  <h2>用一句话，把数据变成图</h2>
                  <p>选择一个数据集，直接问“画出……”“对比……”“按……分布”。后端会先让大模型生成图表规格，字段不合法时自动改用规则引擎。</p>
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
                  <div className="avatar">{message.role === 'user' ? '你' : 'A'}</div>
                  <div className="bubble-wrap">
                    <div className="bubble">
                      {message.role === 'assistant' && <PlannerBadge planner={message.planner} />}
                      <p>{message.text}</p>
                      {message.summary && <div className="message-summary">{message.summary}</div>}
                      {message.warning && <div className="message-warning">{message.warning}</div>}
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message assistant">
                  <div className="avatar">A</div>
                  <div className="bubble thinking">
                    <span /><span /><span />
                    <em>正在理解问题并生成图表规格…</em>
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
                <h2>{chart?.spec.title ?? '等待生成'}</h2>
                {chart && <p>{chart.summary}</p>}
              </div>
              {chart && (
                <button type="button" className="ghost-button" onClick={exportPng}>
                  <Icon name="download" /> PNG
                </button>
              )}
            </div>

            <div className="chart-body">
              {chart ? (
                <ReactECharts
                  ref={chartRef}
                  option={chart.option as EChartsOption}
                  notMerge
                  lazyUpdate
                  style={{ width: '100%', height: '100%' }}
                />
              ) : (
                <div className="chart-empty">
                  <span className="empty-mark"><Icon name="chart" /></span>
                  <strong>第一张图，等你发问</strong>
                  <p>试试左侧数据集自带的建议问题，或上传自己的 CSV。</p>
                </div>
              )}
            </div>

            {chart && (
              <div className="chart-foot">
                <span>使用 {formatCount(chart.usedRows)} 行</span>
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
