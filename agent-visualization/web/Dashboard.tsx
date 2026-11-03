import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EChartsOption } from 'echarts';
import ChartView from './ChartView';
import type {
  DashboardAlert,
  DashboardDeviceDetail,
  DashboardDeviceSummary,
  DashboardPoint,
  DashboardSummary,
} from './types';

const API = '/api';
const LEVEL_COLORS = ['#4d9868', '#b8860b', '#d97706', '#c9363e'];
const LEVEL_SOFT = ['#ebf7f0', '#fdf6e3', '#fff1e8', '#fdf2f0'];

interface Props {
  onBack: () => void;
}

function levelColor(level: number): string {
  return LEVEL_COLORS[level] ?? '#838280';
}

function levelSoft(level: number): string {
  return LEVEL_SOFT[level] ?? '#f4f5f4';
}

function trendClass(trend: string): string {
  if (trend === '恶化') return 'down';
  if (trend === '改善') return 'up';
  return 'flat';
}

function applyPoint<T extends DashboardDeviceSummary>(device: T, point: DashboardPoint): T {
  const latestAlerts = point.alertMessage
    ? [
        {
          id: `${device.id}-${point.fileIdx}`,
          fileIdx: point.fileIdx,
          level: point.level,
          levelName: point.levelName,
          type: point.alertType || 'alert',
          message: point.alertMessage,
        },
        ...device.latestAlerts,
      ].slice(0, 5)
    : device.latestAlerts;
  return {
    ...device,
    fileIdx: point.fileIdx,
    score: point.score,
    level: point.level,
    levelName: point.levelName,
    stage: point.stage,
    stageName: point.stageName,
    relRul: point.relRul,
    trend: point.trend,
    rmsRatio: point.rmsRatio,
    envRatio: point.envRatio,
    alertCount: device.alertCount + (point.alertMessage ? 1 : 0),
    latestAlerts,
  } as T;
}

function readEvent(event: Event): Record<string, unknown> | null {
  try {
    return JSON.parse((event as MessageEvent).data) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function formatGeneratedAt(value?: string): string {
  if (!value) return '尚未生成';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export default function Dashboard({ onBack }: Props) {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [detail, setDetail] = useState<DashboardDeviceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [toast, setToast] = useState('');
  const [live, setLive] = useState(false);
  const [replay, setReplay] = useState({ running: false, fileIdx: 0, totalFiles: 0 });
  const [pushLoading, setPushLoading] = useState(false);
  const [refreshLoading, setRefreshLoading] = useState(false);

  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;

  const loadDetail = useCallback(async (id: string) => {
    const response = await fetch(`${API}/dashboard/device/${encodeURIComponent(id)}?limit=400`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '加载设备详情失败');
    setDetail(data as DashboardDeviceDetail);
  }, []);

  const loadSummary = useCallback(async (pick?: string) => {
    const response = await fetch(`${API}/dashboard/summary`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '加载看板失败');
    const payload = data as DashboardSummary;
    setSummary(payload);
    const nextId = pick ?? selectedRef.current ?? payload.devices[0]?.id ?? '';
    const valid = payload.devices.some((device) => device.id === nextId);
    setSelectedId(valid ? nextId : payload.devices[0]?.id ?? '');
  }, []);

  useEffect(() => {
    setLoading(true);
    loadSummary()
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => setLoading(false));
  }, [loadSummary]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setDetail(null);
    loadDetail(selectedId).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  }, [selectedId, loadDetail]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(''), 4500);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    const source = new EventSource(`${API}/dashboard/stream`);
    source.addEventListener('hello', () => setLive(true));

    source.addEventListener('summary', (event) => {
      const data = readEvent(event);
      if (!data) return;
      setSummary((previous) => (previous ? { ...previous, summary: data as DashboardSummary['summary'] } : previous));
    });

    source.addEventListener('update', (event) => {
      const data = readEvent(event);
      if (!data) return;
      const deviceId = String(data.deviceId ?? '');
      const point = data.point as DashboardPoint;
      if (!deviceId || !point) return;
      setSummary((previous) => {
        if (!previous) return previous;
        return {
          ...previous,
          summary: (data.summary as DashboardSummary['summary']) ?? previous.summary,
          devices: previous.devices.map((device) =>
            device.id === deviceId ? applyPoint(device, point) : device,
          ),
        };
      });
      setDetail((previous) => {
        if (!previous || previous.device.id !== deviceId) return previous;
        const history = [...previous.history, point].slice(-400);
        return { ...previous, device: applyPoint(previous.device, point), history };
      });
      setReplay({
        running: true,
        fileIdx: Number(data.fileIdx ?? point.fileIdx),
        totalFiles: Number(data.totalFiles ?? 0),
      });
    });

    source.addEventListener('alert', (event) => {
      const data = readEvent(event);
      if (!data) return;
      const alert = data.alert as DashboardAlert;
      if (!alert) return;
      setDetail((previous) => {
        if (!previous || previous.device.id !== String(data.deviceId ?? '')) return previous;
        const exists = previous.alerts.some((item) => item.id === alert.id);
        return exists ? previous : { ...previous, alerts: [alert, ...previous.alerts].slice(0, 30) };
      });
      setToast(`新告警：${alert.message}`);
    });

    source.addEventListener('replay', (event) => {
      const data = readEvent(event);
      if (!data) return;
      const running = Boolean(data.running);
      setReplay({
        running,
        fileIdx: Number(data.fileIdx ?? 0),
        totalFiles: Number(data.totalFiles ?? 0),
      });
      if (running && Number(data.fileIdx ?? 0) === 0) {
        const deviceId = String(data.deviceId ?? '');
        setDetail((previous) =>
          previous && previous.device.id === deviceId
            ? { ...previous, history: [], alerts: [] }
            : previous,
        );
        setSummary((previous) =>
          previous
            ? {
                ...previous,
                devices: previous.devices.map((device) =>
                  device.id === deviceId ? { ...device, alertCount: 0, latestAlerts: [] } : device,
                ),
              }
            : previous,
        );
      }
    });

    source.onerror = () => setLive(false);
    return () => source.close();
  }, []);

  const selected = summary?.devices.find((device) => device.id === selectedId) ?? null;
  const latest = detail?.history[detail.history.length - 1] ?? null;
  const point = latest ?? (detail
    ? {
        fileIdx: detail.device.fileIdx,
        score: detail.device.score,
        level: detail.device.level,
        levelName: detail.device.levelName,
        stage: detail.device.stage,
        stageName: detail.device.stageName,
        relRul: detail.device.relRul,
        trend: detail.device.trend,
        rmsRatio: detail.device.rmsRatio,
        envRatio: detail.device.envRatio,
        rmsScore: 0,
        kurtScore: 0,
        envScore: 0,
        stageScore: 0,
        rulScore: 0,
        rms: 0,
        kurtosis: 0,
        envPeak: 0,
        explanation: '',
        alertType: '',
        alertMessage: '',
      }
    : null);

  const chartOption = useMemo<EChartsOption>(() => {
    const history = detail?.history ?? [];
    return {
      tooltip: { trigger: 'axis' },
      grid: { top: 18, right: 18, bottom: 30, left: 42 },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: history.map((item) => `#${item.fileIdx}`),
        axisLabel: { color: '#838280', fontSize: 10, interval: Math.max(0, Math.floor(history.length / 10) - 1) },
        axisLine: { lineStyle: { color: '#d5dbd8' } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: '#838280', fontSize: 10 },
        splitLine: { lineStyle: { color: '#eef0ee' } },
      },
      series: [
        {
          name: '健康分',
          type: 'line',
          data: history.map((item) => item.score),
          smooth: true,
          symbol: 'none',
          lineStyle: { color: '#4d9868', width: 2 },
          areaStyle: { color: 'rgba(77,152,104,0.12)' },
          markLine: {
            silent: true,
            symbol: 'none',
            label: { color: '#838280', fontSize: 9, formatter: '{b}' },
            data: [
              { yAxis: 85, name: '健康线', lineStyle: { color: '#4d9868', type: 'dashed' } },
              { yAxis: 70, name: '注意线', lineStyle: { color: '#b8860b', type: 'dashed' } },
              { yAxis: 50, name: '警告线', lineStyle: { color: '#c9363e', type: 'dashed' } },
            ],
          },
        },
      ],
    } as EChartsOption;
  }, [detail]);

  async function handleReplay() {
    if (!selectedId) return;
    setError('');
    try {
      const response = await fetch(`${API}/dashboard/replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deviceId: selectedId, intervalMs: 500 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '启动回放失败');
      setDetail((previous) => (previous ? { ...previous, history: [], alerts: [] } : previous));
      setSummary((previous) =>
        previous
          ? {
              ...previous,
              devices: previous.devices.map((device) =>
                device.id === selectedId ? { ...device, alertCount: 0, latestAlerts: [] } : device,
              ),
            }
          : previous,
      );
      setReplay({ running: true, fileIdx: 0, totalFiles: Number(data.totalFiles ?? 0) });
      setToast(`开始回放 ${selectedId.split('/').pop()}，共 ${data.totalFiles} 个文件`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function handleStop() {
    try {
      await fetch(`${API}/dashboard/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
      setReplay((previous) => ({ ...previous, running: false }));
      if (selectedId) await loadDetail(selectedId);
      await loadSummary(selectedId);
      setToast('已停止回放');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function handlePush() {
    if (!selectedId) return;
    setPushLoading(true);
    setError('');
    try {
      const response = await fetch(`${API}/dashboard/push`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deviceId: selectedId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '推送失败');
      setToast(data.message || '已推送');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setPushLoading(false);
    }
  }

  async function handleRefresh() {
    setRefreshLoading(true);
    setError('');
    try {
      const response = await fetch(`${API}/dashboard/refresh`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '刷新评分失败');
      setSummary(data.summary as DashboardSummary);
      if (selectedId) await loadDetail(selectedId);
      setToast('评分已重新生成');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRefreshLoading(false);
    }
  }

  const activeLevel = point?.level ?? selected?.level ?? 0;

  return (
    <div className="dashboard">
      <header className="dash-top">
        <div className="dash-top-left">
          <button type="button" className="dash-back" onClick={onBack}>
            ← 返回
          </button>
          <div>
            <h1>设备健康看板</h1>
            <p>
              {summary ? `${summary.summary.total} 台轴承 · 0–100 分 · ${formatGeneratedAt(summary.generatedAt)}` : '正在加载设备评分'}
            </p>
          </div>
        </div>
        <div className="dash-top-right">
          <span className={`dash-live ${live ? 'on' : 'off'}`}>
            <i />
            {live ? '实时推送已连接' : '实时推送断开'}
          </span>
          <button type="button" className="dash-button" onClick={() => void handleRefresh()} disabled={refreshLoading}>
            {refreshLoading ? '刷新中…' : '刷新评分'}
          </button>
          <button
            type="button"
            className="dash-button primary"
            onClick={() => void handlePush()}
            disabled={!selectedId || pushLoading}
          >
            {pushLoading ? '推送中…' : '推送当前告警'}
          </button>
        </div>
      </header>

      {error && <div className="dash-error">{error}</div>}
      {toast && <div className="dash-toast">{toast}</div>}

      {loading && !summary ? (
        <div className="dash-loading">正在读取设备评分…</div>
      ) : !summary ? (
        <div className="dash-loading">暂无可用的设备评分。</div>
      ) : (
        <>
          <section className="dash-stats">
            <div className="dash-stat total">
              <span>设备总数</span>
              <strong>{summary.summary.total}</strong>
            </div>
            <div className="dash-stat healthy">
              <span>健康</span>
              <strong>{summary.summary.healthy}</strong>
            </div>
            <div className="dash-stat attention">
              <span>注意</span>
              <strong>{summary.summary.attention}</strong>
            </div>
            <div className="dash-stat warning">
              <span>警告</span>
              <strong>{summary.summary.warning}</strong>
            </div>
            <div className="dash-stat danger">
              <span>危险</span>
              <strong>{summary.summary.danger}</strong>
            </div>
            <p className="dash-stats-note">
              全寿命试验均已运行至失效，末端状态多为危险；选择设备后点击“回放”查看从健康到失效的退化过程。
            </p>
          </section>

          <div className="dash-body">
            <aside className="dash-devices">
              <div className="dash-devices-head">
                <strong>设备列表</strong>
                <span>{summary.devices.length} 台</span>
              </div>
              <div className="dash-device-list">
                {summary.devices.map((device) => (
                  <button
                    type="button"
                    key={device.id}
                    className={`dash-device ${device.id === selectedId ? 'active' : ''}`}
                    onClick={() => setSelectedId(device.id)}
                  >
                    <div className="dash-device-top">
                      <strong>{device.name}</strong>
                      <span className={`dash-badge lv-${device.level}`}>{device.levelName}</span>
                    </div>
                    <div className="dash-device-meta">
                      <span>{device.condition}</span>
                      <span>#{device.fileIdx}/{device.totalFiles}</span>
                    </div>
                    <div className="dash-device-score">
                      <em style={{ color: levelColor(device.level) }}>{device.score.toFixed(1)}</em>
                      <span className={`dash-trend ${trendClass(device.trend)}`}>{device.trend}</span>
                      {device.alertCount > 0 && <span className="dash-alert-count">{device.alertCount} 条告警</span>}
                    </div>
                  </button>
                ))}
              </div>
            </aside>

            <main className="dash-main">
              <section className="dash-hero">
                <div
                  className="dash-ring"
                  style={{
                    '--dash-score': `${Math.max(0, Math.min(100, point?.score ?? 0)) * 3.6}deg`,
                    '--dash-color': levelColor(activeLevel),
                  } as React.CSSProperties}
                >
                  <div className="dash-ring-inner">
                    <strong>{point ? point.score.toFixed(0) : '--'}</strong>
                    <span>分</span>
                  </div>
                </div>
                <div className="dash-hero-info">
                  <div className="dash-hero-title">
                    <h2>{selected?.name ?? detail?.device.name ?? '选择设备'}</h2>
                    <span className={`dash-badge big lv-${activeLevel}`}>{point?.levelName ?? selected?.levelName ?? '--'}</span>
                  </div>
                  <p className="dash-hero-sub">
                    {selected?.condition ?? detail?.device.condition ?? ''} · 全寿命文件 #{point?.fileIdx ?? 0}/
                    {selected?.totalFiles ?? detail?.device.totalFiles ?? 0}
                  </p>
                  <div className="dash-metrics">
                    <div>
                      <span>健康阶段</span>
                      <strong>{point?.stageName ?? '--'}</strong>
                    </div>
                    <div>
                      <span>相对 RUL</span>
                      <strong>{point ? point.relRul.toFixed(2) : '--'}</strong>
                    </div>
                    <div>
                      <span>RMS 倍数</span>
                      <strong>{point ? `${point.rmsRatio.toFixed(1)}×` : '--'}</strong>
                    </div>
                    <div>
                      <span>包络倍数</span>
                      <strong>{point ? `${point.envRatio.toFixed(1)}×` : '--'}</strong>
                    </div>
                  </div>
                  <div className="dash-actions">
                    <button type="button" className="dash-button primary" onClick={() => void handleReplay()} disabled={!selectedId || replay.running}>
                      回放退化过程
                    </button>
                    <button type="button" className="dash-button" onClick={() => void handleStop()} disabled={!replay.running}>
                      停止回放
                    </button>
                    <span className="dash-progress">
                      {replay.running ? `回放中 ${replay.fileIdx}/${replay.totalFiles}` : `当前观测点 ${point?.fileIdx ?? 0}/${selected?.totalFiles ?? 0}`}
                    </span>
                  </div>
                </div>
              </section>

              <section className="dash-card">
                <div className="dash-card-head">
                  <h3>健康分数趋势</h3>
                  <span>标记线：85 健康 / 70 注意 / 50 警告</span>
                </div>
                {(detail?.history.length ?? 0) > 0 ? (
                  <div className="dash-chart-box">
                    <ChartView option={chartOption} />
                  </div>
                ) : (
                  <div className="dash-empty">点击“回放退化过程”查看评分随时间的变化。</div>
                )}
              </section>

              <div className="dash-two-col">
                <section className="dash-card">
                  <div className="dash-card-head">
                    <h3>最近告警</h3>
                    <span>{detail?.alerts.length ?? 0} 条</span>
                  </div>
                  <div className="dash-alert-list">
                    {detail?.alerts.length ? (
                      detail.alerts.slice(0, 12).map((alert) => (
                        <div key={alert.id} className={`dash-alert-item lv-${alert.level}`}>
                          <span className="dash-alert-level">{alert.levelName}</span>
                          <div>
                            <strong>文件 #{alert.fileIdx}</strong>
                            <p>{alert.message}</p>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="dash-empty">当前观测点暂无告警。</div>
                    )}
                  </div>
                </section>

                <section className="dash-card">
                  <div className="dash-card-head">
                    <h3>评分依据</h3>
                    <span>规则加权</span>
                  </div>
                  {point ? (
                    <div className="dash-explain">
                      {[
                        ['RMS', point.rmsScore, '30%'],
                        ['峭度', point.kurtScore, '20%'],
                        ['包络峰值', point.envScore, '20%'],
                        ['健康阶段', point.stageScore, '20%'],
                        ['相对 RUL', point.rulScore, '10%'],
                      ].map(([label, value, weight]) => (
                        <div key={String(label)} className="dash-component">
                          <div className="dash-component-head">
                            <span>{label}</span>
                            <em>{Number(value).toFixed(0)} 分 · 权重 {weight}</em>
                          </div>
                          <div className="dash-bar">
                            <i style={{ width: `${Math.max(0, Math.min(100, Number(value)))}%` }} />
                          </div>
                        </div>
                      ))}
                      <p className="dash-explain-text">{point.explanation || '暂无评分依据。'}</p>
                    </div>
                  ) : (
                    <div className="dash-empty">选择设备后显示评分依据。</div>
                  )}
                </section>
              </div>
            </main>
          </div>
        </>
      )}
    </div>
  );
}