import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import type { Response } from 'express';
import { parse } from 'csv-parse/sync';
import { PROJECT_ROOT } from './datasets';

const execFileAsync = promisify(execFile);

const SCORE_CSV = path.join(PROJECT_ROOT, 'results', 'health_scores.csv');
const FEATURES_CSV = path.join(PROJECT_ROOT, 'results', 'week1', 'features.csv');
const SCORE_SCRIPT = path.join(PROJECT_ROOT, 'src', 'agent-maintenance', 'score_devices.py');

const STAGE_NAMES: Record<number, string> = { 0: '健康', 1: '退化', 2: '故障' };
const LEVEL_NAMES: Record<number, string> = { 0: '健康', 1: '注意', 2: '警告', 3: '危险' };

export interface DashboardPoint {
  fileIdx: number;
  score: number;
  level: number;
  levelName: string;
  stage: number;
  stageName: string;
  relRul: number;
  rms: number;
  kurtosis: number;
  envPeak: number;
  trend: string;
  rmsRatio: number;
  envRatio: number;
  rmsScore: number;
  kurtScore: number;
  envScore: number;
  stageScore: number;
  rulScore: number;
  explanation: string;
  alertType: string;
  alertMessage: string;
}

export interface DashboardAlert {
  id: string;
  fileIdx: number;
  level: number;
  levelName: string;
  type: string;
  message: string;
}

export interface DashboardDevice {
  id: string;
  name: string;
  condition: string;
  totalFiles: number;
  history: DashboardPoint[];
  alerts: DashboardAlert[];
}

export interface DashboardDeviceSummary {
  id: string;
  name: string;
  condition: string;
  totalFiles: number;
  fileIdx: number;
  score: number;
  level: number;
  levelName: string;
  stage: number;
  stageName: string;
  relRul: number;
  trend: string;
  rmsRatio: number;
  envRatio: number;
  alertCount: number;
  latestAlerts: DashboardAlert[];
}

export interface DashboardSummary {
  ok: true;
  generatedAt: string;
  summary: {
    total: number;
    healthy: number;
    attention: number;
    warning: number;
    danger: number;
  };
  devices: DashboardDeviceSummary[];
}

export class DashboardError extends Error {
  status: number;

  constructor(message: string, status = 500) {
    super(message);
    this.status = status;
  }
}

function num(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : String(value ?? '');
}

function pythonBin(): string {
  if (process.env.VIZ_PYTHON) return process.env.VIZ_PYTHON;
  const venv = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  return fs.existsSync(venv) ? venv : 'python3';
}

let cache: DashboardDevice[] | null = null;
let cacheMtime = 0;
let refreshPromise: Promise<string> | null = null;

function readDevices(): DashboardDevice[] {
  if (!fs.existsSync(SCORE_CSV)) {
    throw new DashboardError(
      '尚未生成设备评分，请先运行 scripts 或点击“刷新评分”。',
      503,
    );
  }
  const mtime = fs.statSync(SCORE_CSV).mtimeMs;
  if (cache && mtime === cacheMtime) return cache;

  const rows = parse(fs.readFileSync(SCORE_CSV, 'utf8'), {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as Record<string, string>[];

  const groups = new Map<string, DashboardDevice>();
  for (const row of rows) {
    const id = text(row.bearing_key);
    if (!id) continue;
    let device = groups.get(id);
    if (!device) {
      device = {
        id,
        name: text(row.bearing),
        condition: text(row.condition),
        totalFiles: 0,
        history: [],
        alerts: [],
      };
      groups.set(id, device);
    }
    const stage = num(row.stage);
    const level = num(row.level);
    const point: DashboardPoint = {
      fileIdx: num(row.file_idx),
      score: num(row.score),
      level,
      levelName: text(row.level_name) || LEVEL_NAMES[level] || '未知',
      stage,
      stageName: STAGE_NAMES[stage] ?? '未知',
      relRul: num(row.rel_rul),
      rms: num(row.rms_mean),
      kurtosis: num(row.kurtosis_v),
      envPeak: num(row.env_peak),
      trend: text(row.trend) || '平稳',
      rmsRatio: num(row.rms_ratio),
      envRatio: num(row.env_ratio),
      rmsScore: num(row.rms_score),
      kurtScore: num(row.kurt_score),
      envScore: num(row.env_score),
      stageScore: num(row.stage_score),
      rulScore: num(row.rul_score),
      explanation: text(row.explanation),
      alertType: text(row.alert_type),
      alertMessage: text(row.alert_message),
    };
    device.history.push(point);
    if (point.alertMessage) {
      device.alerts.push({
        id: `${id}-${point.fileIdx}`,
        fileIdx: point.fileIdx,
        level: point.level,
        levelName: point.levelName,
        type: point.alertType || 'alert',
        message: point.alertMessage,
      });
    }
  }

  cache = [...groups.values()]
    .map((device) => {
      device.history.sort((a, b) => a.fileIdx - b.fileIdx);
      device.totalFiles = device.history.length;
      return device;
    })
    .sort((a, b) => a.id.localeCompare(b.id));
  cacheMtime = mtime;
  return cache;
}

async function ensureScores(): Promise<void> {
  if (!fs.existsSync(SCORE_CSV)) {
    await refreshScores();
    return;
  }
  if (fs.existsSync(FEATURES_CSV)) {
    const scoreTime = fs.statSync(SCORE_CSV).mtimeMs;
    const featureTime = fs.statSync(FEATURES_CSV).mtimeMs;
    if (featureTime > scoreTime) await refreshScores();
  }
}

export async function refreshScores(): Promise<string> {
  if (!fs.existsSync(SCORE_SCRIPT)) {
    throw new DashboardError(`未找到打分脚本：${path.relative(PROJECT_ROOT, SCORE_SCRIPT)}`, 503);
  }
  if (!fs.existsSync(FEATURES_CSV)) {
    throw new DashboardError(
      '未找到 results/week1/features.csv，请先运行第 1 周特征提取。',
      503,
    );
  }
  if (refreshPromise) return refreshPromise;

  refreshPromise = (async () => {
    try {
      const { stdout, stderr } = await execFileAsync(pythonBin(), [SCORE_SCRIPT], {
        cwd: PROJECT_ROOT,
        timeout: 120_000,
        maxBuffer: 32 * 1024 * 1024,
      });
      cache = null;
      cacheMtime = 0;
      return (stdout || stderr).trim();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new DashboardError(`设备评分生成失败：${message.slice(0, 240)}`, 500);
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

function summaryOf(device: DashboardDevice, point: DashboardPoint): DashboardDeviceSummary {
  return {
    id: device.id,
    name: device.name,
    condition: device.condition,
    totalFiles: device.totalFiles,
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
    alertCount: device.alerts.filter((alert) => alert.fileIdx <= point.fileIdx).length,
    latestAlerts: device.alerts
      .filter((alert) => alert.fileIdx <= point.fileIdx)
      .slice(-5),
  };
}

const livePoints = new Map<string, DashboardPoint>();

function allSummaries(): DashboardSummary {
  const devices = readDevices();
  const counts = { healthy: 0, attention: 0, warning: 0, danger: 0 };
  const summaries = devices.map((device) => {
    const point = livePoints.get(device.id) ?? device.history[device.history.length - 1];
    if (point.level === 0) counts.healthy += 1;
    else if (point.level === 1) counts.attention += 1;
    else if (point.level === 2) counts.warning += 1;
    else counts.danger += 1;
    return summaryOf(device, point);
  });
  summaries.sort((a, b) => b.level - a.level || a.score - b.score);
  return {
    ok: true,
    generatedAt: new Date(fs.statSync(SCORE_CSV).mtimeMs).toISOString(),
    summary: { total: summaries.length, ...counts },
    devices: summaries,
  };
}

function summaryCounts(): DashboardSummary['summary'] {
  const devices = readDevices();
  const counts = { total: devices.length, healthy: 0, attention: 0, warning: 0, danger: 0 };
  for (const device of devices) {
    const point = livePoints.get(device.id) ?? device.history[device.history.length - 1];
    if (point.level === 0) counts.healthy += 1;
    else if (point.level === 1) counts.attention += 1;
    else if (point.level === 2) counts.warning += 1;
    else counts.danger += 1;
  }
  return counts;
}

export async function getDashboardSummary(): Promise<DashboardSummary> {
  await ensureScores();
  return allSummaries();
}

export async function getDashboardDevice(id: string, limit = 400): Promise<{
  ok: true;
  device: DashboardDeviceSummary;
  history: DashboardPoint[];
  alerts: DashboardAlert[];
}> {
  await ensureScores();
  const device = readDevices().find((item) => item.id === id);
  if (!device) throw new DashboardError('设备不存在', 404);
  const point = livePoints.get(device.id) ?? device.history[device.history.length - 1];
  return {
    ok: true,
    device: summaryOf(device, point),
    history: device.history.slice(-Math.max(20, Math.min(limit, 2000))),
    alerts: device.alerts.slice(-30).reverse(),
  };
}

export function dashboardInfo(): {
  configured: boolean;
  devices: number;
  webhook: boolean;
  generatedAt: string | null;
} {
  let devices = 0;
  let generatedAt: string | null = null;
  try {
    const items = readDevices();
    devices = items.length;
    generatedAt = fs.statSync(SCORE_CSV).mtime.toISOString();
  } catch {
    // 未生成评分时不阻塞 /api/health
  }
  return {
    configured: fs.existsSync(SCORE_CSV),
    devices,
    webhook: Boolean(process.env.VIZ_ALERT_WEBHOOK?.trim()),
    generatedAt,
  };
}

/* ── SSE ── */
const clients = new Set<Response>();

function publish(event: string, data: unknown): void {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const client of clients) {
    try {
      client.write(payload);
    } catch {
      clients.delete(client);
    }
  }
}

export function subscribeDashboard(res: Response): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write(`event: hello\ndata: ${JSON.stringify({ ok: true })}\n\n`);
  clients.add(res);
  const keepAlive = setInterval(() => {
    try {
      res.write(': keep-alive\n\n');
    } catch {
      clearInterval(keepAlive);
      clients.delete(res);
    }
  }, 15_000);
  res.on('close', () => {
    clearInterval(keepAlive);
    clients.delete(res);
  });
}

/* ── 回放 ── */
let replayTimer: NodeJS.Timeout | null = null;
let replayDeviceId: string | null = null;
let replayCursor = 0;
let replayTotal = 0;

export function stopDashboardReplay(notify = true): void {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
  }
  if (replayDeviceId) {
    livePoints.delete(replayDeviceId);
    if (notify) {
      publish('replay', {
        deviceId: replayDeviceId,
        running: false,
        fileIdx: replayCursor,
        totalFiles: replayTotal,
      });
      publish('summary', summaryCounts());
    }
  }
  replayDeviceId = null;
  replayCursor = 0;
  replayTotal = 0;
}

export async function startDashboardReplay(deviceId: string, intervalMs = 700): Promise<{
  ok: true;
  deviceId: string;
  intervalMs: number;
  totalFiles: number;
}> {
  await ensureScores();
  const device = readDevices().find((item) => item.id === deviceId);
  if (!device) throw new DashboardError('设备不存在', 404);
  stopDashboardReplay(false);
  replayDeviceId = deviceId;
  replayCursor = 0;
  replayTotal = device.history.length;
  const interval = Math.max(150, Math.min(intervalMs, 5000));

  publish('replay', {
    deviceId,
    running: true,
    fileIdx: 0,
    totalFiles: replayTotal,
    intervalMs: interval,
  });

  replayTimer = setInterval(() => {
    try {
      const current = readDevices().find((item) => item.id === deviceId);
      if (!current || replayCursor >= current.history.length) {
        const finished = replayDeviceId;
        const finishedTotal = replayTotal;
        const finishedCursor = replayCursor;
        stopDashboardReplay(false);
        if (finished) {
          publish('replay', {
            deviceId: finished,
            running: false,
            fileIdx: finishedCursor,
            totalFiles: finishedTotal,
          });
          publish('summary', summaryCounts());
        }
        return;
      }
      const point = current.history[replayCursor];
      replayCursor += 1;
      livePoints.set(deviceId, point);
      publish('update', {
        deviceId,
        point,
        summary: summaryCounts(),
        fileIdx: replayCursor,
        totalFiles: replayTotal,
      });
      if (point.alertMessage) {
        publish('alert', {
          deviceId,
          alert: {
            id: `${deviceId}-${point.fileIdx}`,
            fileIdx: point.fileIdx,
            level: point.level,
            levelName: point.levelName,
            type: point.alertType || 'alert',
            message: point.alertMessage,
          },
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      publish('error', { deviceId, message });
      stopDashboardReplay();
    }
  }, interval);

  return { ok: true, deviceId, intervalMs: interval, totalFiles: replayTotal };
}

/* ── 机器人推送 ── */
export async function pushDeviceAlert(deviceId: string): Promise<{
  ok: true;
  message: string;
  preview: string;
}> {
  await ensureScores();
  const device = readDevices().find((item) => item.id === deviceId);
  if (!device) throw new DashboardError('设备不存在', 404);
  const point = livePoints.get(device.id) ?? device.history[device.history.length - 1];
  const alerts = device.alerts.filter((alert) => alert.fileIdx <= point.fileIdx).slice(-5);
  const webhook = process.env.VIZ_ALERT_WEBHOOK?.trim();
  if (!webhook) {
    throw new DashboardError(
      '未配置 VIZ_ALERT_WEBHOOK。请在 .env 填入钉钉/企业微信机器人 Webhook 后重启服务。',
      503,
    );
  }

  const lines = [
    '### 设备健康告警',
    `- 设备：${device.name}（${device.condition}）`,
    `- 健康分：${point.score.toFixed(2)} / 100（${point.levelName}）`,
    `- 阶段：${point.stageName}；相对 RUL：${point.relRul.toFixed(2)}；趋势：${point.trend}`,
    `- 进度：第 ${point.fileIdx} / ${device.totalFiles} 个文件`,
    `- 评分依据：${point.explanation}`,
    alerts.length
      ? `- 最近告警：\n${alerts.map((alert) => `  - 文件 ${alert.fileIdx}：${alert.message}`).join('\n')}`
      : '- 最近告警：无',
    '',
    '> 来自工业数据分析智能体 · 设备健康看板',
  ];
  const preview = lines.join('\n');

  const isWeCom = /qyapi\.weixin\.qq\.com/i.test(webhook);
  const payload = isWeCom
    ? { msgtype: 'markdown', markdown: { content: preview } }
    : { msgtype: 'markdown', markdown: { title: '设备健康告警', text: preview } };

  let response: globalThis.Response;
  try {
    response = await fetch(webhook, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(8000),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new DashboardError(`推送失败：${message.slice(0, 160)}`, 502);
  }
  const body = await response.text();
  if (!response.ok) {
    throw new DashboardError(`推送失败：HTTP ${response.status} ${body.slice(0, 160)}`, 502);
  }
  try {
    const parsed = JSON.parse(body) as { errcode?: number; errmsg?: string; code?: number; msg?: string };
    const code = parsed.errcode ?? parsed.code;
    if (typeof code === 'number' && code !== 0) {
      throw new DashboardError(`推送失败：${parsed.errmsg || parsed.msg || `错误码 ${code}`}`, 502);
    }
  } catch (error) {
    if (error instanceof DashboardError) throw error;
    // 非 JSON 响应（例如自建网关返回纯文本）视为 HTTP 2xx 成功
  }
  return { ok: true, message: '已推送到机器人', preview };
}