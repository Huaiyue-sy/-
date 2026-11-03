import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { APP_DIR, PROJECT_ROOT } from './datasets';

const execFileAsync = promisify(execFile);

const CLASS_CN: Record<string, string> = {
  crazing: '龟裂',
  inclusion: '夹杂',
  patches: '斑块',
  pitted_surface: '麻点',
  rolled_in_scale: '氧化铁皮压入',
  scratches: '划伤',
};

export const DEFAULT_CONFIDENCE = 0.25;

export interface Detection {
  className: string;
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface VisionDetection {
  detections: Detection[];
  width: number;
  height: number;
  elapsedMs: number;
  annotatedImage: string;
  model: string;
  conf: number;
}

export interface DetectionCount {
  className: string;
  label: string;
  count: number;
  maxConfidence: number;
}

export interface VisionInfo {
  configured: boolean;
  model: string;
  python: string;
  confidence: number;
}

function yoloScriptPath(): string {
  return path.join(APP_DIR, 'python', 'yolo_detect.py');
}

export function visionModelPath(): string {
  const raw = process.env.VIZ_YOLO_MODEL;
  if (raw) return path.isAbsolute(raw) ? raw : path.join(PROJECT_ROOT, raw);
  return path.join(PROJECT_ROOT, 'results', 'week2_3', 'yolo', 'train', 'weights', 'best.pt');
}

export function pythonBin(): string {
  if (process.env.VIZ_PYTHON) return process.env.VIZ_PYTHON;
  const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  return fs.existsSync(venvPython) ? venvPython : 'python3';
}

export function confidenceThreshold(): number {
  const raw = Number(process.env.VIZ_YOLO_CONF ?? DEFAULT_CONFIDENCE);
  if (!Number.isFinite(raw)) return DEFAULT_CONFIDENCE;
  return Math.min(0.95, Math.max(0.01, raw));
}

export function visionConfigured(): boolean {
  return fs.existsSync(yoloScriptPath()) && fs.existsSync(visionModelPath());
}

export function visionInfo(): VisionInfo {
  return {
    configured: visionConfigured(),
    model: path.relative(PROJECT_ROOT, visionModelPath()),
    python: pythonBin(),
    confidence: confidenceThreshold(),
  };
}

interface RawDetection {
  class_name?: unknown;
  confidence?: unknown;
  box?: unknown;
}

interface RawOutput {
  ok?: boolean;
  error?: string;
  detections?: RawDetection[];
  width?: number;
  height?: number;
  elapsed_ms?: number;
  annotated?: string;
}

function parseOutput(stdout: string): RawOutput | null {
  const lines = stdout.trim().split(/\r?\n/).reverse();
  for (const line of lines) {
    const text = line.trim();
    if (!text.startsWith('{')) continue;
    try {
      return JSON.parse(text) as RawOutput;
    } catch {
      // 继续尝试上一行
    }
  }
  return null;
}

function normalizeDetection(raw: RawDetection): Detection | null {
  if (typeof raw.class_name !== 'string' || !raw.class_name) return null;
  const confidence = Number(raw.confidence);
  const box = Array.isArray(raw.box) ? raw.box.map(Number) : [];
  if (!Number.isFinite(confidence) || box.length !== 4 || box.some((value) => !Number.isFinite(value))) {
    return null;
  }
  return {
    className: raw.class_name,
    label: CLASS_CN[raw.class_name] ?? raw.class_name,
    confidence,
    box: [box[0], box[1], box[2], box[3]],
  };
}

export async function detectDefects(imagePath: string, conf = confidenceThreshold()): Promise<VisionDetection> {
  if (!visionConfigured()) {
    throw new Error('YOLO 模型或检测脚本未就绪，请确认 results/week2_3/yolo/train/weights/best.pt 存在。');
  }

  let stdout = '';
  try {
    const result = await execFileAsync(pythonBin(), [
      yoloScriptPath(),
      '--image', imagePath,
      '--model', visionModelPath(),
      '--conf', String(conf),
    ], {
      cwd: PROJECT_ROOT,
      timeout: 120_000,
      maxBuffer: 64 * 1024 * 1024,
    });
    stdout = result.stdout;
  } catch (error) {
    const failed = error as { stdout?: string; killed?: boolean; signal?: string };
    const parsed = parseOutput(failed.stdout ?? '');
    if (parsed?.error) throw new Error(parsed.error);
    if (failed.killed || failed.signal) throw new Error('YOLO 检测超时，请换一张更小的图片重试。');
    throw error;
  }

  const data = parseOutput(stdout);
  if (!data) throw new Error('YOLO 检测脚本没有返回有效结果。');
  if (data.ok === false) throw new Error(data.error || 'YOLO 检测失败。');
  if (typeof data.annotated !== 'string' || !data.annotated) {
    throw new Error('YOLO 检测脚本没有返回标注图。');
  }

  const detections = Array.isArray(data.detections)
    ? data.detections.map(normalizeDetection).filter((item): item is Detection => item !== null)
    : [];

  return {
    detections,
    width: Number(data.width) || 0,
    height: Number(data.height) || 0,
    elapsedMs: Number(data.elapsed_ms) || 0,
    annotatedImage: data.annotated,
    model: path.basename(visionModelPath()),
    conf,
  };
}

export function summarizeDetections(
  detections: Detection[],
  conf: number,
): { counts: DetectionCount[]; summary: string } {
  const map = new Map<string, DetectionCount>();
  for (const detection of detections) {
    const item = map.get(detection.className)
      ?? { className: detection.className, label: detection.label, count: 0, maxConfidence: 0 };
    item.count += 1;
    item.maxConfidence = Math.max(item.maxConfidence, detection.confidence);
    map.set(detection.className, item);
  }
  const counts = [...map.values()].sort(
    (a, b) => b.count - a.count || b.maxConfidence - a.maxConfidence,
  );
  const summary = detections.length > 0
    ? `检出 ${detections.length} 个缺陷目标：${counts.map((item) => `${item.label} ×${item.count}`).join('、')}（置信度阈值 ${conf}）`
    : `在置信度阈值 ${conf} 下未检出缺陷目标`;
  return { counts, summary };
}

export function templateAnswer(detections: Detection[], counts: DetectionCount[], conf: number): string {
  if (detections.length === 0) {
    return `YOLO 在图片中未检出缺陷目标（置信度阈值 ${conf}）。`
      + '这可能表示表面状态良好，也可能是缺陷过小或图像质量不足；建议降低阈值复检或人工确认。';
  }
  const lines = counts.map(
    (item) => `· ${item.label}：${item.count} 处，最高置信度 ${Math.round(item.maxConfidence * 100)}%`,
  );
  return `YOLO 在图片中检出 ${detections.length} 个缺陷目标（置信度阈值 ${conf}）：\n`
    + `${lines.join('\n')}\n`
    + '建议结合缺陷类型、位置和现场工艺进一步人工复检。';
}
