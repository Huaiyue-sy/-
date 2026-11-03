export type ColumnKind = 'number' | 'category' | 'date' | 'text';

export interface ColumnMeta {
  name: string;
  kind: ColumnKind;
  unique: number;
  min?: number;
  max?: number;
}

export interface DatasetMeta {
  id: string;
  name: string;
  group: string;
  description: string;
  source: 'builtin' | 'upload';
  rowCount: number;
  columns: ColumnMeta[];
  suggestedQuestions: string[];
  sample: boolean;
}

export interface ChartSpec {
  type: string;
  title: string;
  x?: string;
  y?: string | string[];
  series?: string;
  agg?: string;
  filter?: { column: string; value: string };
  sort?: string;
  limit?: number;
  explanation?: string;
}

export interface ChatResponse {
  kind: 'chart' | 'chat';
  answer: string;
  spec?: ChartSpec;
  option?: Record<string, unknown>;
  planner?: 'llm';
  warning?: string;
  summary?: string;
  usedRows?: number;
  preview?: { columns: string[]; rows: Record<string, string | number | null>[] };
  dataset?: { id: string; name: string; rowCount: number };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  kind?: 'chart' | 'chat' | 'vision';
  planner?: 'llm';
  warning?: string;
  summary?: string;
  pending?: boolean;
}

export interface SessionRecord {
  id: string;
  title: string;
  datasetId: string;
  datasetName: string;
  messages: ChatMessage[];
  chart: ChatResponse | null;
  vision: VisionResponse | null;
  createdAt: number;
  updatedAt: number;
}

export interface Detection {
  className: string;
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface DetectionCount {
  className: string;
  label: string;
  count: number;
  maxConfidence: number;
}

export interface VisionResponse {
  kind: 'vision';
  answer: string;
  image: string;
  detections: Detection[];
  counts: DetectionCount[];
  summary: string;
  model: string;
  conf: number;
  width: number;
  height: number;
  elapsedMs: number;
  planner?: 'llm';
  warning?: string;
}

export interface HealthInfo {
  ok: boolean;
  llm: {
    configured: boolean;
    mode: string;
    model: string;
    baseUrl: string;
  };
  datasets: number;
  vision?: {
    configured: boolean;
    model: string;
    python: string;
    confidence: number;
  };
  dashboard?: {
    configured: boolean;
    devices: number;
    webhook: boolean;
    generatedAt: string | null;
  };
}

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
  ok: boolean;
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

export interface DashboardDeviceDetail {
  ok: boolean;
  device: DashboardDeviceSummary;
  history: DashboardPoint[];
  alerts: DashboardAlert[];
}
