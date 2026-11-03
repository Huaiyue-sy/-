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
}
