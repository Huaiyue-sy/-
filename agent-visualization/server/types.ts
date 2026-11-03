export type ColumnKind = 'number' | 'category' | 'date' | 'text';

export type Row = Record<string, string | number | null>;

export interface ColumnProfile {
  name: string;
  kind: ColumnKind;
  missing: number;
  unique: number;
  min?: number;
  max?: number;
  mean?: number;
  samples: string[];
}

export interface Dataset {
  id: string;
  name: string;
  group: string;
  description: string;
  source: 'builtin' | 'upload';
  rows: Row[];
  columns: ColumnProfile[];
  file?: string;
  truncated?: boolean;
}

export interface DatasetMeta {
  id: string;
  name: string;
  group: string;
  description: string;
  source: 'builtin' | 'upload';
  rowCount: number;
  columns: Array<Pick<ColumnProfile, 'name' | 'kind' | 'unique' | 'min' | 'max'>>;
  suggestedQuestions: string[];
  sample: boolean;
}

export type ChartType = 'line' | 'bar' | 'scatter' | 'boxplot' | 'heatmap' | 'pie' | 'histogram';

export interface ChartSpec {
  type: ChartType;
  title: string;
  x?: string;
  y?: string | string[];
  series?: string;
  agg?: 'mean' | 'sum' | 'count' | 'none';
  filter?: { column: string; value: string };
  sort?: 'asc' | 'desc' | 'none';
  limit?: number;
  explanation?: string;
}

export interface ChartBuildResult {
  option: Record<string, unknown>;
  summary: string;
  usedRows: number;
  warning?: string;
}

export interface ChatResult {
  answer: string;
  spec: ChartSpec;
  option: Record<string, unknown>;
  planner: 'llm' | 'rules';
  warning?: string;
  summary: string;
  usedRows: number;
  preview: { columns: string[]; rows: Row[] };
  dataset: { id: string; name: string; rowCount: number };
}
