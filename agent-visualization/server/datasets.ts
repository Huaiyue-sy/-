import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { parse } from 'csv-parse/sync';
import type { ColumnKind, ColumnProfile, Dataset, DatasetMeta, Row } from './types';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const APP_DIR = path.resolve(HERE, '..');
export const PROJECT_ROOT = path.resolve(HERE, '../..');
const DATA_DIR = path.join(APP_DIR, 'data');
const MAX_ROWS = 50_000;
const DATE_RE = /^\d{4}[-/]\d{1,2}[-/]\d{1,2}/;

interface BuiltinDef {
  id: string;
  name: string;
  group: string;
  description: string;
  candidates: string[];
}

const BUILTINS: BuiltinDef[] = [
  {
    id: 'bearing_features',
    name: '轴承特征全量',
    group: '轴承数据',
    description:
      'XJTU-SY 全部振动文件的 56 列特征：工况、轴承、健康阶段、相对 RUL、时域/频域/包络谱特征。',
    candidates: ['results/week1/features.csv', 'data/bearing_features_sample.csv'],
  },
  {
    id: 'bearing_models',
    name: '轴承模型对比',
    group: '轴承数据',
    description: '随机森林、SVM、KNN、逻辑回归、CNN、LSTM、RNN 的测试准确率与 macroF1。',
    candidates: ['data/bearing_models.csv'],
  },
  {
    id: 'classify_health',
    name: '四模型 CV / 留出指标',
    group: '轴承数据',
    description: '第 1 周四个传统模型的 5 折 CV 与按轴承留出测试指标。',
    candidates: ['data/classify_health_results.csv'],
  },
  {
    id: 'cnn_lstm_rf',
    name: 'CNN / LSTM / RF',
    group: '轴承数据',
    description: '第 2 周深度模型与随机森林的同数据同划分对比。',
    candidates: ['data/cnn_vs_lstm_vs_rf.csv'],
  },
  {
    id: 'rul_models',
    name: 'RUL 回归对比',
    group: '轴承数据',
    description: 'GradientBoosting、RandomForest、LSTM 的 MAE / RMSE / R²。',
    candidates: ['data/rul_results.csv'],
  },
  {
    id: 'feature_ranking',
    name: '特征重要性排序',
    group: '轴承数据',
    description: '互信息与 ANOVA 排序后的特征清单。',
    candidates: ['data/feature_ranking.csv'],
  },
  {
    id: 'neu_models',
    name: 'ResNet / EfficientNet',
    group: '表面瑕疵',
    description: 'NEU-DET 六分类：ResNet18 与 EfficientNet-B0 的预训练/从零对照。',
    candidates: ['data/neu_models.csv'],
  },
  {
    id: 'yolo_per_class',
    name: 'YOLO 分类别指标',
    group: '表面瑕疵',
    description: 'YOLOv8n 在六类缺陷上的 P / R / mAP50 / mAP50-95。',
    candidates: ['data/yolo_per_class.csv'],
  },
  {
    id: 'optimization',
    name: '学习率 × 损失消融',
    group: '表面瑕疵',
    description: 'ResNet18 在 3 个学习率 × 3 种损失下的测试准确率。',
    candidates: ['data/optimization_results.csv'],
  },
  {
    id: 'yolo_overall',
    name: 'YOLO 总体指标',
    group: '表面瑕疵',
    description: 'YOLOv8n 30 epochs 的总体 mAP50、mAP50-95、mP、mR。',
    candidates: ['data/yolo_results.csv'],
  },
];

const cache = new Map<string, Dataset>();

function firstExisting(paths: string[]): string | null {
  for (const p of paths) {
    if (path.isAbsolute(p)) {
      if (fs.existsSync(p)) return p;
      continue;
    }
    const fromApp = path.join(APP_DIR, p);
    if (fs.existsSync(fromApp)) return fromApp;
    const fromRoot = path.join(PROJECT_ROOT, p);
    if (fs.existsSync(fromRoot)) return fromRoot;
  }
  return null;
}

function normalizeCell(value: unknown): string {
  if (value === null || value === undefined) return '';
  return String(value).replace(/^\uFEFF/, '').trim();
}

function inferColumn(name: string, rawValues: string[]): ColumnProfile {
  const nonEmpty = rawValues.filter((v) => v !== '');
  const missing = rawValues.length - nonEmpty.length;
  const uniqueSet = new Set(nonEmpty);
  const numeric = nonEmpty.map(Number).filter((n) => Number.isFinite(n));
  const numericRatio = nonEmpty.length ? numeric.length / nonEmpty.length : 0;
  const allInt = numeric.every((n) => Number.isInteger(n));
  const looksDate = nonEmpty.length > 0 && nonEmpty.slice(0, 50).every((v) => DATE_RE.test(v));
  const nameHint = /stage|class|类别|阶段|等级|状态/i.test(name);

  let kind: ColumnKind = 'text';
  if (numericRatio >= 0.85 && numeric.length > 0) {
    if (allInt && uniqueSet.size <= 12 && nameHint) kind = 'category';
    else kind = 'number';
  } else if (looksDate) {
    kind = 'date';
  } else if (uniqueSet.size <= Math.max(24, Math.round(rawValues.length * 0.1))) {
    kind = 'category';
  }

  const profile: ColumnProfile = {
    name,
    kind,
    missing,
    unique: uniqueSet.size,
    samples: [...uniqueSet].slice(0, 6).map((v) => v.slice(0, 48)),
  };
  if (kind === 'number' && numeric.length) {
    profile.min = Math.min(...numeric);
    profile.max = Math.max(...numeric);
    profile.mean = numeric.reduce((a, b) => a + b, 0) / numeric.length;
  }
  return profile;
}

function recordsToDataset(
  id: string,
  name: string,
  group: string,
  description: string,
  source: 'builtin' | 'upload',
  records: Record<string, unknown>[],
  file?: string,
): Dataset {
  const limited = records.slice(0, MAX_ROWS);
  const headers = Object.keys(limited[0] ?? {});
  const columns = headers.map((header) => {
    const values = limited.map((r) => normalizeCell(r[header]));
    return inferColumn(header, values);
  });
  const columnMap = new Map(columns.map((c) => [c.name, c]));
  const rows: Row[] = limited.map((record) => {
    const row: Row = {};
    for (const header of headers) {
      const raw = normalizeCell(record[header]);
      const kind = columnMap.get(header)?.kind;
      if (raw === '') row[header] = null;
      else if (kind === 'number') {
        const n = Number(raw);
        row[header] = Number.isFinite(n) ? n : null;
      } else row[header] = raw;
    }
    return row;
  });
  return {
    id,
    name,
    group,
    description,
    source,
    rows,
    columns,
    file,
    truncated: records.length > MAX_ROWS,
  };
}

function parseCsv(text: string): Record<string, unknown>[] {
  return parse(text, {
    columns: true,
    skip_empty_lines: true,
    relax_column_count: true,
    bom: true,
    trim: true,
  }) as Record<string, unknown>[];
}

function loadBuiltin(def: BuiltinDef): Dataset {
  const cached = cache.get(def.id);
  if (cached) return cached;
  const file = firstExisting(def.candidates);
  if (!file) throw new Error(`内置数据集文件不存在：${def.name}`);
  const text = fs.readFileSync(file, 'utf8');
  const records = parseCsv(text);
  const isSample = file.endsWith('bearing_features_sample.csv');
  const name = isSample ? `${def.name}（抽样）` : def.name;
  const description = isSample
    ? `${def.description} 当前使用 600 行分层抽样快照；本机存在完整 features.csv 时会自动使用全量数据。`
    : def.description;
  const dataset = recordsToDataset(def.id, name, def.group, description, 'builtin', records, file);
  cache.set(def.id, dataset);
  return dataset;
}

export function getDataset(id: string): Dataset | undefined {
  if (cache.has(id)) return cache.get(id);
  const def = BUILTINS.find((d) => d.id === id);
  if (!def) return undefined;
  try {
    return loadBuiltin(def);
  } catch {
    return undefined;
  }
}

function suggestionsFor(id: string): string[] {
  switch (id) {
    case 'bearing_features':
      return [
        '画出 Bearing1_1 的 RMS 随文件序号变化',
        '按工况对比 RMS 分布',
        '各健康阶段样本占比',
      ];
    case 'bearing_models':
      return ['对比各模型的测试准确率和 macroF1', '按模型画测试准确率柱状图'];
    case 'classify_health':
      return ['对比四个模型的测试准确率和 macroF1', '各模型的 CV 准确率'];
    case 'cnn_lstm_rf':
      return ['对比 CNN、LSTM、RF 的测试准确率', 'CNN 与随机森林的 macroF1'];
    case 'rul_models':
      return ['对比 RUL 模型的测试 MAE 和 RMSE', '各模型的测试 R²'];
    case 'feature_ranking':
      return ['画互信息 Top 15 特征条形图', '哪些特征排在前面？'];
    case 'neu_models':
      return ['对比 ResNet18 与 EfficientNet-B0 的测试准确率', '预训练与从零训练的准确率对比'];
    case 'yolo_per_class':
      return ['画出六类缺陷的 mAP50 柱状图', '哪类缺陷的召回率最低？'];
    case 'optimization':
      return ['画学习率×损失函数的测试准确率热力图', '对比不同学习率的测试准确率'];
    case 'yolo_overall':
      return ['YOLO 的 mAP50 和 mAP50-95', 'YOLO 的精确率和召回率'];
    default:
      return ['按第一个类别列统计数量', '画第一个数值列的分布', '选择两个数值列做散点图'];
  }
}

export function toMeta(dataset: Dataset): DatasetMeta {
  const isSample = Boolean(dataset.file?.endsWith('bearing_features_sample.csv'));
  return {
    id: dataset.id,
    name: dataset.name,
    group: dataset.group,
    description: dataset.description,
    source: dataset.source,
    rowCount: dataset.rows.length,
    columns: dataset.columns.map((c) => ({
      name: c.name,
      kind: c.kind,
      unique: c.unique,
      min: c.min,
      max: c.max,
    })),
    suggestedQuestions: suggestionsFor(dataset.id),
    sample: isSample,
  };
}

export function listDatasets(): DatasetMeta[] {
  const metas: DatasetMeta[] = [];
  for (const def of BUILTINS) {
    try {
      metas.push(toMeta(loadBuiltin(def)));
    } catch {
      // 文件缺失时跳过，前端仍可上传 CSV
    }
  }
  return metas;
}

export function addUpload(buffer: Buffer, filename: string): Dataset {
  const text = buffer.toString('utf8');
  const records = parseCsv(text);
  if (!records.length) throw new Error('CSV 没有数据行');
  const id = `upload-${randomUUID().slice(0, 8)}`;
  const name = filename.replace(/\.csv$/i, '') || '上传数据';
  const dataset = recordsToDataset(
    id,
    name,
    '上传数据',
    `上传文件：${filename}（${records.length} 行）`,
    'upload',
    records,
  );
  cache.set(id, dataset);
  return dataset;
}
