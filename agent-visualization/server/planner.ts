import type { ChartSpec, ChartType, ColumnProfile, Dataset } from './types';
import { pretty } from './labels';

interface AliasGroup {
  keys: string[];
  test: (name: string) => boolean;
  score?: (name: string) => number;
}

const ALIASES: AliasGroup[] = [
  { keys: ['mAP50-95', 'mAP50_95', 'map50-95', 'map50_95'], test: (n) => /map50[_-]?95/i.test(n) },
  { keys: ['mAP50', 'map50'], test: (n) => /map50(?!.*95)/i.test(n) },
  { keys: ['准确率', 'accuracy', 'acc'], test: (n) => /准确率|acc/i.test(n) && !/macro/i.test(n), score: (n) => (/测试|test/i.test(n) ? 4 : 0) },
  { keys: ['macroF1', 'macro f1', 'macro_f1', 'f1'], test: (n) => /macro.?f1|f1/i.test(n), score: (n) => (/测试|test/i.test(n) ? 4 : 0) },
  { keys: ['精确率', 'precision'], test: (n) => /precision|精确率/i.test(n) },
  { keys: ['召回率', 'recall'], test: (n) => /recall|召回率/i.test(n) },
  { keys: ['rms', '均方根'], test: (n) => /^rms|rms_/i.test(n), score: (n) => (n === 'rms_mean' ? 4 : n === 'rms_h' ? 3 : n === 'rms_v' ? 2 : 0) },
  { keys: ['健康阶段', '阶段', 'stage'], test: (n) => /stage|阶段/i.test(n) },
  { keys: ['轴承', 'bearing'], test: (n) => /bearing|轴承/i.test(n) },
  { keys: ['工况', 'condition'], test: (n) => /condition|工况/i.test(n) },
  { keys: ['文件序号', '序号', 'file_idx', 'index'], test: (n) => /file_idx|文件序号|序号/i.test(n) },
  { keys: ['剩余寿命', '寿命', 'rul'], test: (n) => /rel_rul|rul|剩余寿命|寿命/i.test(n) },
  { keys: ['模型', 'model'], test: (n) => /model|模型/i.test(n) },
  { keys: ['学习率', 'lr'], test: (n) => /^lr$|学习率/i.test(n) },
  { keys: ['损失', 'loss'], test: (n) => /loss|损失/i.test(n) },
  { keys: ['类别', '缺陷', 'class'], test: (n) => /class|类别|缺陷/i.test(n) },
  { keys: ['mae'], test: (n) => /mae/i.test(n), score: (n) => (/测试|test/i.test(n) ? 6 : 0) },
  { keys: ['rmse'], test: (n) => /rmse/i.test(n), score: (n) => (/测试|test/i.test(n) ? 6 : 0) },
  { keys: ['r2', 'r²'], test: (n) => /r2|r²/i.test(n), score: (n) => (/测试|test/i.test(n) ? 6 : 0) },
  { keys: ['互信息', 'mutual'], test: (n) => /mutual|互信息/i.test(n) },
  { keys: ['anova', 'f值'], test: (n) => /anova|f值/i.test(n) },
  { keys: ['特征', 'feature'], test: (n) => /^feature$|特征/i.test(n) },
  { keys: ['预训练', 'pretrained'], test: (n) => /pretrained|预训练/i.test(n) },
  { keys: ['参数量', 'params'], test: (n) => /params|参数量/i.test(n) },
];

const TYPE_LABELS: Record<ChartType, string> = {
  line: '折线图',
  bar: '柱状图',
  scatter: '散点图',
  boxplot: '箱线图',
  heatmap: '热力图',
  pie: '饼图',
  histogram: '直方图',
};

function isNumeric(col: ColumnProfile): boolean {
  return col.kind === 'number';
}

const METRIC_RE = /acc|f1|precision|recall|mae|rmse|r2|map50|mutual|anova|instances|params|rms|准确率|召回|精确|互信息|mP|mR/i;

function isMetricColumn(name: string): boolean {
  return METRIC_RE.test(name);
}

function isTrueCategorical(col: ColumnProfile, dataset: Dataset): boolean {
  if (col.kind === 'category' || col.kind === 'date') return true;
  return col.kind === 'text' && col.unique <= 80 && dataset.rows.length > 0;
}

function isAxisCategorical(col: ColumnProfile, dataset: Dataset): boolean {
  if (isTrueCategorical(col, dataset)) return true;
  return col.kind === 'number' && col.unique <= 12 && !isMetricColumn(col.name);
}

function metricKey(name: string): string {
  if (/map50[_-]?95/i.test(name)) return 'map50_95';
  if (/map50/i.test(name)) return 'map50';
  if (/macro.?f1|f1/i.test(name)) return 'f1';
  if (/准确率|acc/i.test(name)) return 'acc';
  if (/precision|精确率/i.test(name)) return 'precision';
  if (/recall|召回率/i.test(name)) return 'recall';
  if (/mae/i.test(name)) return 'mae';
  if (/rmse/i.test(name)) return 'rmse';
  if (/r2|r²/i.test(name)) return 'r2';
  if (/rms/i.test(name)) return 'rms';
  if (/mutual/i.test(name)) return 'mutual';
  if (/anova/i.test(name)) return 'anova';
  if (/instances/i.test(name)) return 'instances';
  return name;
}

function dedupeMetrics(columns: ColumnProfile[]): ColumnProfile[] {
  const seen = new Set<string>();
  const result: ColumnProfile[] = [];
  for (const col of columns) {
    const key = metricKey(col.name);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(col);
  }
  return result;
}

function mentionedColumns(question: string, dataset: Dataset): string[] {
  const q = question.toLowerCase();
  const scores = new Map<string, number>();
  for (const col of dataset.columns) {
    const name = col.name.toLowerCase();
    const label = pretty(col.name).toLowerCase();
    let score = 0;
    if (name.length >= 2 && q.includes(name)) score = Math.max(score, 60 + name.length);
    if (label.length >= 2 && q.includes(label)) score = Math.max(score, 50 + label.length);
    for (const alias of ALIASES) {
      if (alias.keys.some((key) => q.includes(key.toLowerCase())) && alias.test(col.name)) {
        score = Math.max(score, 30 + (alias.score?.(col.name) ?? 0));
      }
    }
    if (score > 0) scores.set(col.name, score);
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1]).map(([name]) => name);
}

function detectType(question: string, dataset: Dataset): ChartType {
  const q = question.toLowerCase();
  if (/热力|矩阵|heatmap/.test(q)) return 'heatmap';
  if (/箱线|箱型|boxplot|box plot/.test(q)) return 'boxplot';
  if (/饼图|占比|比例|pie/.test(q)) return 'pie';
  if (/直方|频数|histogram|hist/.test(q)) return 'histogram';
  if (/散点|相关|关系|scatter/.test(q)) return 'scatter';
  if (/折线|趋势|变化|随|曲线|line/.test(q)) return 'line';
  if (/分布/.test(q)) return 'boxplot';
  if (/柱状|条形|对比|比较|排行|排名|bar/.test(q)) return 'bar';
  if (dataset.columns.some((c) => c.name === 'model')) return 'bar';
  if (dataset.columns.some((c) => c.name === 'file_idx')) return 'line';
  return 'bar';
}

function preferredCategory(columns: ColumnProfile[]): ColumnProfile | undefined {
  const order = ['model', '模型', 'condition', '工况', 'class', '类别', '缺陷', 'stage', '健康阶段', 'bearing', '轴承'];
  for (const key of order) {
    const hit = columns.find((c) => c.name === key);
    if (hit) return hit;
  }
  const nonDate = columns.filter((c) => c.kind !== 'date');
  return nonDate[0] ?? columns[0];
}

function findColumn(dataset: Dataset, name: string | undefined): ColumnProfile | undefined {
  if (!name) return undefined;
  return dataset.columns.find((c) => c.name === name);
}

function chooseFilter(question: string, dataset: Dataset): ChartSpec['filter'] | undefined {
  const bearing = question.match(/Bearing\d+_\d+/i)?.[0];
  if (bearing) {
    const col = dataset.columns.find((c) => c.name === 'bearing')
      ?? dataset.columns.find((c) => c.name === 'bearing_key');
    if (col) return { column: col.name, value: bearing };
  }
  const condition = question.match(/\d+(?:\.\d+)?Hz\d+kN/i)?.[0];
  if (condition) {
    const col = dataset.columns.find((c) => c.name === 'condition');
    if (col) return { column: col.name, value: condition };
  }
  return undefined;
}

function chooseSeries(question: string, dataset: Dataset, x: string | undefined,
  y: string | string[] | undefined, type: ChartType, filter: ChartSpec['filter'] | undefined): string | undefined {
  const q = question.toLowerCase();
  const axisCats = dataset.columns.filter((c) => isAxisCategorical(c, dataset));
  const explicit = q.match(/按\s*([^\s，,。]+?)\s*(?:分组|着色|分色|区分)/)?.[1];
  if (explicit) {
    const hit = dataset.columns.find((c) =>
      c.name.toLowerCase() === explicit.toLowerCase() || pretty(c.name).toLowerCase() === explicit.toLowerCase());
    if (hit && hit.name !== x) return hit.name;
  }
  if (type === 'heatmap') {
    const mentioned = mentionedColumns(question, dataset)
      .map((name) => findColumn(dataset, name))
      .filter((c): c is ColumnProfile => Boolean(c && isAxisCategorical(c, dataset) && c.name !== x));
    return mentioned[0]?.name ?? axisCats.find((c) => c.name !== x)?.name;
  }
  if (Array.isArray(y)) return undefined;
  if (type === 'line' && !filter) {
    if (/各轴承|不同轴承|每个轴承/.test(q)) {
      const col = dataset.columns.find((c) => c.name === 'bearing');
      if (col && col.name !== x) return col.name;
    }
    const col = dataset.columns.find((c) => c.name === 'condition');
    if (col && col.name !== x && dataset.columns.some((c) => c.name === 'file_idx')) return col.name;
  }
  return undefined;
}

function chooseColumns(question: string, dataset: Dataset, type: ChartType) {
  const q = question.toLowerCase();
  const mentioned = mentionedColumns(question, dataset);
  const numeric = dataset.columns.filter(isNumeric);
  const trueCats = dataset.columns.filter((c) => isTrueCategorical(c, dataset));
  const axisCats = dataset.columns.filter((c) => isAxisCategorical(c, dataset));
  const mentionedTrueCats = mentioned.map((n) => findColumn(dataset, n))
    .filter((c): c is ColumnProfile => Boolean(c && isTrueCategorical(c, dataset)));
  const mentionedAxisCats = mentioned.map((n) => findColumn(dataset, n))
    .filter((c): c is ColumnProfile => Boolean(c && isAxisCategorical(c, dataset)));
  const mentionedNums = mentioned.map((n) => findColumn(dataset, n))
    .filter((c): c is ColumnProfile => Boolean(c && isNumeric(c)));

  let x: string | undefined;
  if (type === 'histogram') {
    x = undefined;
  } else if (type === 'heatmap') {
    const priority = ['lr', 'condition', 'stage', 'health_stage', 'pretrained', 'model', 'class'];
    const preferredAxis = priority
      .map((name) => mentionedAxisCats.find((c) => c.name === name))
      .find((c): c is ColumnProfile => Boolean(c));
    x = preferredAxis?.name ?? mentionedAxisCats[0]?.name
      ?? preferredCategory(axisCats)?.name ?? axisCats[0]?.name;
  } else if (type === 'pie' || type === 'bar' || type === 'boxplot') {
    x = mentionedTrueCats[0]?.name ?? mentionedAxisCats[0]?.name
      ?? preferredCategory(trueCats)?.name ?? trueCats[0]?.name;
  } else if (type === 'scatter') {
    if (dedupeMetrics(mentionedNums).length < 2 && mentionedTrueCats.length >= 1) {
      return chooseColumns(question, dataset, 'boxplot');
    }
    x = mentionedNums[0]?.name ?? numeric[0]?.name;
  } else {
    const fileIdx = dataset.columns.find((c) => c.name === 'file_idx');
    x = (q.includes('文件序号') && fileIdx) ? fileIdx.name : (mentionedNums[0]?.name ?? fileIdx?.name ?? numeric[0]?.name);
  }

  const yCandidates = dedupeMetrics(mentionedNums.filter((c) => c.name !== x));
  const wantsCount = /统计数量|数量|次数|计数|多少个|count/i.test(question)
    && !/平均|均值|总和|sum/i.test(question);
  const hasConjunction = /和|与|及|、|以及/.test(question);
  const effectiveYCandidates = hasConjunction ? yCandidates : yCandidates.slice(0, 1);
  let y: string | string[] | undefined;
  if (wantsCount && !yCandidates.length) {
    y = undefined;
  } else if (type === 'pie' && !effectiveYCandidates.length) {
    y = undefined;
  } else if (effectiveYCandidates.length) {
    const values = effectiveYCandidates.slice(0, 3).map((c) => c.name);
    y = values.length === 1 ? values[0] : values;
  } else if (type !== 'pie' && type !== 'bar') {
    const fallback = numeric.find((c) => c.name !== x);
    y = fallback?.name;
  } else {
    y = numeric.find((c) => c.name !== x)?.name;
  }

  if (x && y) {
    const yArr = Array.isArray(y) ? y : [y];
    if (yArr.includes(x)) {
      const alternative = trueCats.find((c) => c.name !== x)?.name ?? axisCats.find((c) => c.name !== x)?.name;
      if (alternative) x = alternative;
    }
  }
  if (type === 'scatter' && x && y) {
    const yFirst = Array.isArray(y) ? y[0] : y;
    if (!isNumeric(findColumn(dataset, x) ?? { kind: 'text' } as ColumnProfile) ||
        !isNumeric(findColumn(dataset, yFirst) ?? { kind: 'text' } as ColumnProfile)) {
      // 散点图需要两个数值列，无法满足时退回箱线图
      return chooseColumns(question, dataset, 'boxplot');
    }
  }
  if (type === 'heatmap' && (!x || !y)) {
    return chooseColumns(question, dataset, 'bar');
  }
  return { x, y, mentionedTrueCats, mentionedAxisCats, mentionedNums };
}

function buildTitle(type: ChartType, x: string | undefined, y: string | string[] | undefined, series: string | undefined): string {
  const yLabel = Array.isArray(y) ? y.map(pretty).join('、') : (y ? pretty(y) : '样本数');
  const xLabel = pretty(x);
  switch (type) {
    case 'line': return `${yLabel}${xLabel ? ` 随 ${xLabel} 变化` : ''}`;
    case 'bar': return `${xLabel ? `${xLabel} 的 ` : ''}${yLabel}`;
    case 'boxplot': return `${xLabel ? `${xLabel} 分组的 ` : ''}${yLabel} 分布`;
    case 'scatter': return `${xLabel} 与 ${yLabel} 的关系`;
    case 'heatmap': return `${xLabel}${series ? ` × ${pretty(series)}` : ''} 的 ${yLabel}`;
    case 'pie': return `${xLabel || '类别'}占比`;
    case 'histogram': return `${yLabel} 分布直方图`;
    default: return yLabel;
  }
}

export function rulePlan(dataset: Dataset, question: string): ChartSpec {
  let type = detectType(question, dataset);
  let { x, y, mentionedTrueCats, mentionedAxisCats, mentionedNums } = chooseColumns(question, dataset, type);
  const filter = chooseFilter(question, dataset);
  let series = chooseSeries(question, dataset, x, y, type, filter);

  const xProfile = findColumn(dataset, x);
  const yFirst = Array.isArray(y) ? y[0] : y;
  const yProfile = findColumn(dataset, yFirst);
  if (type === 'scatter' && (!xProfile || xProfile.kind !== 'number' || !yProfile || yProfile.kind !== 'number')) {
    type = 'boxplot';
  }
  if (type === 'boxplot' && (!xProfile || !isAxisCategorical(xProfile, dataset) || !yProfile)) {
    type = 'bar';
  }

  if (type === 'boxplot' && !y) {
    type = 'bar';
    y = numericFallback(dataset, x);
  }
  if (type === 'heatmap' && (!x || !y || !series)) {
    type = 'bar';
  }
  if (type === 'scatter' && !x) {
    type = 'bar';
  }
  if (!y && !['pie', 'bar'].includes(type)) {
    y = numericFallback(dataset, x);
  }
  if (type === 'pie' && !x) {
    x = mentionedTrueCats[0]?.name ?? preferredCategory(dataset.columns.filter((c) => isTrueCategorical(c, dataset)))?.name
      ?? mentionedAxisCats[0]?.name;
  }
  if (!x && type !== 'histogram') {
    x = mentionedTrueCats[0]?.name ?? preferredCategory(dataset.columns.filter((c) => isTrueCategorical(c, dataset)))?.name
      ?? mentionedAxisCats[0]?.name ?? mentionedNums[0]?.name;
  }
  if (series === x || (Array.isArray(y) ? y.includes(series ?? '') : series === y)) series = undefined;

  const agg: ChartSpec['agg'] = type === 'boxplot' || type === 'scatter' || type === 'histogram'
    ? 'none'
    : (y ? (type === 'pie' ? 'sum' : 'mean') : 'count');
  const sort: ChartSpec['sort'] = type === 'line'
    ? 'asc'
    : (/最低|最少|从低到高|升序/.test(question) ? 'asc'
      : (type === 'pie' || /从高到低|降序|排名|排行|最高|top/i.test(question) ? 'desc' : 'none'));
  const baseLimit = type === 'line' ? 300
    : type === 'scatter' ? 800
      : type === 'heatmap' ? 60
        : type === 'pie' ? 20
          : 30;
  const topMatch = question.match(/(?:top|前)\s*(\d+)/i);
  const limit = topMatch ? Math.min(baseLimit, Math.max(1, Number(topMatch[1]))) : baseLimit;

  return {
    type,
    title: buildTitle(type, x, y, series),
    x,
    y,
    series,
    agg,
    filter,
    sort,
    limit,
    explanation: `规则引擎：${TYPE_LABELS[type]}，x=${x ?? '无'}，y=${Array.isArray(y) ? y.join(' / ') : (y ?? '计数')}${series ? `，分组=${series}` : ''}${filter ? `，筛选 ${pretty(filter.column)}=${filter.value}` : ''}。`,
  };
}

function numericFallback(dataset: Dataset, exclude?: string): string | undefined {
  return dataset.columns.find((c) => isNumeric(c) && c.name !== exclude && !/file_idx/i.test(c.name))?.name
    ?? dataset.columns.find((c) => isNumeric(c) && c.name !== exclude)?.name;
}

export { TYPE_LABELS };
