import type { ChartBuildResult, ChartSpec, ColumnProfile, Dataset, Row } from './types';
import { pretty } from './labels';

const COLORS = ['#46c8ff', '#ff7a90', '#ffd166', '#39d98a', '#a78bfa', '#ff9f5a', '#5eead4', '#f472b6'];
const AXIS_LABEL = '#93aec7';
const AXIS_LINE = '#294862';
const SPLIT_LINE = 'rgba(41,72,98,.55)';
const TEXT = '#e6f1fb';

type Value = string | number | null;

function asNumber(value: Value): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' && value.trim() !== '') {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function asString(value: Value): string {
  if (value === null || value === undefined || value === '') return '空值';
  return String(value);
}

function columnOf(dataset: Dataset, name: string | undefined): ColumnProfile | undefined {
  if (!name) return undefined;
  return dataset.columns.find((c) => c.name === name);
}

function formatCategory(column: string | undefined, value: Value): string {
  const raw = asString(value);
  if (column === 'stage' || column === 'health_stage') {
    return raw === '0' ? '健康' : raw === '1' ? '退化' : raw === '2' ? '故障' : raw;
  }
  if (column === 'pretrained') {
    return raw === '1' ? '预训练' : raw === '0' ? '从零训练' : raw;
  }
  if (column === 'loss') {
    if (raw === 'CE') return 'CE';
    if (/^CE_LS/i.test(raw)) return `CE+LS ${raw.split('_LS')[1] ?? ''}`.trim();
    if (/focal/i.test(raw)) return raw.replace('Focal_g', 'Focal γ=');
  }
  return raw;
}

function filterRows(dataset: Dataset, filter: ChartSpec['filter']): { rows: Row[]; warning?: string } {
  if (!filter) return { rows: dataset.rows };
  const col = columnOf(dataset, filter.column);
  if (!col) return { rows: dataset.rows, warning: `筛选字段 ${filter.column} 不存在，已忽略筛选。` };
  const needle = filter.value.toLowerCase();
  const matched = dataset.rows.filter((row) => {
    const raw = row[filter.column];
    if (raw === null || raw === undefined) return false;
    return String(raw).toLowerCase().includes(needle);
  });
  if (!matched.length) {
    return { rows: dataset.rows, warning: `筛选 ${pretty(filter.column)}=${filter.value} 没有匹配行，已展示全部数据。` };
  }
  return { rows: matched };
}

function aggregate(values: number[], agg: ChartSpec['agg']): number | null {
  if (!values.length) return null;
  if (agg === 'count') return values.length;
  const sum = values.reduce((a, b) => a + b, 0);
  if (agg === 'sum') return sum;
  return sum / values.length;
}

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function sortEntries(entries: [string, number | null][], sort: ChartSpec['sort']): [string, number | null][] {
  if (sort === 'none') return entries;
  const factor = sort === 'asc' ? 1 : -1;
  return [...entries].sort((a, b) => {
    const av = a[1] ?? Number.NEGATIVE_INFINITY;
    const bv = b[1] ?? Number.NEGATIVE_INFINITY;
    return (av - bv) * factor;
  });
}

function baseOption(spec: ChartSpec): Record<string, unknown> {
  return {
    backgroundColor: 'transparent',
    animation: false,
    color: COLORS,
    textStyle: {
      color: TEXT,
      fontFamily: '-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif',
    },
    tooltip: {
      trigger: spec.type === 'pie' || spec.type === 'scatter' ? 'item' : 'axis',
      backgroundColor: 'rgba(8, 20, 34, .94)',
      borderColor: AXIS_LINE,
      borderWidth: 1,
      textStyle: { color: TEXT, fontSize: 12 },
      axisPointer: { type: spec.type === 'bar' ? 'shadow' : 'line', lineStyle: { color: '#46c8ff66' } },
    },
    legend: {
      top: 12,
      type: 'scroll',
      textStyle: { color: AXIS_LABEL, fontSize: 11 },
      pageTextStyle: { color: AXIS_LABEL },
      icon: 'roundRect',
      itemWidth: 10,
      itemHeight: 8,
    },
    grid: { left: 56, right: 28, top: 58, bottom: 64, containLabel: true },
    xAxis: {
      type: 'category',
      axisLine: { lineStyle: { color: AXIS_LINE } },
      axisTick: { show: false },
      axisLabel: { color: AXIS_LABEL, fontSize: 11, hideOverlap: true },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisLabel: { color: AXIS_LABEL, fontSize: 11 },
      splitLine: { lineStyle: { color: SPLIT_LINE, type: 'dashed' } },
    },
    series: [],
  };
}

function axisName(name: string | undefined): Record<string, unknown> {
  return name
    ? {
      name,
      nameLocation: 'middle',
      nameGap: 34,
      nameTextStyle: { color: AXIS_LABEL, fontSize: 11 },
    }
    : {};
}

function buildLine(spec: ChartSpec, dataset: Dataset, rows: Row[]): ChartBuildResult {
  const xCol = columnOf(dataset, spec.x);
  if (!xCol) throw new Error('折线图缺少 x 字段');
  const yCols = (Array.isArray(spec.y) ? spec.y : spec.y ? [spec.y] : [])
    .map((name) => columnOf(dataset, name))
    .filter((c): c is ColumnProfile => Boolean(c));
  const xNumeric = xCol.kind === 'number' && xCol.unique > 12;

  const categoryMap = new Map<string, number | string>();
  for (const row of rows) {
    const key = asString(row[xCol.name]);
    if (!categoryMap.has(key)) {
      const n = asNumber(row[xCol.name]);
      categoryMap.set(key, n ?? key);
    }
  }
  let categories = [...categoryMap.keys()];
  const numericSort = xNumeric || categories.every((c) => Number.isFinite(Number(c)));
  categories.sort((a, b) => (numericSort ? Number(a) - Number(b) : a.localeCompare(b, 'zh-CN')));
  if (spec.sort === 'desc') categories.reverse();
  categories = categories.slice(0, spec.limit ?? 300);
  const visible = new Set(categories);

  const seriesNames = new Set<string>();
  if (spec.series) {
    for (const row of rows) seriesNames.add(formatCategory(spec.series, row[spec.series]));
  } else {
    seriesNames.add('');
  }

  const yList = yCols.length ? yCols : [{ name: '__count__', kind: 'number' } as ColumnProfile];
  const series: Array<Record<string, unknown>> = [];
  for (const yCol of yList) {
    for (const group of seriesNames) {
      const data = categories.map((cat) => {
        const bucket = rows.filter((row) => {
          const xMatch = asString(row[xCol.name]) === cat && visible.has(cat);
          const sMatch = !spec.series || formatCategory(spec.series, row[spec.series]) === group;
          return xMatch && sMatch;
        });
        const values = bucket.map((row) => yCol.name === '__count__' ? 1 : asNumber(row[yCol.name])).filter((v): v is number => v !== null);
        return yCol.name === '__count__' ? (bucket.length || null) : aggregate(values, spec.agg ?? 'mean');
      });
      const name = [group, yCol.name === '__count__' ? '记录数' : pretty(yCol.name)].filter(Boolean).join(' · ');
      series.push({
        name,
        type: 'line',
        data,
        smooth: true,
        showSymbol: false,
        symbolSize: 5,
        connectNulls: false,
        lineStyle: { width: 2 },
        emphasis: { focus: 'series' },
      });
    }
  }

  const option = baseOption(spec);
  option.xAxis = {
    ...(option.xAxis as Record<string, unknown>),
    type: 'category',
    data: categories,
    boundaryGap: false,
    ...axisName(pretty(xCol.name)),
    axisLabel: { color: AXIS_LABEL, fontSize: 11, hideOverlap: true, rotate: categories.some((c) => c.length > 10) ? 22 : 0 },
  };
  option.yAxis = { ...(option.yAxis as Record<string, unknown>), ...axisName(yCols.length === 1 ? pretty(yCols[0].name) : '数值') };
  option.series = series;
  if (categories.length > 80) {
    option.dataZoom = [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', height: 14, bottom: 16, borderColor: AXIS_LINE, fillerColor: 'rgba(70,200,255,.12)', textStyle: { color: AXIS_LABEL } },
    ];
    (option.grid as Record<string, unknown>).bottom = 78;
  }
  return {
    option,
    summary: `折线图：${pretty(xCol.name)} × ${yCols.map((c) => pretty(c.name)).join('、')}${spec.series ? `，按 ${pretty(spec.series)} 分组` : ''}，共 ${rows.length} 行。`,
    usedRows: rows.length,
  };
}

function buildBar(spec: ChartSpec, dataset: Dataset, rows: Row[]): ChartBuildResult {
  const xCol = columnOf(dataset, spec.x);
  if (!xCol) throw new Error('柱状图缺少 x 字段');
  const yCols = (Array.isArray(spec.y) ? spec.y : spec.y ? [spec.y] : [])
    .map((name) => columnOf(dataset, name))
    .filter((c): c is ColumnProfile => Boolean(c));
  const yList = yCols.length ? yCols : [{ name: '__count__', kind: 'number' } as ColumnProfile];

  const categorySet = new Set<string>();
  for (const row of rows) categorySet.add(formatCategory(xCol.name, row[xCol.name]));
  let categories = [...categorySet];
  const numeric = xCol.kind === 'number' || categories.every((c) => Number.isFinite(Number(c)));
  categories.sort((a, b) => (numeric ? Number(a) - Number(b) : a.localeCompare(b, 'zh-CN')));

  const valueFor = (cat: string, yCol: ColumnProfile, group?: string): number | null => {
    const bucket = rows.filter((row) => {
      const xMatch = formatCategory(xCol.name, row[xCol.name]) === cat;
      const sMatch = !spec.series || !group || formatCategory(spec.series, row[spec.series]) === group;
      return xMatch && sMatch;
    });
    if (yCol.name === '__count__') return bucket.length || null;
    const values = bucket.map((row) => asNumber(row[yCol.name])).filter((v): v is number => v !== null);
    return aggregate(values, spec.agg ?? 'mean');
  };

  if (spec.sort !== 'none') {
    const score = new Map<string, number>();
    for (const cat of categories) {
      const v = valueFor(cat, yList[0]);
      score.set(cat, v ?? Number.NEGATIVE_INFINITY);
    }
    categories = sortEntries(categories.map((c) => [c, score.get(c) ?? null]), spec.sort).map(([c]) => c);
  }
  categories = categories.slice(0, spec.limit ?? 30);

  const groups = new Set<string>();
  if (spec.series) {
    for (const row of rows) groups.add(formatCategory(spec.series, row[spec.series]));
  }
  const groupList = groups.size ? [...groups] : [''];
  const series: Array<Record<string, unknown>> = [];
  for (const yCol of yList) {
    for (const group of groupList) {
      series.push({
        name: [group, yCol.name === '__count__' ? '记录数' : pretty(yCol.name)].filter(Boolean).join(' · '),
        type: 'bar',
        data: categories.map((cat) => valueFor(cat, yCol, group)),
        barMaxWidth: 46,
        itemStyle: { borderRadius: [4, 4, 0, 0] },
        emphasis: { focus: 'series' },
      });
    }
  }

  const option = baseOption(spec);
  option.xAxis = {
    ...(option.xAxis as Record<string, unknown>),
    type: 'category',
    data: categories,
    ...axisName(pretty(xCol.name)),
    axisLabel: { color: AXIS_LABEL, fontSize: 11, hideOverlap: true, rotate: categories.some((c) => c.length > 8) ? 20 : 0 },
  };
  option.yAxis = { ...(option.yAxis as Record<string, unknown>), ...axisName(yCols.length === 1 ? pretty(yCols[0].name) : '数值') };
  option.series = series;
  return {
    option,
    summary: `柱状图：${pretty(xCol.name)} × ${yCols.map((c) => pretty(c.name)).join('、') || '记录数'}，共 ${rows.length} 行。`,
    usedRows: rows.length,
  };
}

function buildBoxplot(spec: ChartSpec, dataset: Dataset, rows: Row[]): ChartBuildResult {
  const xCol = columnOf(dataset, spec.x);
  const yName = Array.isArray(spec.y) ? spec.y[0] : spec.y;
  const yCol = columnOf(dataset, yName);
  if (!xCol || !yCol) throw new Error('箱线图缺少 x 或 y 字段');

  const groups = new Map<string, number[]>();
  for (const row of rows) {
    const key = spec.series
      ? `${formatCategory(xCol.name, row[xCol.name])} · ${formatCategory(spec.series, row[spec.series])}`
      : formatCategory(xCol.name, row[xCol.name]);
    const v = asNumber(row[yCol.name]);
    if (v === null) continue;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(v);
  }
  let entries = [...groups.entries()].map(([name, values]) => {
    const sorted = [...values].sort((a, b) => a - b);
    return {
      name,
      values: sorted,
      median: quantile(sorted, 0.5),
      box: [sorted[0], quantile(sorted, 0.25), quantile(sorted, 0.5), quantile(sorted, 0.75), sorted[sorted.length - 1]],
    };
  });
  if (spec.sort !== 'none') {
    entries = [...entries].sort((a, b) => (a.median - b.median) * (spec.sort === 'asc' ? 1 : -1));
  }
  entries = entries.slice(0, spec.limit ?? 30);

  const option = baseOption(spec);
  option.xAxis = {
    ...(option.xAxis as Record<string, unknown>),
    data: entries.map((e) => e.name),
    ...axisName(pretty(xCol.name)),
    axisLabel: { color: AXIS_LABEL, fontSize: 11, hideOverlap: true, rotate: entries.some((e) => e.name.length > 8) ? 20 : 0 },
  };
  option.yAxis = { ...(option.yAxis as Record<string, unknown>), ...axisName(pretty(yCol.name)) };
  option.series = [{
    name: pretty(yCol.name),
    type: 'boxplot',
    data: entries.map((e) => e.box),
    boxWidth: ['18%', '42%'],
    itemStyle: { color: 'rgba(70,200,255,.18)', borderColor: '#46c8ff', borderWidth: 1.5 },
    emphasis: { itemStyle: { borderColor: '#9be7ff' } },
  }];
  return {
    option,
    summary: `箱线图：${pretty(xCol.name)} 分组的 ${pretty(yCol.name)} 分布，共 ${entries.length} 组。`,
    usedRows: rows.length,
  };
}

function buildScatter(spec: ChartSpec, dataset: Dataset, rows: Row[]): ChartBuildResult {
  const xCol = columnOf(dataset, spec.x);
  const yName = Array.isArray(spec.y) ? spec.y[0] : spec.y;
  const yCol = columnOf(dataset, yName);
  if (!xCol || !yCol) throw new Error('散点图缺少 x 或 y 字段');

  const groups = new Map<string, Array<[number, number]>>();
  for (const row of rows) {
    const x = asNumber(row[xCol.name]);
    const y = asNumber(row[yCol.name]);
    if (x === null || y === null) continue;
    const key = spec.series ? formatCategory(spec.series, row[spec.series]) : '';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push([x, y]);
  }
  const limit = spec.limit ?? 800;
  const series = [...groups.entries()].map(([name, data], index) => ({
    name: name || pretty(yCol.name),
    type: 'scatter',
    data: data.slice(0, limit),
    symbolSize: 8,
    itemStyle: { opacity: 0.75, color: COLORS[index % COLORS.length] },
    emphasis: { focus: 'series' },
  }));

  const option = baseOption(spec);
  option.xAxis = {
    ...(option.xAxis as Record<string, unknown>),
    type: 'value',
    ...axisName(pretty(xCol.name)),
    axisLabel: { color: AXIS_LABEL, fontSize: 11 },
    splitLine: { lineStyle: { color: SPLIT_LINE, type: 'dashed' } },
  };
  option.yAxis = { ...(option.yAxis as Record<string, unknown>), ...axisName(pretty(yCol.name)) };
  option.series = series;
  return {
    option,
    summary: `散点图：${pretty(xCol.name)} 与 ${pretty(yCol.name)}，共 ${series.reduce((n, s) => n + s.data.length, 0)} 个点。`,
    usedRows: rows.length,
  };
}

function buildHeatmap(spec: ChartSpec, dataset: Dataset, rows: Row[]): ChartBuildResult {
  const xCol = columnOf(dataset, spec.x);
  const seriesCol = columnOf(dataset, spec.series);
  const yName = Array.isArray(spec.y) ? spec.y[0] : spec.y;
  const yCol = columnOf(dataset, yName);
  if (!xCol || !seriesCol || !yCol) throw new Error('热力图缺少 x / 分组 / y 字段');

  const xValues = new Set<string>();
  const yValues = new Set<string>();
  const buckets = new Map<string, number[]>();
  for (const row of rows) {
    const x = formatCategory(xCol.name, row[xCol.name]);
    const s = formatCategory(seriesCol.name, row[seriesCol.name]);
    const v = asNumber(row[yCol.name]);
    xValues.add(x);
    yValues.add(s);
    if (v === null) continue;
    const key = `${x}\u0000${s}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(v);
  }
  let xs = [...xValues];
  let ys = [...yValues];
  const numeric = xs.every((x) => Number.isFinite(Number(x)));
  xs.sort((a, b) => (numeric ? Number(a) - Number(b) : a.localeCompare(b, 'zh-CN')));
  ys.sort((a, b) => a.localeCompare(b, 'zh-CN'));
  xs = xs.slice(0, spec.limit ?? 60);
  ys = ys.slice(0, spec.limit ?? 60);

  const data: Array<[number, number, number]> = [];
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let yi = 0; yi < ys.length; yi += 1) {
    for (let xi = 0; xi < xs.length; xi += 1) {
      const values = buckets.get(`${xs[xi]}\u0000${ys[yi]}`) ?? [];
      if (!values.length) continue;
      const v = aggregate(values, spec.agg ?? 'mean');
      if (v === null) continue;
      data.push([xi, yi, Number(v.toFixed(4))]);
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  if (!Number.isFinite(min)) {
    min = 0;
    max = 1;
  }

  const option = baseOption(spec);
  option.xAxis = {
    ...(option.xAxis as Record<string, unknown>),
    type: 'category',
    data: xs,
    splitArea: { show: true, areaStyle: { color: ['rgba(18,38,58,.35)', 'rgba(12,28,44,.35)'] } },
  };
  option.yAxis = {
    type: 'category',
    data: ys,
    axisLine: { lineStyle: { color: AXIS_LINE } },
    axisTick: { show: false },
    axisLabel: { color: AXIS_LABEL, fontSize: 11 },
    splitArea: { show: true, areaStyle: { color: ['rgba(18,38,58,.35)', 'rgba(12,28,44,.35)'] } },
  };
  option.legend = { show: false };
  option.grid = { left: 24, right: 28, top: 48, bottom: 84, containLabel: true };
  option.visualMap = {
    min,
    max,
    precision: 2,
    calculable: true,
    orient: 'horizontal',
    left: 'center',
    bottom: 18,
    itemWidth: 12,
    itemHeight: 120,
    text: ['高', '低'],
    textStyle: { color: AXIS_LABEL },
    inRange: { color: ['#e6f7ff', '#9be7ff', '#46c8ff', '#ffd166'] },
  };
  option.series = [{
    name: pretty(yCol.name),
    type: 'heatmap',
    data,
    label: { show: data.length <= 30, color: '#082033', fontSize: 10, formatter: (p: { value: number[] }) => String(p.value[2]) },
    emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,.45)' } },
  }];
  return {
    option,
    summary: `热力图：${pretty(xCol.name)} × ${pretty(seriesCol.name)} 的 ${pretty(yCol.name)}，${data.length} 个单元格。`,
    usedRows: rows.length,
  };
}

function buildPie(spec: ChartSpec, dataset: Dataset, rows: Row[]): ChartBuildResult {
  const xCol = columnOf(dataset, spec.x);
  if (!xCol) throw new Error('饼图缺少 x 字段');
  const yName = Array.isArray(spec.y) ? spec.y[0] : spec.y;
  const yCol = columnOf(dataset, yName);
  const buckets = new Map<string, number[]>();
  for (const row of rows) {
    const key = formatCategory(xCol.name, row[xCol.name]);
    if (!buckets.has(key)) buckets.set(key, []);
    const v = yCol ? asNumber(row[yCol.name]) : 1;
    if (v !== null) buckets.get(key)!.push(v);
  }
  let entries: [string, number | null][] = [...buckets.entries()].map(([name, values]) => [name, aggregate(values, yCol ? 'sum' : 'count')]);
  entries = sortEntries(entries, spec.sort === 'none' ? 'desc' : spec.sort).slice(0, spec.limit ?? 20);
  const option = baseOption(spec);
  option.tooltip = { ...(option.tooltip as Record<string, unknown>), trigger: 'item', formatter: '{b}<br/>{c}（{d}%）' };
  option.legend = { ...(option.legend as Record<string, unknown>), orient: 'vertical', left: 12, top: 96, bottom: 24 };
  option.grid = { left: 0, right: 0, top: 0, bottom: 0 };
  option.series = [{
    name: pretty(xCol.name),
    type: 'pie',
    radius: ['38%', '68%'],
    center: ['54%', '56%'],
    data: entries.map(([name, value]) => ({ name, value: value ?? 0 })),
    label: { color: AXIS_LABEL, formatter: '{b}\n{d}%', fontSize: 11 },
    labelLine: { lineStyle: { color: AXIS_LINE } },
    itemStyle: { borderColor: '#071421', borderWidth: 2 },
    emphasis: { scale: true, scaleSize: 5 },
  }];
  return {
    option,
    summary: `饼图：${pretty(xCol.name)} 的${yCol ? pretty(yCol.name) : '记录数'}占比，共 ${entries.length} 类。`,
    usedRows: rows.length,
  };
}

function buildHistogram(spec: ChartSpec, dataset: Dataset, rows: Row[]): ChartBuildResult {
  const yName = Array.isArray(spec.y) ? spec.y[0] : spec.y;
  const yCol = columnOf(dataset, yName);
  if (!yCol) throw new Error('直方图缺少数值字段');
  const values = rows.map((row) => asNumber(row[yCol.name])).filter((v): v is number => v !== null);
  if (!values.length) throw new Error('该字段没有可用的数值');
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bins = 20;
  const width = max === min ? 1 : (max - min) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.floor((v - min) / width));
    counts[idx] += 1;
  }
  const labels = counts.map((_, i) => {
    const start = min + i * width;
    const end = start + width;
    return `${start.toFixed(2)}–${end.toFixed(2)}`;
  });
  const option = baseOption(spec);
  option.xAxis = {
    ...(option.xAxis as Record<string, unknown>),
    data: labels,
    ...axisName(pretty(yCol.name)),
    axisLabel: { color: AXIS_LABEL, fontSize: 10, rotate: 30, hideOverlap: true },
  };
  option.yAxis = { ...(option.yAxis as Record<string, unknown>), ...axisName('频数') };
  option.series = [{
    name: '频数',
    type: 'bar',
    data: counts,
    barMaxWidth: 30,
    itemStyle: { borderRadius: [3, 3, 0, 0], color: '#46c8ff' },
  }];
  return {
    option,
    summary: `直方图：${pretty(yCol.name)} 的分布（${bins} 个区间，${values.length} 个数值）。`,
    usedRows: rows.length,
  };
}

export function buildChart(dataset: Dataset, spec: ChartSpec): ChartBuildResult {
  const filtered = filterRows(dataset, spec.filter);
  if (!filtered.rows.length) throw new Error('筛选后没有数据，请调整筛选条件。');
  let result: ChartBuildResult;
  switch (spec.type) {
    case 'line': result = buildLine(spec, dataset, filtered.rows); break;
    case 'bar': result = buildBar(spec, dataset, filtered.rows); break;
    case 'scatter': result = buildScatter(spec, dataset, filtered.rows); break;
    case 'boxplot': result = buildBoxplot(spec, dataset, filtered.rows); break;
    case 'heatmap': result = buildHeatmap(spec, dataset, filtered.rows); break;
    case 'pie': result = buildPie(spec, dataset, filtered.rows); break;
    case 'histogram': result = buildHistogram(spec, dataset, filtered.rows); break;
    default: throw new Error(`不支持的图表类型：${(spec as ChartSpec).type}`);
  }
  return {
    ...result,
    warning: filtered.warning,
  };
}
