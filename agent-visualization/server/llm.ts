import path from 'node:path';
import dotenv from 'dotenv';
import { z } from 'zod';
import type { ChartSpec, Dataset } from './types';
import { pretty } from './labels';
import { APP_DIR, PROJECT_ROOT } from './datasets';

dotenv.config({ path: path.join(APP_DIR, '.env') });
dotenv.config({ path: path.join(PROJECT_ROOT, '.env') });

const SPEC_SCHEMA = z.object({
  type: z.enum(['line', 'bar', 'scatter', 'boxplot', 'heatmap', 'pie', 'histogram']),
  title: z.string().min(1).max(100).optional(),
  x: z.string().optional(),
  y: z.union([z.string(), z.array(z.string()).min(1).max(3)]).optional(),
  series: z.string().optional(),
  agg: z.enum(['mean', 'sum', 'count', 'none']).optional(),
  filter: z.object({
    column: z.string(),
    value: z.union([z.string(), z.number()]).transform(String),
  }).optional(),
  sort: z.enum(['asc', 'desc', 'none']).optional(),
  limit: z.number().int().min(1).max(500).optional(),
  explanation: z.string().max(400).optional(),
});

export interface LlmConfig {
  apiKey?: string;
  baseUrl: string;
  model: string;
}

export function getLlmConfig(): LlmConfig {
  const apiKey = process.env.VIZ_LLM_API_KEY || process.env.DEEPSEEK_API_KEY;
  const rawBase = process.env.VIZ_LLM_BASE_URL || process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com';
  const baseUrl = rawBase.trim().replace(/\/+$/, '').replace(/\/chat\/completions$/i, '');
  const model = process.env.VIZ_LLM_MODEL || process.env.DEEPSEEK_MODEL || 'deepseek-chat';
  return { apiKey, baseUrl, model };
}

export function llmConfigured(): boolean {
  return Boolean(getLlmConfig().apiKey);
}

function buildPrompt(dataset: Dataset, question: string): string {
  const profile = dataset.columns.map((c) => ({
    name: c.name,
    label: pretty(c.name),
    kind: c.kind,
    unique: c.unique,
    min: c.min,
    max: c.max,
    samples: c.samples.slice(0, 5),
  }));
  return [
    '你是工业数据可视化规划器。根据字段画像和用户问题，生成一个图表规格。',
    '只输出 JSON，不要输出 Markdown 或解释性前后缀。',
    'JSON 字段：type, title, x, y, series, agg, filter, sort, limit, explanation。',
    'type 只能是：line（折线）、bar（柱状）、scatter（散点）、boxplot（箱线）、heatmap（热力）、pie（饼图）、histogram（直方）。',
    '规则：',
    '1. x、y、series、filter.column 必须是字段列表中真实存在的 name，不得编造。',
    '2. y 是数值字段，支持字符串或最多 3 个字段的数组；x 是类别/时间/数值字段。',
    '3. 对比多个指标时把 y 写成数组，例如 ["test_acc","test_macroF1"]。',
    '4. 需要筛选某个轴承/工况/模型时，用 filter: {"column":"bearing","value":"Bearing1_1"}。',
    '5. 趋势用 line，类别对比用 bar，两个数值关系用 scatter，分组分布用 boxplot，两个类别交叉的数值用 heatmap，占比用 pie，单数值分布用 histogram。',
    '6. 如果问题只要求数量/占比，可以省略 y，用 agg="count"。',
    '7. title 用简洁中文；explanation 用一句话说明选图理由。',
    `数据集：${dataset.name}（${dataset.rows.length} 行）`,
    `字段画像：${JSON.stringify(profile)}`,
    `用户问题：${question}`,
  ].join('\n');
}

function extractJson(text: string): unknown {
  const cleaned = text.trim().replace(/^```(?:json)?/i, '').replace(/```$/i, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start < 0 || end < start) throw new Error('模型未返回 JSON');
  return JSON.parse(cleaned.slice(start, end + 1));
}

function validateSpec(dataset: Dataset, raw: unknown): ChartSpec {
  const parsed = SPEC_SCHEMA.parse(raw);
  const find = (name: string) => dataset.columns.find((c) => c.name === name);
  if (parsed.x && !find(parsed.x)) throw new Error(`x 字段不存在：${parsed.x}`);
  if (parsed.series && !find(parsed.series)) throw new Error(`series 字段不存在：${parsed.series}`);
  if (parsed.filter && !find(parsed.filter.column)) throw new Error(`筛选字段不存在：${parsed.filter.column}`);
  const ys = parsed.y === undefined ? [] : (Array.isArray(parsed.y) ? parsed.y : [parsed.y]);
  for (const y of ys) {
    const col = find(y);
    if (!col) throw new Error(`y 字段不存在：${y}`);
    if (col.kind !== 'number') throw new Error(`y 字段不是数值列：${y}`);
  }
  if (['boxplot', 'scatter', 'heatmap', 'histogram'].includes(parsed.type) && !ys.length) {
    throw new Error(`${parsed.type} 需要数值 y 字段`);
  }
  if (parsed.type === 'heatmap' && (!parsed.x || !parsed.series)) {
    throw new Error('heatmap 需要 x 和 series 两个分类字段');
  }
  if (['line', 'bar', 'pie', 'boxplot', 'heatmap'].includes(parsed.type) && !parsed.x) {
    throw new Error(`${parsed.type} 需要 x 字段`);
  }
  return {
    ...parsed,
    y: Array.isArray(parsed.y) ? parsed.y : parsed.y,
    title: parsed.title || `${ys.map(pretty).join('、') || '数据'} 可视化`,
    agg: parsed.agg ?? (ys.length ? 'mean' : 'count'),
    sort: parsed.sort ?? (parsed.type === 'line' ? 'asc' : 'none'),
    limit: parsed.limit ?? (parsed.type === 'line' ? 300 : 30),
  };
}

export async function planWithLlm(dataset: Dataset, question: string): Promise<ChartSpec> {
  const config = getLlmConfig();
  if (!config.apiKey) throw new Error('未配置 LLM API key');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 35_000);
  try {
    const response = await fetch(`${config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify({
        model: config.model,
        temperature: 0,
        max_tokens: 900,
        messages: [
          { role: 'system', content: '你是严格的 JSON 图表规划器，只输出合法 JSON。' },
          { role: 'user', content: buildPrompt(dataset, question) },
        ],
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}${body ? `：${body.slice(0, 160)}` : ''}`);
    }
    const data = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
    const content = data.choices?.[0]?.message?.content;
    if (!content) throw new Error('模型返回内容为空');
    return validateSpec(dataset, extractJson(content));
  } finally {
    clearTimeout(timer);
  }
}

export interface PlanResult {
  spec: ChartSpec;
  planner: 'llm' | 'rules';
  warning?: string;
}

export async function planChart(dataset: Dataset, question: string, rulePlan: (d: Dataset, q: string) => ChartSpec): Promise<PlanResult> {
  if (llmConfigured()) {
    try {
      return { spec: await planWithLlm(dataset, question), planner: 'llm' };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[viz-agent] LLM 调用失败，改用规则引擎：${message.slice(0, 200)}`);
      const reason = message.startsWith('HTTP 401')
        ? '在线模型未授权（401）'
        : message.startsWith('HTTP 402')
          ? '在线模型额度不足（402）'
          : '在线模型暂时不可用';
      return {
        spec: rulePlan(dataset, question),
        planner: 'rules',
        warning: `${reason}，已切换规则引擎。`,
      };
    }
  }
  return { spec: rulePlan(dataset, question), planner: 'rules' };
}
