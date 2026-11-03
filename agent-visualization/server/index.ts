import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { addUpload, APP_DIR, getDataset, listDatasets, toMeta } from './datasets';
import { buildChart } from './chart';
import { rulePlan } from './planner';
import { getLlmConfig, llmConfigured, planChart } from './llm';

const app = express();
const PORT = Number(process.env.PORT || 8787);

app.use(cors());
app.use(express.json({ limit: '2mb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!file.originalname.toLowerCase().endsWith('.csv')) {
      cb(new Error('只支持 .csv 文件'));
      return;
    }
    cb(null, true);
  },
});

app.get('/api/health', (_req, res) => {
  const config = getLlmConfig();
  res.json({
    ok: true,
    service: 'industrial-viz-agent',
    llm: {
      configured: llmConfigured(),
      mode: llmConfigured() ? 'openai-compatible' : 'rules-only',
      model: config.model,
      baseUrl: config.baseUrl,
    },
    datasets: listDatasets().length,
  });
});

app.get('/api/datasets', (_req, res) => {
  res.json({ datasets: listDatasets() });
});

app.get('/api/datasets/:id', (req, res) => {
  const dataset = getDataset(req.params.id);
  if (!dataset) {
    res.status(404).json({ error: '数据集不存在' });
    return;
  }
  res.json(toMeta(dataset));
});

app.post('/api/datasets/upload', upload.single('file'), (req, res) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: '请选择 CSV 文件' });
      return;
    }
    const dataset = addUpload(req.file.buffer, req.file.originalname);
    res.json(toMeta(dataset));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.status(400).json({ error: `CSV 解析失败：${message}` });
  }
});

app.post('/api/chat', async (req, res) => {
  const { datasetId, question } = req.body ?? {};
  if (typeof datasetId !== 'string' || typeof question !== 'string' || !question.trim()) {
    res.status(400).json({ error: '需要 datasetId 和非空 question' });
    return;
  }
  if (question.length > 500) {
    res.status(400).json({ error: '问题过长，请控制在 500 字以内' });
    return;
  }
  const dataset = getDataset(datasetId);
  if (!dataset) {
    res.status(404).json({ error: '数据集不存在，请重新选择或上传' });
    return;
  }
  try {
    const plan = await planChart(dataset, question.trim(), rulePlan);
    const built = buildChart(dataset, plan.spec);
    res.json({
      answer: plan.spec.explanation || `已生成${plan.spec.title}。`,
      spec: plan.spec,
      option: built.option,
      planner: plan.planner,
      warning: plan.warning || built.warning,
      summary: built.summary,
      usedRows: built.usedRows,
      preview: {
        columns: dataset.columns.map((c) => c.name),
        rows: dataset.rows.slice(0, 8),
      },
      dataset: {
        id: dataset.id,
        name: dataset.name,
        rowCount: dataset.rows.length,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.status(500).json({ error: `生成图表失败：${message}` });
  }
});

const webDist = path.join(APP_DIR, 'web', 'dist');
if (fs.existsSync(webDist)) {
  app.use(express.static(webDist));
  app.use((req, res, next) => {
    if (req.method === 'GET' && !req.path.startsWith('/api')) {
      res.sendFile(path.join(webDist, 'index.html'));
      return;
    }
    next();
  });
}

app.use((error: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  const message = error instanceof Error ? error.message : String(error);
  res.status(400).json({ error: message });
});

app.listen(PORT, () => {
  const mode = llmConfigured() ? `LLM ${getLlmConfig().model}` : '规则引擎';
  console.log(`[viz-agent] http://localhost:${PORT}  (${mode})`);
});
