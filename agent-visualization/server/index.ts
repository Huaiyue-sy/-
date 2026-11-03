import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { addUpload, APP_DIR, getDataset, listDatasets, toMeta } from './datasets';
import { buildChart } from './chart';
import { analyzeDetections, getLlmConfig, llmConfigured, normalizeHistory, replyWithLlm } from './llm';
import {
  confidenceThreshold,
  detectDefects,
  summarizeDetections,
  templateAnswer,
  visionConfigured,
  visionInfo,
} from './vision';

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

const imageUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024, files: 1 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'].includes(ext)) {
      cb(new Error('只支持 JPG / PNG / BMP / WebP / TIFF 图片'));
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
      mode: llmConfigured() ? 'online' : 'unconfigured',
      model: config.model,
      baseUrl: config.baseUrl,
    },
    datasets: listDatasets().length,
    vision: visionInfo(),
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

function llmFailureReason(message: string): string {
  if (message.startsWith('HTTP 401')) return '在线模型未授权（401）';
  if (message.startsWith('HTTP 402')) return '在线模型额度不足（402）';
  if (message.startsWith('HTTP 429')) return '在线模型请求过于频繁（429）';
  if (/abort|timeout/i.test(message)) return '在线模型响应超时';
  return '在线模型暂时不可用';
}

app.post('/api/chat', async (req, res) => {
  const { datasetId, question, history } = req.body ?? {};
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

  const text = question.trim();
  const meta = {
    id: dataset.id,
    name: dataset.name,
    rowCount: dataset.rows.length,
  };

  if (!llmConfigured()) {
    res.status(503).json({
      error: '未配置在线模型，请在 agent-visualization/.env 或项目根目录 .env 中配置 VIZ_LLM_API_KEY / DEEPSEEK_API_KEY。',
    });
    return;
  }

  try {
    const turns = normalizeHistory(history);
    const last = turns[turns.length - 1];
    if (last && last.role === 'user' && last.content === text) turns.pop();
    const result = await replyWithLlm(dataset, text, turns);

    if (!result.spec) {
      res.json({
        kind: 'chat',
        answer: result.reply,
        planner: 'llm',
        warning: result.chartWarning,
        dataset: meta,
      });
      return;
    }

    let built;
    try {
      built = buildChart(dataset, result.spec);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.json({
        kind: 'chat',
        answer: result.reply,
        planner: 'llm',
        warning: `图表生成失败，已改为文字回答：${message}`,
        dataset: meta,
      });
      return;
    }

    res.json({
      kind: 'chart',
      answer: result.reply,
      spec: result.spec,
      option: built.option,
      planner: 'llm',
      warning: result.chartWarning || built.warning,
      summary: built.summary,
      usedRows: built.usedRows,
      preview: {
        columns: dataset.columns.map((c) => c.name),
        rows: dataset.rows.slice(0, 8),
      },
      dataset: meta,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`[viz-agent] 在线模型调用失败：${message.slice(0, 200)}`);
    res.status(502).json({
      error: `${llmFailureReason(message)}，请检查网络、密钥和模型配置后重试。`,
    });
  }
});

app.post('/api/vision/detect', imageUpload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: '请选择一张图片' });
      return;
    }
    if (!visionConfigured()) {
      res.status(503).json({
        error: '未找到 YOLO 模型，请先完成第 3 周 YOLO 训练，或配置 VIZ_YOLO_MODEL 指向权重文件。',
      });
      return;
    }

    const question = typeof req.body?.question === 'string'
      ? req.body.question.trim().slice(0, 500)
      : '';
    const requested = Number(req.body?.conf);
    const conf = Number.isFinite(requested) && requested > 0 && requested < 1
      ? Math.min(0.95, Math.max(0.01, requested))
      : confidenceThreshold();

    const tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'viz-yolo-'));
    const ext = path.extname(req.file.originalname).toLowerCase() || '.jpg';
    const imagePath = path.join(tempDir, `input${ext}`);
    try {
      await fs.promises.writeFile(imagePath, req.file.buffer);
      const detection = await detectDefects(imagePath, conf);
      const { counts, summary } = summarizeDetections(detection.detections, conf);

      let answer = templateAnswer(detection.detections, counts, conf);
      let planner: 'llm' | undefined;
      let warning: string | undefined;

      if (llmConfigured()) {
        try {
          answer = await analyzeDetections({
            question,
            detections: detection.detections,
            width: detection.width,
            height: detection.height,
            conf,
          });
          planner = 'llm';
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          warning = `在线模型解读失败，已返回检测摘要：${message.slice(0, 160)}`;
        }
      }

      res.json({
        kind: 'vision',
        answer,
        image: `data:image/jpeg;base64,${detection.annotatedImage}`,
        detections: detection.detections,
        counts,
        summary,
        model: detection.model,
        conf,
        width: detection.width,
        height: detection.height,
        elapsedMs: detection.elapsedMs,
        planner,
        warning,
      });
    } finally {
      await fs.promises.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`[viz-agent] 缺陷检测失败：${message.slice(0, 200)}`);
    res.status(500).json({ error: `缺陷检测失败：${message}` });
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
  const mode = llmConfigured() ? `在线模型 ${getLlmConfig().model}` : '未配置在线模型（提问将报错）';
  console.log(`[viz-agent] http://localhost:${PORT}  (${mode})`);
});
