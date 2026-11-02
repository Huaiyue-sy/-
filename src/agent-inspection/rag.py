"""推理模块的知识检索（RAG）：TF-IDF 检索工艺文件，回答"缺陷与工艺原因"问题。

检索器用字符级 n-gram，不需要分词依赖；工艺文件在 craft_docs/。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CraftRAG:
    def __init__(self, docs_dir: str | Path = None):
        root = Path(docs_dir) if docs_dir else Path(__file__).parent / "craft_docs"
        self.files = sorted(root.glob("*.md"))
        if not self.files:
            raise FileNotFoundError(f"{root} 下没有工艺文件。")
        self.docs = [(f, f.read_text(encoding="utf-8")) for f in self.files]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                                          sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([text for _, text in self.docs])

    def retrieve(self, query: str, top_k: int = 2) -> list[tuple[Path, float]]:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.files[i], float(sims[i])) for i in order if sims[i] > 1e-6]

    def snippet(self, path: Path, max_chars: int = 400) -> str:
        text = Path(path).read_text(encoding="utf-8")
        return text[:max_chars] + ("…" if len(text) > max_chars else "")

    def causes_for_class(self, class_name: str) -> str:
        """给定缺陷类别名，返回对应工艺文件的常见原因摘要。"""
        for f, text in self.docs:
            if f.stem.lower().startswith(class_name.lower()):
                lines = [ln for ln in text.splitlines() if ln.strip()]
                try:
                    start = next(i for i, ln in enumerate(lines) if "常见工艺原因" in ln)
                    end = next((i for i in range(start + 1, len(lines))
                                if lines[i].startswith("## ")), len(lines))
                    return "\n".join(lines[start:end]).strip()
                except StopIteration:
                    return self.snippet(f, 300)
        hits = self.retrieve(class_name, top_k=1)
        return self.snippet(hits[0][0], 300) if hits else "未检索到相关工艺文件。"
