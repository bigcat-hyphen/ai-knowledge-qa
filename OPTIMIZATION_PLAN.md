# AI 知识库问答助手 — 上传速度优化计划

> 创建时间: 2026-06-17
> 目标: 大文件上传处理速度提升 3-5 倍
> 当前状态: 7/8 完成，59/59 测试通过，0 个 Bug
> 测试环境: 16 核 CPU, ONNX Runtime 1.26.0 (CPU only)

---

## 一、性能瓶颈分析

当前上传流程:

```
前端上传文件 → load_document(解析) → split_text(切块) → create_collection(嵌入+入库) → 返回结果
```

经过代码审查，识别出以下瓶颈（按影响排序）:

| # | 瓶颈 | 位置 | 影响程度 | 预估提速 | 状态 |
|---|------|------|----------|----------|------|
| 1 | ~~嵌入批次过小 (EMBED_BATCH=100)~~ | rag_engine.py | 极高 | 2-3x | ✅ 已改为 512 |
| 2 | ~~FTS 全量重建~~ | rag_engine.py | 高 | 1.5-3x | ✅ 已改为增量触发器 |
| 3 | ~~OpenAI 客户端重复创建~~ | rag_engine.py | 中高 | 1.2-1.5x | ✅ 已缓存 |
| 4 | ~~线程池 worker 数偏低 (5)~~ | rag_engine.py | 中 | 1.2-1.5x | ✅ 动态计算 |
| 5 | ~~数据库写入未优化~~ | rag_engine.py | 中 | 1.1-1.3x | ✅ PRAGMA 已修复 |
| 6 | 嵌入与切块串行执行 | app.py | 中 | 1.1-1.2x | ⏭️ 跳过 |
| 7 | ~~Hash fallback 嵌入无 numpy 加速~~ | rag_engine.py | 低 | 1.1x | ✅ 已用 numpy |
| 8 | ~~扫描版 PDF OCR 线程数过多 + 无预处理~~ | rag_engine.py | 极高 | 1.1x | ✅ 已优化 |

---

## 二、优化方案

### OPT-01: 增大嵌入批次大小 ⭐ 最高优先级

**文件:** `rag_engine.py`
**问题:** `EMBED_BATCH = 100`，但 OpenAI `text-embedding-3-small` API 单次支持最多 2048 个 input。1000 个 chunks 需要 10 次 API 调用，每次都有 HTTP 往返开销。
**修复方案:** 将 `EMBED_BATCH` 改为 `512` 或 `1024`。
**状态:** ✅ 已完成 (2026-06-17) — `EMBED_BATCH` 改为 512

---

### OPT-02: FTS 增量更新替代全量重建

**文件:** `rag_engine.py`
**问题:** 每次 `create_collection()` 都执行 `INSERT INTO docs_fts(docs_fts) VALUES('rebuild')`，重建整个 FTS 索引。
**修复方案:** 添加 INSERT 触发器，删除 `rebuild` 语句。
**状态:** ✅ 已完成 (2026-06-17) — 添加 `docs_fts_insert` 触发器，删除 `rebuild` 语句

---

### OPT-03: OpenAI 客户端复用

**文件:** `rag_engine.py`
**问题:** `_create_client(model_config)` 每次调用都创建新的 OpenAI 客户端和 HTTP 连接。
**修复方案:** 按 `(api_key, base_url)` 缓存 client 实例。
**状态:** ✅ 已完成 (2026-06-17) — `_client_cache` + `_client_cache_lock` 缓存机制

---

### OPT-04: 增加嵌入线程池 worker 数

**文件:** `rag_engine.py`
**问题:** 固定 `max_workers=5`，大文件批次多时排队等待。
**修复方案:** 动态计算 worker 数。
**状态:** ✅ 已完成 (2026-06-17) — `max_workers = min(max(len(batches), 5), 15)`

---

### OPT-05: 数据库批量写入优化

**文件:** `rag_engine.py`
**问题:** 批量写入时没有利用 SQLite 性能优化手段。
**修复方案:** 在事务外设置 `PRAGMA synchronous=NORMAL` + `cache_size=-64000`，写入后恢复 `synchronous=FULL`。
**状态:** ✅ 已完成 (2026-06-17) — PRAGMA 移到事务之前，59 测试通过

---

### OPT-06: 流水线化 — 切块与嵌入并行

**文件:** `app.py`
**问题:** 切块完成后才开始嵌入，没有流水线化。切块本身很快（<100ms），实际收益有限。
**状态:** ⏭️ 跳过（可选，收益有限，切块本身 <100ms）

---

### OPT-07: Hash fallback 嵌入 numpy 加速

**文件:** `rag_engine.py`
**问题:** 无 API key 时的 hash fallback 使用纯 Python 循环。
**修复方案:** 使用 numpy 向量化 `np.add.at()`。
**状态:** ✅ 已完成 (2026-06-17) — numpy 向量化替代纯 Python 循环

---

### OPT-08: 扫描版 PDF OCR 加速

**文件:** `rag_engine.py`
**问题:** 扫描版 PDF（无文字层）处理极慢。实测 29MB/205 页的扫描 PDF 需要 497 秒（8.3 分钟），是同大小非扫描 PDF 的 200 倍。

**优化内容:**
1. `max_workers` 从 8 改为 4（减少 GIL 竞争）
2. 图像转灰度（PIL `convert('L')`，质量损失 <6%）
3. 两阶段架构：预渲染所有页面 → 纯 OCR 推理（消除线程内图像处理开销）

**实测结果（29MB / 205 页扫描 PDF）:**
| 版本 | 耗时 | 每页 | 提速 |
|------|------|------|------|
| 原始 (8 线程, RGB) | 497s | 2.42s | 1.0x |
| 优化后 (4 线程, 灰度, 两阶段) | 442-456s | 2.16s | **1.1x** |

**结论:** 提速约 10%。瓶颈在 RapidOCR 的 ONNX 推理本身（CPU-only），已触及纯 CPU OCR 的性能上限。进一步提速需：
- GPU 加速（需 CUDA 版 ONNX Runtime + 显卡）
- API OCR（已有 `_ocr_pdf_api` 实现，需配置 API key）

**状态:** ✅ 已完成 (2026-06-17) — 4 线程 + 灰度 + 两阶段架构

---

## 三、优化进度追踪

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| OPT-01: 增大嵌入批次 | ✅ 已完成 | 2026-06-17 | EMBED_BATCH 100→512 |
| OPT-02: FTS 增量更新 | ✅ 已完成 | 2026-06-17 | 添加 INSERT 触发器，移除全量 rebuild |
| OPT-03: 客户端复用 | ✅ 已完成 | 2026-06-17 | 按 (api_key, base_url) 缓存 client |
| OPT-04: 线程池调优 | ✅ 已完成 | 2026-06-17 | 动态 worker: min(max(batches,5),15) |
| OPT-05: DB 写入优化 | ✅ 已完成 | 2026-06-17 | PRAGMA 移到事务外，WAL+NORMAL, 64MB cache |
| OPT-06: 流水线化 | ⏭️ 跳过 | — | 可选，收益有限，切块 <100ms |
| OPT-07: Hash numpy 加速 | ✅ 已完成 | 2026-06-17 | numpy 向量化替代纯 Python 循环 |
| OPT-08: 扫描 PDF OCR 加速 | ✅ 已完成 | 2026-06-17 | 4 线程 + 灰度 + 两阶段，497s→442s (1.1x) |

**总进度: 7/8 项完成，1 项跳过** — 非扫描 PDF 已达秒级，扫描 PDF 受限于 CPU OCR 引擎

---

## 四、验证方法

每项优化完成后:

1. 运行 `python -m pytest tests/ -v` 确保功能不退化
2. 准备测试文件对比上传处理耗时
3. 更新本文档进度

**综合目标:** 大文件 (2000 chunks) 上传处理时间从当前水平降低到 1/3~1/5。
