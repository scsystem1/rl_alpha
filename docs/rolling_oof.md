# 三折滚动 reward（v8）

主方案为 `r1_oof`，对照为 `r2_paired_oof`。历史 `r0`、`r1`、`r2_lcb` 的 reward 定义保留；历史实验复现应使用原代码和原 prompt，不把 v8 结果与历史结果混写。

## 评分定义

| 折 | 权重拟合 A | 冻结权重评分 B |
|---|---|---|
| 1 | 2010–2012 | 2013–2014 |
| 2 | 2012–2014 | 2015–2016 |
| 3 | 2014–2016 | 2017–2018 |

每折对旧池与加入候选后的新池分别拟合 ridge，然后在 B 的相同日期和股票支持集上计算每日 RNIC 差值 `d`。`r1_oof = mean(d)`；`r2_paired_oof = mean(d) - 1.645 * HAC_SE(d)`，HAC lag=20。三段 B 的有效日期每日等权，既不要求逐折为正，也不截掉负增量。

标签为次日收盘进场、t+21 收盘退出。A 和 B 都按真实交易日位置剔除退出日期越界的标签，因子 lookback 保留此前历史。每折拟合和评分均至少需要 252 个有效日，同时达到 80% 日期和观测覆盖率；不足时报错，不退回样本内评分。

HAC 保留完整交易日轴：有效日期的中心化差值为 `d_t - mean(d)`，缺口贡献为零。Bartlett 加权协方差和除以有效评分日数的平方，再开方得到均值标准误。不会把被 purge 的日期删除后重新连接，也不会用三个折均值代替 daily 序列。

固定股票支持集、每日截面变换、因子缺失时零意见的规则沿用既有实现。三折共享变换结果及 daily sufficient statistics，只有小型 ridge 系统分别拟合。三个剪枝预选仍通过 saliency 确定，使用各折 saliency 的均值。add-only reward、剪枝方案排序、正式 replacement 复核使用同一个配对比较接口。

## 外层权重及记录

- `PoolScore.objective` / `mean_ic`：池的 OOF 平均 RNIC。
- `PoolScore.standard_error`：池级 OOF 均值的 HAC SE，仅作诊断。
- `PoolScore.weights` / 快照 `train.weights`：**完整训练期**拟合权重，供外层验证使用。
- 快照 `fold_weights`：各 A 的拟合权重；绝不把它们的平均值用于外层验证。
- 候选 `add_increment` / `post_prune_increment`：`mean_delta`、配对 `standard_error`、`penalty`、`reward`、有效日数和各折增量均值。

两个新 reward 的快照均按 2019–2021 的平均 RNIC 选择，权重来自完整 2010–2018。最终仍按既有 evaluator 在 train+validation 拟合，再冻结到 test。本轮没有把外层改成三年滚动；内部与外层权重估计窗口不同，需作为实验限制说明。

checkpoint、冻结 stage spec 和 manifest 使用 v8 语义。恢复时校验 reward 设置、折日期/交易日轴及 prompt contract；不兼容时必须使用新实验 ID。

## Prompt

Base-LLM 和 GRPO 共用 `unified_rolling_summary_v8`：完整公式池、预测期限、Balanced-22 风险基准、一个 OOF RNIC、每个公式一个有符号系数。系数为每折权重先按 L1 范数归一化、再跨折平均；它表示组合的使用方式，不等于因子重要性。R0/R1/旧 R2 的 prompt 数值也统一来自中性化三折诊断，每个池版本只计算一次。

主题提示、相关矩阵、逐折统计、显著性数值和长历史都不进入 prompt。六个价量输入和模型先验仍限制搜索空间，去掉提示不保证主题自动多样化。

实际锁定版本的 Qwen3.5-2B tokenizer/config 哈希已核验。普通 20 因子池为 763 tokens；从固定随机候选中选取较长合法公式组成的 20 因子池为 1813 tokens，均满足 4096 总长度及 128 输出预留。这是长度验收，不是理解能力或市场效果验证。具体公式、模板哈希及结果见 `rolling_oof_token_profile.json`。

```bash
python scripts/benchmark_prompts.py \
  --model /data/shared/huggingface/Qwen3.5-2B \
  --output /data/sunyuxiang/rl_alpha/runs/rolling_oof_token_profile.json
```

## 训练环境中的验收与对照

本地 CPU 已覆盖独立逐折参考、标签边界、评分标签不影响本折权重、固定支持、缓存/批量一致性、配对 HAC、剪枝/入池、恢复、GRPO worker 整批评分/重放、archive round-trip 和外层全历史权重。

2026-09-05 验收结果：可运行 CPU 子集 **158 passed，12 skipped，9 deselected**，`compileall` 和 `git diff --check` 通过。跳过项缺少 Torch/xgrammar；排除项涉及 real-data 标记及本机未安装的 AlphaGen、QuantEvolver、Verl 依赖。这不是完整 GPU 集成验收。本机缺少真实 processed panel 和 CUDA/模型权重，以下优化器 smoke、prompt 采样消融及市场效果矩阵尚未执行，须在现有训练环境运行。

先运行全训练期的两步 smoke，并用同一目录 `--resume` 检查恢复；各 reward 使用不同 run directory：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/smoke_grpo.py \
  --reward r1_oof --updates 2 \
  --train-start 2010-01-01 --train-end 2018-12-31 \
  --run-dir /data/sunyuxiang/rl_alpha/runs/rolling_oof_smoke_r1
```

`configs/experiment/rolling_oof.yaml` 固定 Random/Base-LLM/GRPO × 样本内 R1/OOF 均值/OOF 配对 LCB × seeds 0/1/2，每格 250 轮、每轮 8 个 proposals。GPU 映射沿用现有主机配置；确认运行环境后启用该文件已有的昂贵任务开关，使用新的实验 ID：

```bash
rlalpha matrix run --config configs/experiment/rolling_oof.yaml --experiment-id rolling_oof_v8
```

Prompt 有无摘要的离线对照不改变正式模板路由、不更新模型、不执行入池。输入必须是训练快照 archive；按预先指定的候选预算选取快照，禁止按 validation/test 成绩挑选。两个分支复用同一冻结池和相同随机种子序列，默认每池每分支 16 组 × 8 个 proposals：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_prompt_feedback.py \
  --processed-root /data/sunyuxiang/rl_alpha/processed \
  --snapshots /data/sunyuxiang/rl_alpha/runs/rolling_oof_v8/base_llm/r1_oof/seed_0/checkpoints/snapshots.jsonl \
  --budgets 400 800 1600 --groups 16 --seed 0 \
  --output /data/sunyuxiang/rl_alpha/runs/prompt_feedback_seed0.json
```

输出有效率、重复率、候选与池成员相关性、OOF 增量分布及逐候选记录。对 seeds 1/2 使用对应 archive 和独立输出路径重复。先比较时间切分，再比较配对惩罚与策略学习；不据此直接调整多个超参数。

内部 B 被搜索反复使用，属于训练反馈。OOF 不消除自适应搜索过拟合，1.645 也不是选择后置信保证；已查看的 2022–2025 测试结果只能提供探索性证据。参考：[Dwork 等，2015](https://proceedings.neurips.cc/paper_files/paper/2015/file/bad5f33780c42f2588878a9d07405083-Paper.pdf)。
