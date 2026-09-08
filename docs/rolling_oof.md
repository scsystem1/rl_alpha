# Expanding OOF、v7 prompt 与 GRPO 单一 reward（v9）

当前实现保留 `r1_oof`（配对 OOF 均值）与 `r2_paired_oof`（温和配对 LCB）。Base-LLM/GRPO 的实际提示文本恢复历史 `b121f43` 的 `unified_compact_v7`；checkpoint/reward 语义为 v9。历史 v8 实验应使用 `41ee884` 复现，不能用新默认值或新 shaping 恢复旧 run。

## 时间窗口与统计量

| 折 | 权重拟合 A | 冻结权重评分 B |
|---|---|---|
| 1 | 2010–2012 | 2013–2014 |
| 2 | 2010–2014 | 2015–2016 |
| 3 | 2010–2016 | 2017–2018 |

每折对旧池和加入候选后的新池分别拟合 ridge，在 B 的相同日期和固定股票支持集上计算每日 RNIC 差值 `d_t`。标签为次日收盘进场、t+21 收盘退出。A 和 B 都按真实交易日位置剔除标签退出日期越界的日期；保留因子所需历史。后折 A 可以使用前折已经成熟的评分标签，本折 B 的标签不能影响本折拟合。

```text
mu = mean(d_t over all finite scoring days)
se = gap_aware_mean_se(d_t, lag=20)
r1_oof raw delta = mu
r2_paired_oof raw delta = mu - 0.5 * se
```

原始增量和入池统计量保留负值，不要求三折分别为正。R2 的默认 critical_value 从 1.645 改为 0.5；旧 `r2_lcb` 的默认值同步为 0.5，但仍可显式指定 1.645。

### 为什么不按 3/5/7 年重新缩放 SE

目标是“每个未来评分日的平均增量”。因此均值按有效评分日数加权：`mu = sum(n_score,k * mu_k) / sum(n_score,k)`。训练日数不是评分样本数，不应按 3:5:7 加权，也不应将各折 SE 再乘或除 `sqrt(n_fit)`。按观测到的 SE 做逆方差加权还会改变原有的时间等权目标。

HAC 直接使用每个评分日的残差乘积，允许不同折具有不同的波动幅度；没有假设三个折同方差或独立。它保留完整交易日轴，purge 缺口贡献零中心化分数，但不计入样本数，不会把缺口两侧拼成相邻日。保持相对于整体均值的中心化，不通过逐折去均值隐藏折间表现差异。lag=20 与标签重叠有关，不随 A 的年数扩大。

这仍是实现出的 OOF 时间序列的近似不确定性度量，不完整覆盖训练样本重采样导致的权重误差、长期结构变化或自适应搜索选择偏差。共享/嵌套训练期也使三个折不是三次独立重复试验；本版不引入 refit bootstrap 或三个均值上的 t 检验。

新增可审计字段：候选的 `PoolIncrement.fold_valid_days`、`fold_standard_errors`；快照的 `fold_fit_valid_days`、`fold_score_valid_days`、`fold_standard_errors`、`score_aggregation` 和 `uncertainty_estimator`。候选级 SE 对应增量，快照级 SE 对应池 RNIC，二者不能混用。

## Ridge 与外层评估

默认 ridge 从 0.001 提高为 0.01，并同步到 reward factory、直接构造的 objective/combiner、配置模型和最终 evaluator。正式 runner 继续将 evaluation ridge 传给训练 reward，因此训练与外层使用一致的正则强度。所有参数仍可显式覆盖；调参后的结果属于新实验。

代码求解 `(G + ridge * I) w = p`，G 和 p 是每日截面矩的时间均值，不是全样本求和。因此 3/5/7 年拟合时 ridge 的单位相同，不再按样本量做额外除法。0.01 加强病态/共线方向上的收缩；这是合理的初始设置，并非已经在真实面板上验证的最优值。

`PoolScore.weights` 仍是完整 train 拟合权重，供 validation 使用；最终选定公式后仍在 train+validation 重拟合，再冻结用于 test。Expanding 内层更接近该累积拟合方式，但不能消除拟合长度和预测跨度的全部差异。外层快照选择、pool capacity、替换方案、搜索预算不在本次改动范围内。

## Prompt

Base-LLM 和 GRPO 共用原 v7 公式池、DSL 元素列表、diversity hints 和输出约束。没有逐因子权重或 OOF RNIC 摘要；训练入口也不再为 prompt 计算无用的三折数值摘要。结构化解码 grammar 保持不变。

prompt contract 继续对实际 user template 和 grammar 哈希。`scripts/benchmark_prompt_feedback.py` 是历史 v8 的有/无摘要对照；v7 下两条文本相同，因此该脚本会在开始加载数据/模型前明确停止，避免把同一 prompt 跑两遍误报为消融。历史脚本在 `41ee884` 使用。`scripts/benchmark_prompts.py` 仍可测量当前实际 prompt 长度，旧 `rolling_oof_token_profile.json` 仅是 v8 历史证据。

## 单一训练 reward 与 advantage

原始搜索 fitness、配对统计量和入池规则不经过 clipping。训练 reward 在已有 add-only 增量 `delta_i` 上使用一个映射：

```text
P = valid unique candidates with delta_i > min_delta
scale = max(median(delta_i for i in P), min_delta, 1e-5)
        # no P: finite floor for diagnostics; all valid rewards are zero

invalid / historical duplicate / near duplicate -> -1
valid, delta_i <= min_delta                     ->  0
valid, delta_i > min_delta                      ->  delta_i / (scale + delta_i)

advantage_i = reward_i - mean(reward_in_same_prompt_group)
```

默认 `min_delta=1e-5`，R2 的 `delta_i` 已扣掉 `0.5 * paired HAC SE`，并非只看原始均值正负。映射只从合格正增量中估计尺度，保留原有的数值下限和 softsign 上界；负候选数量和幅度不能压低稀疏正候选的 reward。相同 hash 在尺度估计中只出现一次。

没有新增 validity/quality 混合参数，也没有自定义 advantage estimator。实际 Verl 配置使用 `algorithm.adv_estimator=grpo` 和 `algorithm.norm_adv_by_std_in_grpo=false`，只减均值，不进行第二次标准差归一化。配置断言阻止误恢复为除标准差；domain 日志同步使用中心化后的数值。[Verl 官方实现](https://verl.readthedocs.io/en/latest/_modules/verl/trainer/ppo/core_algos.html#compute_grpo_outcome_advantage)支持这一选项。

| 八候选组 | reward | advantage |
|---|---|---|
| 全部合法、没有合格提升 | 全 0 | 全 0 |
| 2 个非法、6 个合法但无提升 | 2 个 -1，6 个 0 | 非法 -0.75，合法 +0.25；合法之间没有质量排名 |
| 1 个合格正增量、7 个合法但无提升 | 1 个 0.5，7 个 0 | 正候选 +0.4375，其余 -0.0625 |

同组多个合格正候选仍按 softsign 得分排序。若组内混有非法项，无提升的合法项仍可能获得正 advantage，这是单一 reward 对有效性的学习；不是强行宣称其提升了市场质量。所有合法项质量得分相同时没有质量排序。

实际更新仍包含 reference KL；全零 policy advantage 不承诺 optimizer 参数绝对不变。PPO clipping、KL、学习率、rollout 8、PPO epochs、优化器/调度器均保持既有配置。关闭标准差归一化改变了梯度尺度，需要真实训练轨迹验证，不能据此预先宣称 RNIC 提升。

组内重复只复用计算结果与诊断，其余重复 completion 的 `valid=false`、reward 为非法惩罚（默认 -1），不再复用代表公式的正 reward。非 GRPO coordinator 的重复/有效性失败也统一为 -1。普通方法的 proposal fitness 和入池仍使用原始增量，GP 不使用上述训练 reward 代替 tournament fitness。

本版仍使用 add-only 学习信号。满池时，正 add-only 候选可能在实际剪枝复核中未通过；不能把 `positive_reward_rate` 当作入池率。奖励 positive part 会损失负候选之间的市场排序信息，也不消除噪声偶发正尾部；这是为了学习有效生成及合格增量而做的明确取舍。

## 检查与运行

checkpoint schema=9，reward/pool semantics=`fixed-universe-expanding-positive-softsign-v9`。当前改动必须使用新实验 ID，不能续跑 v8 或历史 v7 的 optimizer state。

CPU 回归覆盖 expanding 的独立逐折参考、标签 purge、固定支持、异方差与不等评分长度的 HAC、正增量稀疏组、validity-only 组、原始 fitness/入池不被 shaping 改写、worker/主进程一致性、重复项、配置覆盖及 archive/checkpoint round-trip。实际 Verl advantage 测试需要 Torch/Verl；GPU smoke 仍需要远端模型与 CUDA。

2026-09-08 本地 CPU 可运行子集结果：**168 passed，13 skipped，8 deselected**。13 项跳过因 Torch/xgrammar 缺失；8 项排除包括 4 个需要本地 AlphaGen 的测试、3 个需要 Torch/Verl online dataset 的测试和 1 个默认排除的 real-data 测试。最初完整尝试在这些外部依赖导入处失败，没有将它们记为通过。历史 v7 在空池、单因子池、双因子池上的实际消息与当前消息逐字相同，DSL grammar 也相同；`compileall` 和 `git diff --check` 通过。此结果不构成 GPU 优化器或真实市场效果验收。

远端先运行针对性测试与两个 reward 的两步 smoke，再做常规矩阵：

```bash
python -m pytest -q tests/unit/test_walk_forward.py tests/unit/test_rewards_pool.py \
  tests/unit/test_prompts.py tests/unit/test_verl_grpo_adapter.py tests/unit/test_verl_stage_coordinator.py

CUDA_VISIBLE_DEVICES=3 python scripts/smoke_grpo.py \
  --reward r2_paired_oof --updates 2 \
  --train-start 2010-01-01 --train-end 2018-12-31 \
  --run-dir /data/sunyuxiang/rl_alpha/runs/expanding_v9_smoke_r2
```

`r1_oof` 使用独立目录做相同 smoke。检查实际生成的 Verl config 中 `norm_adv_by_std_in_grpo=false`，以及 `domain/no_improvement_groups`、`positive_reward_rate`、`advantage_std`、duplicate 与入池轨迹。模型更新测试通过后使用新 experiment ID，例如 `expanding_oof_v9`；本地修改不代表已执行远端实验。
