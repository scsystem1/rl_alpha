# 近期 Alpha 五窗口训练与评估

配置入口：`configs/experiment/recent_alpha_rolling.yaml`。协议版本为 `recent_alpha_v1`。本次实现不启动正式训练，也不把合成测试当作真实市场或 GPU 训练结果。

## 固定实验

| 测试年 | 搜索期 | 权重校准期 | 测试期 |
| --- | --- | --- | --- |
| 2021 | 2018-07 至 2020-06 | 2020-07 至 2020-12 | 2021 全年 |
| 2022 | 2019-07 至 2021-06 | 2021-07 至 2021-12 | 2022 全年 |
| 2023 | 2020-07 至 2022-06 | 2022-07 至 2022-12 | 2023 全年 |
| 2024 | 2021-07 至 2023-06 | 2023-07 至 2023-12 | 2024 全年 |
| 2025 | 2022-07 至 2024-06 | 2024-07 至 2024-12 | 2025 全年 |

每窗使用 Random、GP、Base-LLM、GRPO，seed 为 0、1、2，共 60 个单元。每单元 100 轮，每轮 8 候选，pool 容量 20。不同单元独立初始化搜索器、模型适配、优化器、pool 和搜索记忆。

两年搜索期分四个半年，OOF 为第 1 段拟合→第 2 段评分、第 2 段拟合→第 3 段评分、第 3 段拟合→第 4 段评分。每个拟合/评分段最低有效日数 80，有效日比例及股票观察覆盖率均为 80%。各段都截掉未在段末成熟的标签；在交易日网格上，信号 t 的标签使用 t+2 至 t+21 的收益。

面板按明确日期加载，因子计算额外保留此前最多 252 个交易日的历史，这些历史不进入监督样本。二十日标签从该区间已有 `daily_total_return` 在内存重算，因此可以恢复旧 2018/2021 分割边界处原本合法的标签；不读取区间结束后的收益，不重建原始数据、风险面板或磁盘标签缓存。

搜索只访问搜索期数据。预算结束后保存实际终态 pool，包括空池失败状态，禁止用校准指标选择历史快照。校准期仅对该 pool 拟合一次 ridge 权重，不拼回搜索期再拟合；校准 RNIC 只标作拟合内诊断。测试期间公式和权重冻结。

## R1 与 paired-LCB

默认 reward 为 `r1_oof`，ridge 为 0.01。不默认运行 LCB。需要预定对照时，在开始实验前将 `experiment.rewards` 改为 `[r1_oof, r2_paired_oof]`，此时为 120 个单元，必须使用新的实验 ID。

两个 reward 共用因子变换、OOF 日期、支持门槛、拟合权重、逐日 RNIC、候选准入和 GRPO reward 映射。对同一旧/新 pool：

\[
d_t=RNIC_{new,t}-RNIC_{old,t},\quad R_1=\bar d,\quad R_2=\bar d-0.5SE_{HAC}(d).
\]

SE 使用 lag=20 的 gap-aware HAC。允许将 paired-LCB 系数显式设为 0 做等价性测试。`r2_lcb` 保留其原有定义，不能当作这次 paired 对照。

## 运行

在已有数据、AlphaGen 修改版 gplearn、QuantEvolver/Verl 和 Qwen 模型的训练环境中运行。先安装项目依赖，配置 `paths` 或 `RLALPHA_*` 环境变量，并核对 GPU 编号。默认沿用原调度器的 GPU 配置，不能直接套用到其他机器。

正式运行前将配置中的 `experiment.auto_start_expensive_jobs` 设为 `true`；默认 `false` 会拒绝启动包含 LLM 的矩阵。这只是沿用现有启动开关。所有配置应在首次运行前确定，不能先以一个配置跑部分单元，再改配置沿用同一实验 ID。

```bash
python -m rlalpha.cli matrix run \
  --config configs/experiment/recent_alpha_rolling.yaml \
  --experiment-id recent_alpha_rolling_v1

python -m rlalpha.cli evaluate run \
  --config configs/experiment/recent_alpha_rolling.yaml \
  --experiment-id recent_alpha_rolling_v1

python -m rlalpha.cli report build \
  --config configs/experiment/recent_alpha_rolling.yaml \
  --experiment-id recent_alpha_rolling_v1
```

外层依年顺序调用现有 matrix。`matrix run --method random --method gp` 可只调度 CPU 方法；这不代表完整实验已完成。测试只允许访问已冻结且通过身份校验的单元。缺少预定窗口、方法或 seed 的正式报告不会宣称完整。

结果路径：

```text
runs/<experiment>/
  rolling_manifest.json
  window_configs/test_<year>.yaml
  test_<year>/<method>/<reward>/seed_<n>/
    effective_config.yaml
    run_identity.json
    final_pool.json
    combiner.json
    test/metrics.json
    test/rnic_daily.parquet
    test/dollar_neutral_daily.parquet
    test/factor_significance.parquet
    test/factor_rnic_daily.parquet
    test/exposures.parquet
  rolling_evaluation.json
  report/report.md
  report/*.csv
  report/*.parquet
  report/figures/dollar_neutral_<year>_<reward>.png
  report/figures/dollar_neutral_<year>_<reward>.svg
  report/figures/dollar_neutral_<year>_<reward>_source.csv
```

窗口解析结果、日期、OOF 折及终态/校准/年度重置策略进入冻结配置和身份校验。允许相同配置、相同窗口续跑；跨窗、旧协议或变更配置需新实验 ID。直接 `search run` 应使用冻结的单窗口配置，不能传尚未解析的 rolling 配置。

## IC、推断和汇总

四项 IC 都先按日计算截面相关，再对有效交易日等权平均，输出各自有效日数。

| 指标 | 计算 |
| --- | --- |
| Raw IC | 每个公式每日 1%/99% 去极值、标准化，用校准期冻结的同一组权重合成，对原始前瞻收益算 Pearson |
| Raw rank IC | 同一原始合成分数对原始收益算 Spearman |
| RNIC | 沿用当前中性化部署分数及标签预处理，在固定评价截面上共同残差化，再算 Pearson |
| Rank RNIC | 对上述双方残差算 Spearman，即先残差化再排名 |

分数标准化不使用测试标签。保留固定股票支持规则、缺失因子零意见和现有标签去极值/标准化约定。因子中性化、收益残差化、逐因子统计及风险暴露诊断均保留。

年度及总体 RNIC/rank RNIC 输出 mean、HAC SE、t、双侧正态近似 p、95% moving-block bootstrap 区间。HAC lag=20，保留原始交易日数组中的 NaN；不将年度尾部标签缺口压成相邻观测。Bootstrap 默认 2,000 次、块长 20、固定 seed，按年份在原交易日数组上抽连续块并保留 NaN；所有方法和 seed 共用日期索引。跨年 bootstrap 每次先计算各年抽样均值，再按原始各年有效日数加权。

方法级推断先按同日三个预定 seed 求均值，再进行时间序列推断；任一 seed 缺失时该方法该日均值缺失，不静默改成两个 seed。总体均值按有效日数加权，不平均年度 t、p 或区间端点。seed 明细及 seed 间均值、样本标准差单独保存。总体推断条件于这些固定 seed 和已有历史年份。

GRPO 对 Base-LLM、GP、Random 的 RNIC/rank RNIC 比较使用同日、同 seed 差值再汇总。它们属于预定比较，p 值未做额外多重校正。单因子 BH 校正仍在各年度、各 pool、每项 RNIC 内独立进行，不把不同年份公式混成一组。

## 组合记账

只运行 dollar-neutral：按中性化部署分数取头尾各 20%，多空分别等权至 +0.5/-0.5，每五日调仓，四个二十日 sleeves，次日收盘执行，再下一交易日开始获得收益。保留单边成本 0/10 bps，Sharpe 的无风险基准为 0。

每年从空仓启动，最后一天正常获得持仓收益，再按净合并持仓绝对值之和计一次收盘清仓换手及成本。最后一天的持仓记录保留，用于收益、清仓及暴露核查。没有改变原 sleeve 固定权重记账约定，也未引入新的持仓漂移模型。

对每个 seed 的年度路径和五年串联日收益分别重算：

\[
R=\prod_t(1+r_t)-1,\quad CAGR=(1+R)^{252/N}-1,
\]
\[
Sharpe=\sqrt{252}\bar r/s_r,\quad MDD=\min_t[W_t/\max_{u\le t}W_u-1],\quad W_0=1.
\]

`annual_return` 保留旧算术年化字段，明确别名为 `annualized_mean_return`；主表展示总收益、CAGR、MDD、Sharpe。五年总体先对各 seed 计算这些指标，再汇总 seed 均值/标准差，不平均年度 Sharpe/MDD，不构造平均 seed 日收益来冒充平均 seed 表现。

全年交易日都参与组合回测，包括无成熟 IC 标签的最后约 21 日。个股持仓收益缺失沿用零贡献、不重新加杠杆的旧政策，缺失股票数、持仓权重覆盖及换手另行记录；组合级日收益数组存在 NaN 或非法财富路径时标为无效，不删除这些日期来算收益。零波动 Sharpe 为 NA。

五张年度图为三个 seed × 两种成本六面板，每面板比较四方法，共用纵轴和零收益基线，直接标注方法；PNG/SVG 和逐日作图源 CSV 一起保存。曲线仅由保存日收益精确复利得到，不重新回测、不平滑。

## 验证范围

新增测试覆盖日期/截尾、跨旧边界标签恢复、无校准搜索、终态 pool、worker 同步和续跑拒绝、R1/paired-LCB 等价关系、四项 IC 人工校验、gap-aware HAC、分层同步 bootstrap、清仓费用和末日收益、初始亏损回撤、零波动、缺失收益、seed 汇总、完整性拒绝及年度曲线数据一致性。

CPU 检查可运行 `pytest -q tests/unit/test_recent_*.py tests/integration/test_recent_pipeline.py`。完整回归仍依赖 AlphaGen 的仓库内修改版 gplearn；GRPO 的真实更新/恢复需 CUDA、模型权重及 Verl 环境；真实面板测试需已有数据。这些环境检查与合成测试分别报告，均不能替代正式 60 单元训练。

2026-09-12 本机验收：Python 3.12 的临时 CPU 环境中，225 项通过、3 项跳过、10 项排除。新增 49 项中 48 项通过，真实 AlphaGen GP 检查因仓库缺失跳过。其余跳过为 xgrammar、Verl 检查；排除的 10 项包括已确认缺少 AlphaGen/Verl 的 9 项既有测试，以及默认标记排除的 1 项真实面板测试。原有 NumPy timedelta 弃用警告仍存在。CLI 入口检查、`git diff --check`、五窗口合成端到端测试及六面板图视觉检查通过。

本轮真实训练验证未执行：本机没有 `/Users/stevens/Documents/rl_alpha/alphagen/gplearn`、训练所需的 QuantEvolver/Verl/CUDA 环境及已配置的真实数据目录。安装临时 PyTorch 只用于 CPU 张量检查；没有运行模型训练，也没有生成可作投资表现证据的五年真实结果。
