# RLAlpha 中文源码导读与实现审计

> 基线：commit `52d5048`，审计日期 2026-08-09。
>
> 读者：第一次接触本项目，但具备 Python、机器学习和基础量化知识的工程师或研究者。
>
> 原则：本文以当前 Python 执行路径和已落盘 artifact 为准；YAML、README 或设计名称只有在确实被代码消费时才被视为运行事实。

本文不是 API 列表，也不是 README 的扩写。它试图回答四个更具体的问题：一条原始 CRSP/Compustat 记录如何变成可搜索的二维因子矩阵；一个候选表达式如何得到 reward 并改变 pool；四种 proposer 的状态和反馈路径有何本质区别；冻结后的 pool 又如何进入统一 test universe、风险中性组合和跨方法报告。

文中使用四类标记：

- **当前实现**：当前 commit 实际执行的行为。
- **配置声明**：YAML 中写出的意图，不保证已经接入执行路径。
- **设计含义**：由实现推导出的实验含义。
- **潜在风险**：容易误读、复现实验时容易踩坑，或实现与配置不一致的地方；不代表本文顺手修改了代码。

## 目录与推荐阅读顺序

1. [项目全景](#1-项目全景)
2. [Data 与风险模型](#2-data-与风险模型)
3. [DSL、AST 与因子执行](#3-dslast-与因子执行)
4. [Reward、组合权重与 Pool](#4-reward组合权重与-pool)
5. [四种搜索方法](#5-四种搜索方法)
6. [统一搜索生命周期](#6-统一搜索生命周期)
7. [Validation 与 Test Evaluation](#7-validation-与-test-evaluation)
8. [实验配置、调度与运行](#8-实验配置调度与运行)
9. [配置消费审计](#9-配置消费审计)
10. [当前实验状态与已知边界](#10-当前实验状态与已知边界)
11. [测试、维护与源码索引](#11-测试维护与源码索引)

如果只想建立主干认知，建议依次读 1、2.7、4、6、7；如果要研究 GRPO，则在此基础上读 3、5.4 和 6.3；如果准备正式运行实验，再读 8、9、10。

---

## 1. 项目全景

### 1.1 端到端执行图

命令入口统一在 [`src/rlalpha/cli.py`](../src/rlalpha/cli.py#L1-L105)。真正的数据流如下：

```text
六份原始 Parquet
  │
  ├─ daily + delistings + membership
  │       │
  │       ├─ CRSP adjustment / delisting audit
  │       ├─ dense OHLCV/return [T, N]
  │       ├─ historical membership [T, N]
  │       └─ 20-day forward label [T, N]
  │
  └─ daily + market + CCM + Compustat + membership
          │
          └─ Balanced-22 exposures [T, N, 22]

PanelStore.load_split(name, history=252)
  │
  ├─ features: dict[str, [T_split+history, N]]
  ├─ label / membership / daily_return: [T_split+history, N]
  ├─ exposures: [T_split+history, N, 22]
  └─ target_slice: 只取 split 内日期
          │
          ▼
typed DSL Node ──evaluate──> signal [T_split, N]
          │
          ├─ exact hash 去重
          ├─ coverage / variability / near-duplicate 检查
          └─ 与 frozen pool 比较完整 pool objective
                  │
                  ▼
Random / GP / Base LLM / staged GRPO
                  │
                  ├─ train-only proposal/reward/admission
                  ├─ validation-only snapshot scoring
                  └─ validation objective 选 final_pool.json
                          │
                          ▼
所有 cell 的 final pool 同时冻结
  │
  ├─ 构造跨方法共同 test universe
  ├─ train+validation 重拟合最终 risk-neutral ridge weights
  ├─ test raw IC / RNIC / Rank RNIC
  ├─ dollar-neutral / fully-neutral portfolio
  └─ paired comparison / report.md
```

最关键的边界有三条：

1. 搜索和 reward 只能访问 train；[`SearchContext`](../src/rlalpha/search/models.py#L42-L52) 不携带 validation/test 指标。
2. validation 只在 pool version 变化时对不可变表达式 snapshot 打分，不反向改写 pool 或模型。
3. orchestrator 的预期顺序是全部 cell 完成后再打开 test；但 `finalize_experiment()` 本身只 glob 当前存在的 `final_pool.json`，不会核对配置矩阵是否齐全。这个实现边界在 10.5 节单独审计。

### 1.2 核心数组、轴和记号

本文使用下列记号：

| 记号 | 代码对象 | Shape | 含义 |
|---|---|---:|---|
| (T) | `dates` | 4,529 | 2008-01-02 至 2025-12-31 的共同交易日轴 |
| (N) | `permnos` | 891 | 全历史出现过的证券轴；不是任一天的成分数 |
| (K) | exposure columns | 22 | intercept + 11 industry dummies + 10 styles |
| (M) | pool entries | 0–20 | 当前因子池容量 |
| (G) | proposal group | 当前实验为 8 | Random/GP/Base 由 experiment `proposal_group_size` 决定；GRPO 在类中强制为 8 |
| `features[name]` | `SplitPanel.features` | `[time, asset]` | 调整后 OHLCV 或 total return |
| `label` | `forward_return_20d` | `[time, asset]` | 从 (t+2) 到 (t+21) 的复合收益 |
| `membership` | membership Zarr | `[time, asset]` | 当日是否为历史 S&P 500 member |
| `exposures` | risk Zarr | `[time, asset, 22]` | 当日 Balanced-22 设计矩阵 |
| `signal` | DSL evaluator output | `[target_time, asset]` | 某个表达式的数值矩阵 |
| `daily_ic` | `PoolScore.daily_ic` | `[target_time]` | 每日横截面 Pearson IC/RNIC |
| `weights` | pool ridge weights | `[M]` | 因子组合权重，不要求非负或和为 1 |

`PanelStore` 在每个 split 前面附加最多 252 个历史交易日，但 `SplitPanel.evaluate()` 最后用 `target_slice` 切回目标区间。因而“加载数组长度”和“搜索/评分长度”不是同一个概念，见 [`data/store.py`](../src/rlalpha/data/store.py#L71-L101)。

### 1.3 模块职责

| 子系统 | 主要文件 | 责任边界 |
|---|---|---|
| 配置/入口 | [`config.py`](../src/rlalpha/config.py), [`cli.py`](../src/rlalpha/cli.py) | 合并 YAML、环境变量覆盖、CLI 路由 |
| Data | [`data/panel.py`](../src/rlalpha/data/panel.py), [`data/store.py`](../src/rlalpha/data/store.py) | 原始数据到 dense panel；按 split 只读加载 |
| Risk | [`risk/builder.py`](../src/rlalpha/risk/builder.py), [`risk/neutralize.py`](../src/rlalpha/risk/neutralize.py) | Balanced-22 构造与横截面残差化 |
| DSL | [`dsl/ast.py`](../src/rlalpha/dsl/ast.py), [`dsl/evaluator.py`](../src/rlalpha/dsl/evaluator.py) | 类型安全表达式、canonical hash、数值执行 |
| Factor/Pool | [`factors/calculator.py`](../src/rlalpha/factors/calculator.py), [`factors/pool.py`](../src/rlalpha/factors/pool.py) | 标准化、IC、ridge combination、精确 admission |
| Reward | [`rewards/base.py`](../src/rlalpha/rewards/base.py) | R0/R1/R2 的共同矩与缓存计算 |
| Search | [`search/coordinator.py`](../src/rlalpha/search/coordinator.py), [`search/run.py`](../src/rlalpha/search/run.py) | proposer 的统一过滤、预算、pool、snapshot 生命周期 |
| GRPO | [`search/grpo/stage_coordinator.py`](../src/rlalpha/search/grpo/stage_coordinator.py) | 常驻 Verl/vLLM、在线 prompt、LoRA update、配对 checkpoint |
| Evaluation | [`evaluation/finalize.py`](../src/rlalpha/evaluation/finalize.py), [`evaluation/portfolio.py`](../src/rlalpha/evaluation/portfolio.py) | 单方法 test 事务、逐池 complete-case support、统计与回测 |
| Orchestration | [`matrix/runner.py`](../src/rlalpha/matrix/runner.py) | cell 展开、CPU/GPU 并发、OOM/restart |
| Reporting | [`reporting/build.py`](../src/rlalpha/reporting/build.py) | cell 表、paired comparisons、跨 seed 汇总 |

---

## 2. Data 与风险模型

### 2.1 六份原始数据如何被发现

[`DatasetContract`](../src/rlalpha/data/contracts.py#L6-L28) 为每类数据规定必要列集合。`discover_data_files()` 只扫描 raw root 顶层的 `*.parquet`，读取 Arrow schema，再判断某份文件是否完整包含某个 contract，见 [`data/discovery.py`](../src/rlalpha/data/discovery.py#L14-L29)。

这意味着文件名不是接口，schema 才是接口。例如 daily 必须至少包含：

```text
PERMNO, DlyCalDt, DlyRet,
DlyOpen, DlyHigh, DlyLow, DlyClose, DlyVol,
DlyCumFacPr, DlyCumFacShr,
DlyCap, ShrOut, DlyDelFlg
```

**当前实现**：某个 contract 恰好匹配一份文件才成功。零份会进入 `missing`，多份会进入 `ambiguous`，strict discovery 会直接抛出 `DataDiscoveryError`。

实际数据快照：

| Contract | 文件 | 行数 | 关键字段与下游用途 |
|---|---|---:|---|
| daily | `sp500_daily_2008_2025.parquet` | 3,131,836 | OHLCV、`DlyRet`、市值、SIC、调整因子 |
| membership | `sp500_membership.parquet` | 911 | `PERMNO` 的 inclusive active interval |
| market | `market_daily_2008_2025.parquet` | 4,529 | `vwretd` 用于 beta/residual volatility |
| ccm | `ccm_links.parquet` | 907 | `gvkey → lpermno` point-in-time link |
| fundamentals | `compustat_annual_2006_2025.parquet` | 14,531 | book equity、profitability、investment、leverage |
| delistings | `delistings_2008_2025.parquet` | 330 | 仅补充 flagged 且缺失的 `DlyRet` |

daily 覆盖 2008-01-02 至 2025-12-31，共 891 个历史 PERMNO。membership 中 911 条 `MbrFlg` 全是数据交付方使用的 `NORM`；14,531 条 Compustat 中 14,513 条满足项目的标准报表筛选条件。

逐表看 schema 与消费关系：

- **daily（27 列）**：`PERMNO/PERMCO` 和 CUSIP/Ticker 是证券标识，但 dense 资产轴只用 `PERMNO`；`DlyCalDt` 是时间轴；`DlyOpen/High/Low/Close/Vol` 经累计调整因子变成六个 DSL feature 中的五个；`DlyRet` 同时成为 `$return`、daily PnL 和 forward label 的底层 return；`DlyCap` 服务 size/book-to-market，`SICCD` 服务 FF12，`ShrOut` 只进入 QA identity；`DlyDelFlg` 触发退市补值。`DlyPrc/DlyRetx`、交易所/证券类型/交易状态等列当前落在分区 Parquet 中，但不进入 dense feature、membership 或 risk exposure。
- **membership（6 列）**：生产路径消费 `PERMNO/MbrStartDt/MbrEndDt/MbrFlg`；`INDFAM` 是 discovery contract 的一部分但不参与过滤，`INDNO` 也未使用。代码不会用 daily 中“某天恰有记录”替代 index membership。
- **market（7 列）**：只有 `DlyCalDt/vwretd` 被 risk builder 读取并 reindex 到共同日期轴；`vwretx/ewretd/ewretx/sprtrn/spindx` 只用于识别该数据集，当前 beta/residual volatility 不使用它们。
- **ccm（9 列）**：`gvkey` 连接 Compustat，`lpermno` 映射 CRSP；`USEDFLAG/linktype/linkprim/linkdt/linkenddt` 决定 point-in-time link；`liid/lpermco` 不参与当前排序或去重。
- **fundamentals（33 列）**：`indfmt/datafmt/popsrc/consol/curcd` 做标准报表过滤；`datadate` 加六个月得到可用日；`seq/ceq/pstk/pstkrv/pstkl/txditc` 形成 book equity，`revt/cogs/xsga/xint` 形成 profitability，`at` 的同比形成 investment，`dltt/dlc/at` 形成 leverage，`sich` 是 SIC fallback。`act/sale/ni/ib/fyear/fyr` 等交付列当前未进入 exposure。
- **delistings（19 列）**：生产代码只读 `PERMNO/DelDlyDt/DelRet`，用交易日精确等值 join；`DelistingDt` 和其他价格、原因、付款字段不参与补值。因而 `DelDlyDt` 缺失或与 daily date 不同不会做邻近日期匹配。

完整列名和行数被写进 [`qa_report.json`](/data/sunyuxiang/rl_alpha/processed/panel/qa_report.json)；字段的最小 discovery contract 以 [`contracts.py`](../src/rlalpha/data/contracts.py#L11-L26) 为准。这里特别要区分“Parquet 有该列”“discovery 要求该列”和“生产计算实际读取该列”三个层次。

**潜在 schema contract 缺口**：daily contract 没有要求 risk builder 会读取的 `SICCD`；fundamentals contract 也没有要求 `pstkrv/pstkl/pstk/txditc/xsga/xint/sich`，但 `compute_accounting_exposures()`/risk builder 会直接索引这些列。当前六份数据都提供了它们，所以构建成功；换一份只满足最小 contract 的 Parquet 可能通过 discovery/data QA，却在 `risk build` 时报 `KeyError`。这是 contract 声明弱于真实消费者的实现风险。

### 2.2 QA 实际检查了什么

[`validate_raw_bundle()`](../src/rlalpha/data/validate.py#L15-L56) 完成以下统计：

1. `PERMNO × DlyCalDt` 重复数；
2. 使用正 `DlyCumFacPr` 后，调整 high/low 是否包住 open/close；
3. 2010-06-30、2018-06-29、2021-06-30、2025-06-30 的历史 member 数；
4. `DlyCap / (abs(DlyClose) * ShrOut)` 的有限正值中位数；
5. close、return、volume 非空覆盖率；
6. 所有源文件路径、行数、列名和内容 fingerprint。

当前结果：

```text
duplicate PERMNO-date             0
membership counts                500 / 505 / 505 / 503
cap identity ratio median        1.0
DlyClose coverage                0.9983948713
DlyRet coverage                  0.9984264821
DlyVol coverage                  0.9983977450
adjusted high valid rate         0.9999987207
adjusted low valid rate          0.9999996802
```

**潜在风险**：当前 `ok` 的 hard failure 只有重复记录和抽查 member 数不在 450–550。OHLC valid rate、coverage 和 cap ratio 虽然被报告，但没有失败阈值；不能把 `ok: true` 解读为这些指标全部经过强制断言。

### 2.3 CRSP 调整与退市收益

[`apply_crsp_adjustments()`](../src/rlalpha/data/adjustments.py#L7-L22) 生成：

\[
\begin{aligned}
P^{adj}_{t,i} &= P^{raw}_{t,i}/DlyCumFacPr_{t,i},\\
V^{adj}_{t,i} &= DlyVol_{t,i}\times DlyCumFacShr_{t,i}.
\end{aligned}
\]

`DlyCumFacPr` 或 `DlyCumFacShr` 缺失、非有限或不大于 0 时，对应调整结果是 NaN，而不是回退到 raw value。当前分别有 259 行无效 price/share factor。

`DlyRet` 被视为 CRSP CIZ 已经给出的 authoritative total return。退市处理 [`fill_missing_delisting_returns()`](../src/rlalpha/data/adjustments.py#L25-L40) 只在以下三个条件同时满足时替换：

```text
DlyDelFlg == "Y"
and DlyRet is missing
and (PERMNO, DlyCalDt) 能匹配有限 DelRet
```

已有有限 `DlyRet` 永远不与 `DelRet` 再复合。实际数据有 2 条 eligible missing，但没有任何一条匹配到有限 `DelRet`，所以 `filled=0, unresolved=2`。

### 2.4 历史 membership

[`membership.py`](../src/rlalpha/data/membership.py#L5-L24) 接受：

```python
ACTIVE_MEMBERSHIP_FLAGS = {"Y", "NORM", "1", "TRUE"}
```

区间判定是：

\[
MbrStartDt_i \le t \le MbrEndDt_i.
\]

两端都包含。dense membership 的构造遍历每条 interval，把该 PERMNO 对应日期位置设为 True，见 [`panel.py`](../src/rlalpha/data/panel.py#L56-L66)。membership 本身不要求股票当日有行情，因此 member=True 的位置仍可能缺 close/volume；后续 `common_mask` 会进一步过滤。

### 2.5 标签时间轴与 split 边界

信号假设在收盘 (t) 后形成；下一交易日收盘 (t+1) 执行；从 (t+2) 开始持有，至 (t+21) 收盘结束。20 日 label 为：

\[
y^{20}_{t,i}
=\prod_{k=2}^{21}(1+r_{t+k,i})-1.
\]

生产路径在 [`_write_dense_artifacts()`](../src/rlalpha/data/panel.py#L39-L55) 中使用共同交易日轴：

```python
for offset in range(2, 22):
    values = returns[offset : offset + T - 21]
    complete &= isfinite(values)
    compounded *= where(isfinite(values), 1 + values, 1)
```

因此：

- `offset=2,...,21`，恰好 20 个 return；
- exit date 对应 `dates[index + 21]`；
- 任一个 return 缺失，label 为 NaN；
- label 只写入 `0:T-21`。

随后针对每个 split 检查 signal date 和 exit date：

```text
split.start <= signal_date <= split.end
exit_date <= split.end
```

不满足则清空 label。权威 split 当前硬编码于 [`data/splits.py`](../src/rlalpha/data/splits.py#L15-L19)：

| Split | Start | End |
|---|---|---|
| train | 2010-01-01 | 2018-12-31 |
| validation | 2019-01-01 | 2021-12-31 |
| test | 2022-01-01 | 2025-12-31 |

独立 DataFrame 参考实现 [`next_close_forward_return()`](../src/rlalpha/data/labels.py#L9-L29) 用每只证券自己的行序列循环，主要用于单元测试；生产 dense artifact 使用共同交易日轴的向量化版本。

### 2.6 Panel artifact、fingerprint 与加载

[`build_panel()`](../src/rlalpha/data/panel.py#L72-L107) 的顺序是：

```text
validate raw bundle
  → discover six files
  → fingerprint all source files
  → reuse compatible existing artifact, or
  → read daily/delistings
  → adjust OHLCV and fill eligible delisting return
  → write year-partitioned daily Parquet
  → build dense feature/return/membership arrays
  → write QA and build manifest atomically
```

落盘结构：

```text
/data/sunyuxiang/rl_alpha/processed/panel/
├── daily/year=YYYY/*.parquet
├── features.zarr/
│   ├── open
│   ├── high
│   ├── low
│   ├── close
│   ├── volume
│   └── return
├── returns.zarr/
│   ├── daily_total_return
│   └── forward_return_20d
├── membership.zarr/membership
├── risk_exposures.zarr/exposures
├── index.json
├── build_manifest.yaml
├── risk_build_manifest.yaml
├── qa_report.json
└── risk_diagnostics.json
```

当前 artifact：

```text
feature/return/membership shape    [4529, 891]
risk exposure shape               [4529, 891, 22]
feature/return dtype               float32
membership dtype                  bool
feature chunks                    [252, min(256, 891)]
risk chunks                       [21, min(256, 891), 22]
finite split-safe labels          2,715,491
median daily members              503
panel build fingerprint           dd68112d...e5c730a3d
risk build fingerprint            07499b36...677efe5
```

reuse 的 fast path 同时要求 source fingerprint、artifact version 和所有 dense path 存在。panel 与 risk 的 `source_fingerprint` 都保守地 hash 六份 raw 文件，因此修改一个只与 risk 有关的 fundamentals 文件也会使 panel fast path 失效。写 JSON/YAML/NPY 使用 [`utils/io.py`](../src/rlalpha/utils/io.py#L13-L50) 的临时文件、`fsync`、`os.replace` 与 `FileLock`，避免半写 checkpoint。

**重要缓存失效边界**：fast path 失效后，`build_panel()` 若发现 `panel/daily/` 已存在，会直接读这个旧分区作为 `adjusted`，不会重新读取最新 raw daily/delistings；随后却用最新六文件 fingerprint 写新 manifest。也就是说，raw daily 或 delistings 原地更新时，单纯重跑 `data build` 可能把旧 adjusted/dense 数据标记为匹配新 fingerprint。当前快照是已成功构建的既有产物，但今后数据版本更新应先修复该 invalidation 路径，或在明确备份/重建流程下移走旧 `panel/daily`，不能把 fast-path miss 等同于全量重建。[分支代码](../src/rlalpha/data/panel.py#L82-L103)

[`PanelStore.load_split()`](../src/rlalpha/data/store.py#L71-L101) 会：

1. 验证请求日期不越出 split；
2. 定位 target date indices；
3. 在 target start 前附加最多 `history=252` 行；
4. 直接按 Zarr array path 切片，避免遍历 group；
5. 从 risk Zarr attrs 读取 exposure column names；
6. 返回 frozen `SplitPanel`。

`SplitPanel.evaluate(node)` 先在含 history 的 features 上执行 DSL，再用 `target_slice` 返回 split 内结果。这样 rolling lookback 有历史上下文，但评分不会把历史行当成目标样本。

### 2.7 Balanced-22 的精确构造

Balanced-22 列为：

```text
intercept                                              1
industry_{NoDur,Durbl,Manuf,Enrgy,Chems,BusEq,        11
          Telcm,Utils,Shops,Hlth,Money}
style_{size,beta_252,momentum_12_1,reversal_1m,        10
       resid_vol_252,amihud_20,book_to_market,
       operating_profitability,investment,leverage}
                                                     ----
                                                       22
```

FF12 映射在 [`risk/ff12.py`](../src/rlalpha/risk/ff12.py#L5-L32)。`Other` 是省略基准组，因此只有前 11 个行业 dummy；配合 intercept 可避免完整 12 dummy 的确定共线性。

#### 六个市场 style

生产实现 [`_rolling_market_styles()`](../src/rlalpha/risk/builder.py#L27-L45) 对每个 asset 沿时间计算：

\[
size_{t,i}=\log(DlyCap_{t,i}).
\]

\[
beta_{t,i}=\frac{Cov_{252}(r_i,r_m)}{Var_{252}(r_m)},
\]

其中 rolling 至少需要 126 个 stock observation；market variance 也在 stock 有值的日期上计算。

\[
resid\_vol_{t,i}=Std_{252}(r_{t,i}-beta_{t,i}r_{m,t}).
\]

\[
reversal\_1m_{t,i}=\exp\left(\sum_{j=0}^{20}\log(1+r_{t-j,i})\right)-1.
\]

\[
momentum\_{12\_1,t,i}
=\exp\left(\sum_{j=21}^{252}\log(1+r_{t-j,i})\right)-1.
\]

\[
amihud\_{20,t,i}
=\log\left(Mean_{20}\frac{|r_{t,i}|}{|close_{t,i}\cdot volume_{t,i}|+10^{-12}}\right),
\]

Amihud 至少需要 15 个 observation。

三个容易被公式掩盖的实现细节：risk builder 直接读取 raw `DlyClose/DlyVol`，不是 panel 的 `adj_close/adj_volume`；`log1p` 只保留 `return > -1`，所以 `-100%` 及更小值在 reversal/momentum window 中视为缺失；`resid_vol_252` 先用每一天各自滚动估出的 beta 形成一条 residual series，再对这条 series 滚动求 std，并不是在每个 t 用同一个 252 日回归残差一次性求标准差。[实现](../src/rlalpha/risk/builder.py#L27-L45)

#### 四个会计 style

[`filter_standard_fundamentals()`](../src/rlalpha/data/fundamentals.py#L7-L18) 只保留：

```text
indfmt=INDL, datafmt=STD, popsrc=D,
consol=C, curcd=USD
```

并定义：

```python
available_date = datadate + 6 calendar months
```

book equity 的 preferred stock 依次选择 `pstkrv`、`pstkl`、`pstk`，缺失视为 0：

\[
BE=shareholders\ equity+txditc-preferred.
\]

`seq` 缺失时，shareholders equity 回退到 `ceq + pstk`。其余 exposure：

\[
OP=\frac{revt-cogs-xsga-xint}{BE},
\]

其中 `xsga/xint` 缺失按 0；只有正 book equity 且必要字段存在时计算。

\[
investment=\frac{at_t}{at_{t-1}}-1,
\qquad
leverage=\frac{dltt+dlc}{at}.
\]

[`select_ccm_links()`](../src/rlalpha/data/fundamentals.py#L21-L38) 的 link 规则为：

1. `USEDFLAG == 1`；
2. `linktype ∈ {LC, LU, LS}`；
3. `linkprim ∈ {P, C}`；
4. `linkdt <= datadate <= linkenddt`，空 end 表示开放；
5. 同一 `lpermno, datadate` 下 `P` 优先于 `C`；
6. 同优先级选 `linkdt` 更晚者。

[`_accounting_arrays()`](../src/rlalpha/risk/builder.py#L48-L70) 对每个 PERMNO 用 `searchsorted(..., side="right") - 1` 做 backward-as-of。记录只在：

```text
available_date <= trading_date <= available_date + 18 months
```

时有效。book-to-market 做单位换算：

\[
book\_to\_market=BE\times1000/DlyCap,
\]

因为 Compustat 会计字段按 million USD、CRSP `DlyCap` 按 thousand USD 解释。SIC 优先使用 CRSP `SICCD`，缺失时回退 Compustat `sich`。

#### 每日横截面预处理

[`preprocess_exposures()`](../src/rlalpha/risk/exposures.py#L20-L50) 只在当日 member 子集内处理每个 style：

1. 有限值太少时整列 disabled 并设 0；门槛是 `max(3, min(30, n//4))`；
2. 对有限值做 1%/99% winsorize；
3. 缺失用 winsorized median 填充；
4. population standard deviation `ddof=0`；
5. 近常数时 disabled，否则 z-score；
6. 拼接 intercept、11 industry dummy、10 style。

非 member 的 exposure 保持 NaN。当前整个 `[4529,891,22]` 矩阵有限率约 0.564；这主要反映全历史 891 只证券中，每天只有约 500 只是 member，而不是风险模型只有 56% 可用。

由于 member 内每个 style 的原始缺失都会被 median 填充、整列不足/常数时会被 0 填充，正常构建后 member 行的 22 列是完整的；行业无法映射时落入省略的 `Other` 基准（11 个 dummy 全 0），也不会产生 NaN。因而 `common_mask` 保留 exposure-complete 条件是一条防御性 contract，而不是允许会计字段缺失直接逐股删样本。

### 2.8 `common_mask` 的实验含义

[`SplitPanel.common_mask`](../src/rlalpha/data/store.py#L36-L40) 是：

```python
membership
& isfinite(close) & (close > 0)
& isfinite(volume) & (volume > 0)
& isfinite(exposures).all(axis=2)
```

训练 objective 再与有限 label 相交，见 [`objective_for()`](../src/rlalpha/search/run.py#L26-L36)。候选 validity 使用的是 `target(common_mask)`，尚未额外要求 label；reward mask 才要求 label。

**设计含义**：R0 虽不做 neutralization，也只在 Balanced-22 完整的 universe 上评分。因此 R0/R1/R2 的主要样本 universe 可比，差别集中在 target/signal 是否残差化以及是否加入 HAC 稳定性惩罚。

---

## 3. DSL、AST 与因子执行

### 3.1 表达空间

[`dsl/operators.py`](../src/rlalpha/dsl/operators.py#L3-L12) 定义唯一合法集合：

| Family | Operators | Arity |
|---|---|---:|
| Feature | `$open`, `$high`, `$low`, `$close`, `$volume`, `$return` | 0 |
| Unary | `Abs`, `Sign`, `Log` | 1 |
| Binary | `Add`, `Sub`, `Mul`, `Div`, `Greater`, `Less` | 2 |
| Rolling | `Ref`, `Mean`, `Sum`, `Std`, `Var`, `Max`, `Min`, `Med`, `Mad`, `Delta`, `WMA`, `EMA`, `TSRank` | signal + window |
| Pair rolling | `Cov`, `Corr` | 2 signals + window |
| Cross-sectional | `CSRank`, `CSZScore` | 1 |

窗口固定为：

```text
1, 5, 10, 20, 40, 60, 120, 252
```

常数固定为：

```text
-2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2
```

没有 VWAP、fundamental feature、自由浮点常数、任意窗口或 future reference。

### 3.2 Node 类型、canonical form 与 hash

[`dsl/ast.py`](../src/rlalpha/dsl/ast.py#L14-L138) 有四个 concrete node：

- `Feature(name)`：必须属于固定 feature 集；
- `Constant(value)`：必须命中固定常数；
- `Window(value)`：只允许固定窗口，且不计入 depth/nodes；
- `Call(operator, args)`：校验 operator、arity、window 位置，并要求该 Call 子树至少含一个 feature。

这个约束有一个边界：root 可以直接是 `Constant(1)`，因为它没有经过 `Call.__post_init__`；它是 AST-valid，但会在 signal validity 阶段因 near-constant 失败。相反，`Add(1,2)` 在构造 `Call` 时就因整棵 Call 子树无 feature 而失败。

`Call.depth` 是 `1 + max(child.depth)`；`Call.nodes` 是 call 自身加所有非 window child node。lookback 递归累加：

```text
Ref/Delta(child, w)       child.lookback + w
其他 rolling(child, w)    child.lookback + w - 1
非 rolling                max child.lookback
```

因此：

```text
Mean(Ref($close,20),120)
```

lookback 为 `20 + 119 = 139`，不是 140。

限制统一由 `validate_limits()` 强制：

```text
depth <= 6, nodes <= 21, lookback <= 252
```

`Add`、`Mul` 是 commutative，canonicalization 会按渲染后的参数字符串排序；其他 operator 保留参数顺序。`expr_hash` 是 canonical string 的 SHA-256。因此语义上交换等价的 `Add(a,b)`/`Add(b,a)` 会 exact deduplicate，而 `Sub(a,b)`/`Sub(b,a)` 不会。

### 3.3 Parser 与 LLM response contract

[`parse_expression()`](../src/rlalpha/dsl/parser.py#L52-L59) 使用自定义 tokenizer 和递归下降 parser，不调用 Python `eval`。它会拒绝：

- 未知 token/operator；
- arity 不符；
- trailing tokens；
- window 放在非 rolling 位置；
- 常数或 window 不在固定集合；
- 越过 AST limits。

[`parse_llm_response()`](../src/rlalpha/dsl/parser.py#L62-L66) 进一步要求整个 completion 完整匹配：

```xml
<expr>...</expr>
```

标签前后出现说明文字也会失败。正则使用 `IGNORECASE`，所以手工输入的 `<EXPR>...</EXPR>` 也可通过；feature token 会先 `.lower()`，operator 名仍区分大小写。XGrammar 生成侧则只允许 grammar 中的小写 tag/feature 和精确 operator spelling。

### 3.4 全部 operator 的真实数值语义

执行器入口是 [`evaluate()`](../src/rlalpha/dsl/evaluator.py#L115-L174)。所有 feature 必须共享 `[time, asset]` shape；Constant 广播成同 shape。

#### Pointwise 与 protected operator

```text
Abs(x)       abs(x)
Sign(x)      sign(x)
Log(x)       log(abs(x) + 1e-6)
Add/Sub/Mul  NumPy pointwise arithmetic
Greater      float(x > y), 即 0.0/1.0
Less         float(x < y), 即 0.0/1.0
```

`Greater/Less` 是一个需要特别记住的 NaN 例外：NumPy 中与 NaN 的大小比较结果为 `False`，再转 float 后是 `0.0`，所以它们不会像算术 operator 一样传播 NaN。后续 validity/reward 的 mask 仍会排除不可交易位置，但嵌套表达式可把上游 NaN 变成 0。

protected division 为：

\[
Div(a,b)=\frac{a}{sign^*(b)\max(|b|,10^{-6})},
\]

其中 `sign*(b)` 对负数是 -1，否则是 +1；零分母按正 `1e-6` 处理。

#### 横截面 operator

`CSRank` 对每一行、跨 asset 做 pandas percentile rank；`CSZScore` 对每一行按有限值计算 `ddof=0` 的 mean/std，零 std 变 NaN。它们本身不知道 membership，非 member 是否参与取决于输入位置是否已经是 NaN；基础 feature 在 panel 中可能对非 member 仍有值，因此表达式执行阶段不自动 mask，mask 在 validity/reward 阶段施加。

#### 时间 rolling operator

除 `Ref/Delta` 外，aggregation rolling 的 minimum observations 是：

\[
min\_periods=\max(1,\lceil0.8w\rceil).
\]

Pair rolling 至少要求 `max(2, ceil(0.8w))` 个共同有限 observation。

| Operator | 实现要点 |
|---|---|
| `Ref(x,w)` | pandas `shift(w)`；只访问过去 |
| `Delta(x,w)` | `x_t - x_{t-w}` |
| `Mean/Sum` | 自定义 cumulative sum/count |
| `Std/Var` | sample variance，分母 `count-1`；近零结果设 NaN |
| `Max/Min/Med` | bottleneck move kernels |
| `Mad` | Numba 并行 rolling median absolute deviation |
| `TSRank` | bottleneck normalized rank 转换为约 `(0,1]` 的窗口内 rank |
| `WMA` | `lfilter` 系数为 `w,w-1,...,1`；按 FIR 语义，当前值乘 `w`、`t-1` 乘 `w-1`，所以越新的 observation 权重越大 |
| `EMA` | pandas `ewm(span=w, adjust=False)` |
| `Cov` | 共同有限位置上的 sample covariance |
| `Corr` | 共同有限位置上的 Pearson correlation |

`WMA` 与其他 unary rolling 一样只要求至少 `ceil(0.8w)` 个有限值；缺失位置在分子按 0 处理，分母只累计对应权重，所以它不是“必须完整窗口”的 WMA。[kernel](../src/rlalpha/dsl/evaluator.py#L70-L79)

### 3.5 Worked example：从文本到 signal

考虑：

```text
CSZScore(Div(Delta($close,20),Std($return,20)))
```

解析树：

```text
Call CSZScore
└── Call Div
    ├── Call Delta
    │   ├── Feature $close
    │   └── Window 20
    └── Call Std
        ├── Feature $return
        └── Window 20
```

属性：

```text
depth       4
nodes       6              # Window 不计 nodes
lookback    max(20, 19)=20
```

执行顺序：

1. `$close/$return` 从 `SplitPanel.features` 复制为 float array；
2. `Delta` 令前 20 行为 NaN，其余为 close 与 20 日前 close 的差；
3. `Std` 对 return 做 20 日 sample std，至少 16 个有限值，近零 std 设 NaN；
4. `Div` 对有限位置做 protected division；
5. `CSZScore` 每天跨全部 finite asset 标准化；
6. `SplitPanel.evaluate()` 取 `target_slice`；
7. validity/reward 才把输出与 common membership mask 相交。

[`SignalCache`](../src/rlalpha/factors/cache.py#L11-L42) 会以每个 `Call` 子树的 `expr_hash` 缓存结果。因此多个表达式共享 `Std($return,20)` 时，第二次执行可以命中内存 LRU；被 pool、pending 或 GP population 保留的完整 signal 还会以 float32 `.npy` 持久化。

### 3.6 四层“合法”不是同一个概念

| 层 | 检查者 | 通过意味着什么 | 典型失败 |
|---|---|---|---|
| XGrammar-valid | vLLM/xgrammar | token 串匹配 CFG | 非法标签、拼错 operator |
| AST-valid | `parse_llm_response` | 类型、arity、limits、固定常数/窗口正确 | 深度过大、constant-only `Call`；root constant 是例外 |
| Signal-valid | `validate_signal` | 真实 train panel 上覆盖、变异性、去相关通过 | constant correlation、coverage 低 |
| Unique market-evaluated | coordinator | 非 exact duplicate，且完成 signal validity | seen hash、budget exhausted |

XGrammar 不编码 depth/nodes/lookback，也允许语法上的 constant 子表达式，所以后三级仍不可省略。

### 3.7 Signal validity

[`validate_signal()`](../src/rlalpha/dsl/validity.py#L20-L46) 接收 signal、coordinator 传入的 common mask、当前 pool signals：

\[
coverage=\frac{\#(finite(signal)\land mask)}{\#mask}\ge0.80.
\]

每天至少 100 个 finite asset 才是 valid day，总 valid days 至少 252。对每个 valid day，按当日有限 signal 计算 population variance；variance 大于 `1e-12` 才是 variable day，variable-day rate 至少 80%。

对当前 pool 的每个 signal，计算 daily Pearson correlation，再在候选 valid days 上取平均和绝对值；最大值不得超过 0.95。注意这里判定的是：

```text
max_j abs(mean_t(daily_corr(candidate, pool_j)))
```

而不是 `mean_t(abs(daily_corr))`，也不是把整个 panel flatten 后计算一次相关。

失败优先级是 coverage → valid days → near constant → near duplicate；只报告第一个失败原因。

---

## 4. Reward、组合权重与 Pool

### 4.1 每日横截面标准化

[`FactorCalculator.standardize()`](../src/rlalpha/factors/calculator.py#L25-L39) 对每天单独执行：

1. mask 外或 signal 非有限位置设 NaN；
2. 取当日有限值 1%/99% quantile；
3. clip；
4. 用 `ddof=0` 的 mean/std z-score；
5. std 不大于 `1e-12` 的整日结果设 NaN。

label 不在此函数中 winsorize/z-score。Pearson correlation 对 label 的平移缩放不敏感，但 outlier 仍会影响 correlation。

[`daily_corr()`](../src/rlalpha/factors/calculator.py#L77-L90) 用共同有限且 mask 的位置，以充分统计量计算 Pearson correlation；样本少于 3 或任一方方差不大于 `1e-24` 时返回 NaN。

### 4.2 Ridge pool weights

设 pool 有 (M) 个每日标准化信号 (s_1,...,s_M)。定义平均 daily factor correlation：

\[
C_{jk}=mean_t\ Corr(s_{j,t},s_{k,t}),
\]

以及预测向量：

\[
\mu_j=mean_t\ Corr(s_{j,t},y_t).
\]

权重为：

\[
w=(C+\lambda I)^{-1}\mu,
\qquad \lambda=10^{-3}.
\]

组合信号：

\[
z_{t,i}=\sum_{j=1}^{M}w_j s_{j,t,i}.
\]

权重不要求非负、不归一化，也不直接限制 turnover。一个负 IC 因子可通过负权重产生正贡献。

独立显式实现是 [`RidgeCombiner`](../src/rlalpha/factors/combiner.py#L9-L32)；训练 reward 使用 [`RewardObjective._daily_ic()`](../src/rlalpha/rewards/base.py#L117-L159) 的矩缓存优化版本。

### 4.3 Fixed-universe moment 实现

`RewardObjective._set_moment_label()` 定义：

```python
fixed_common = mask & isfinite(label)
fixed_count  = fixed_common.sum(axis=1)
```

每个 signal 在固定 universe 内的缺失位置按 0 填入 moment：

```python
filled = where(fixed_common & isfinite(signal), signal, 0)
sum_x, square_x, cross_xy
```

然后通过逐日 sums、squares、cross products 重构 correlation。所有假设 pool 的 score 可复用：

- signal standardization cache；
- residual signal/label cache；
- 每个 signal 的 moments；
- 任意两个 signal 的 cross moments。

**设计含义**：训练 pool objective 的每日样本计数由 mask+label 固定；某因子偶发缺失时在组合 moment 中是零贡献，而不是为每对因子改变样本 universe。这也是 unit test 所称的 `fixed_universe_combination`。

### 4.4 R0：raw mean IC

[`R0Objective.score_pool()`](../src/rlalpha/rewards/r0.py#L9-L15)：

\[
J_{R0}(P)=mean_t\ Corr(z_t,y_t).
\]

空 pool score 是 0。非空但没有有限 daily IC 时 objective 是 `-inf`。

**当前实现**：R0 不残差化 signal/label，但它的 mask 来自 `common_mask & finite(label)`，而 `common_mask` 要求 22 个 exposure 全部有限。R0 应理解为“在共同风险可建模 universe 上的 raw IC”，不是完全不依赖风险数据的 baseline。

### 4.5 R1：mean risk-neutral IC

[`RewardObjective._neutralized_inputs()`](../src/rlalpha/rewards/base.py#L49-L82) 的精确顺序：

1. 原始 candidate signal 每日 winsorize/z-score；
2. 每天把标准化 signal 对 exposure (X_t\in\mathbb{R}^{N_t\times22}) 残差化；
3. 原始 label 每天独立对同一 exposure 残差化并缓存；
4. `_daily_ic()` 对残差信号再次执行每日 winsorize/z-score；
5. 用 residual label 计算 ridge predictive vector 和 pool daily correlation。

风险回归：

\[
s_t=X_t\beta^s_t+\epsilon^s_t,
\qquad
y_t=X_t\beta^y_t+\epsilon^y_t.
\]

最终：

\[
J_{R1}(P)=mean_t\ Corr\left(\sum_jw_j\tilde\epsilon^s_{j,t},\epsilon^y_t\right).
\]

其中 \(\tilde\epsilon^s\) 表示再次标准化后的 signal residual。实现位于 [`R1Objective`](../src/rlalpha/rewards/r1.py#L9-L16)。

[`RiskNeutralizer`](../src/rlalpha/risk/neutralize.py#L18-L69) 默认先做 reduced QR。若 `rank < K` 或 `cond(R)>10^{12}`，回退到：

\[
\hat\beta=(X^TX+10^{-10}I)^{-1}X^Ty.
\]

当日共同样本数必须大于 exposure 列数。诊断记录 observation 数、列数、rank、condition、ridge 和：

\[
\max_k\left|X_k^T\epsilon/N\right|.
\]

### 4.6 R2_LCB：HAC lower confidence bound

[`R2LCBObjective`](../src/rlalpha/rewards/r2_lcb.py#L10-L22) 复用 R1 的 daily RNIC，再计算：

\[
J_{R2}(P)=\overline{RNIC}-1.645\cdot SE_{NW,20}.
\]

[`newey_west_mean_se()`](../src/rlalpha/rewards/statistics.py#L8-L21) 使用 Bartlett weights：

\[
\hat\Omega
=\hat\gamma_0
+2\sum_{\ell=1}^{20}
\left(1-\frac{\ell}{21}\right)\hat\gamma_\ell,
\qquad
SE=\sqrt{\max(0,\hat\Omega)/T}.
\]

有限 observation 少于 2 时 SE 为 NaN；求和 offset 会截到 `T-1`，但 Bartlett 分母仍使用原请求 `lag+1=21`，不会按截短后的 lag 重标权重。1.645 是单侧 95% critical value。这里惩罚的是 mean estimator 的 HAC 标准误，不是 daily RNIC 的原始标准差。

R0/R1 在没有任何有限 daily IC 时返回 `objective=-inf`；R2_LCB 则通过 `lcb_score()` 返回 NaN，只有一个有限日时也因 SE=NaN 而得到 NaN。`PoolManager` 没有额外的 finite-objective guard，且 `max(-1,min(1,100*NaN))` 在 Python 当前参数顺序下会得到 `+1`。真实 panel 的 validity/day count 通常让该分支不可达，但这是 synthetic/异常数据下需要测试或修复的数值边界。

### 4.7 Candidate delta 与精确替换

[`PoolManager.score_candidates()`](../src/rlalpha/factors/pool.py#L34-L50) 先对当前 entries 计算 baseline (J(P))。对每个候选 (c)：

未满容量时：

\[
\Delta(c)=J(P\cup\{c\})-J(P).
\]

pool 已有 20 个因子时，枚举全部槽位：

\[
\Delta(c)=
\max_{j=1,...,20}J(P\setminus\{p_j\}\cup\{c\})-J(P).
\]

每个假设 pool 都重新求 ridge weights；不是按现有因子的单体贡献近似删除。候选已在 pool 时 delta=0、shaped reward=-0.5。

有效候选的 shaped reward：

\[
r(c)=clip(100\Delta(c),-1,1).
\]

`consider_group()` 对整个 frozen group 的 precomputed scores 取最大 delta，只在 `delta > min_delta=1e-5` 时 admission。未满则 append；已满则替换预计算的最佳槽位。一次调用最多让 pool version 增加 1。

### 4.8 Worked example：一组候选如何 admission

假设当前：

```text
capacity=2
P=[A,B]
J(P)=0.0300
min_delta=0.00001
```

本组有 C、D。完整重算得到：

```text
replace A by C → 0.0340
replace B by C → 0.0310   => C best score 0.0340, delta 0.0040

replace A by D → 0.0325
replace B by D → 0.0350   => D best score 0.0350, delta 0.0050
```

两者 reward 分别为 0.4、0.5，但 group 只 admission D，并替换 B：

```text
P'=[A,D]
pool_version += 1
```

不会先加入 C 再用已变化的 pool 计算 D；这就是 frozen-group contract。

### 4.9 Coordinator 的完整 reward 分支

[`SearchCoordinator.run_group()`](../src/rlalpha/search/coordinator.py#L43-L107) 给 outcome 的规则：

| 分支 | `valid` | `market_evaluated` | budget | shaped reward |
|---|---:|---:|---:|---:|
| `node is None` | false | false | 不消耗 | -1.0 |
| seen/pool exact duplicate | false | false | 不消耗 | -0.5 |
| 当前组内 budget 已耗尽 | false | false | 不消耗 | 0.0 |
| evaluator exception | false | false | 不消耗 | 当前默认 0.0 |
| near-duplicate signal | false | false | 不消耗 | -0.5 |
| 其他 validity failure | false | false | 不消耗 | -0.75 |
| valid unique signal | true | true | +1 | `clip(100*delta,-1,1)` |
| 已见的 `gp_rescore` | true | true | 不消耗 | 同正常 delta；未见 hash 仍按首次评估消耗 |

**潜在风险**：signal evaluation exception 没有显式 negative penalty，沿用 dataclass 默认 shaped reward 0；这与 parse/effective validity failure 的惩罚不一致。`pool.score_candidates()` 位于逐候选 `try/except` 之外，reward/ridge 求解本身若抛异常会终止整个 cell，而不是生成一条 `evaluation_error` outcome。

---

## 5. 四种搜索方法

四种方法都实现同一个 [`Searcher` Protocol](../src/rlalpha/search/base.py#L8-L15)：

```python
propose(context, n) -> list[Candidate]
observe(outcomes) -> None
state_dict() -> dict
load_state_dict(state) -> None
```

因此 proposer 只决定“提出什么”和“如何吸收反馈”；解析、市场计算、validity、budget、reward、pool admission、validation snapshot 都在 proposer 外部统一完成。

[`Candidate`](../src/rlalpha/search/models.py#L10-L24) 可以包含合法 `Node`，也可以用 `node=None, raw_text=...` 表示 LLM 无法解析的 completion。合法候选的 identity 是 AST hash；非法候选的 identity 是 `sha256("invalid:" + raw_text)`，便于记录不同失败文本。

### 5.1 Random

[`RandomSearcher`](../src/rlalpha/search/random_search.py#L12-L31) 的 `propose()` 不读 `SearchContext`，只是调用 [`sample_ast()`](../src/rlalpha/dsl/grammar.py#L9-L31)。递归生成规则：

1. depth 到 1，或每层以 22% 概率提前终止；
2. 必须包含 feature 的位置生成随机 feature；允许非必须 feature 的叶子以 30% 概率生成 constant；
3. 非叶子在 unary、binary、rolling、pair、cross 五个 family 中等概率选择；
4. binary 的第二个 child 可以直接是 constant leaf；若继续长成 `Call`，该 Call 自身仍必须包含 feature；
5. 最多尝试 100 次，接受 nodes≤21 且 lookback≤252；
6. 反复失败时回退 `$close`。

Random 的 `observe()` 只累计 outcome 数，不改变提议分布。其可复现性来自：

- `random.Random(seed)`，不依赖全局 NumPy RNG；
- checkpoint 将 `rng.getstate()` pickle 后 base64；
- resume 恢复同一 state，因此后续表达式序列完全一致。

**设计含义**：Random 仍然拥有动态 pool，因为统一 coordinator 会 admission；但 proposal distribution 不因 pool、reward 或历史 outcome 改变。三个 reward 下相同 seed 的初始 proposal 序列原则上相同，后续 budget 被 duplicate/validity 分支消耗的节奏可能因当前 pool 而分叉。

### 5.2 Typed-AST GP

[`GPSearcher`](../src/rlalpha/search/gp.py#L64-L159) 初始化 128 个随机 typed AST；每个 `_Individual` 保存 `node` 和相对当前 pool 的 `fitness`。

#### Selection 与 genetic operators

`_tournament()` 从 population 无放回抽最多 5 个，选 fitness 最大者。`_offspring()` 的 draw 区间当前硬编码为：

```text
[0.00,0.50) subtree crossover
[0.50,0.75) subtree mutation
[0.75,0.90) point mutation
[0.90,1.00) reproduction
```

`_paths()` 遍历所有非 Window 节点，返回从 root 开始的 child index path。Window 不作为可独立交换的 subtree，否则可能把普通 signal child 替成 window。

- **Crossover**：从 parent 1 选一个 path，从 parent 2 选一个 subtree，调用 `_replace()`；
- **Subtree mutation**：把随机 path 换成最大 depth 3 的新随机 AST；
- **Point mutation**：feature/constant 换同类其他值；Call 只在相同 operator family 内换 operator，因此 arity 保持一致；
- **Reproduction**：直接返回第一 parent。

新树必须重新通过 `validate_limits()`；replacement 构造或 limit 失败时保留原 parent。最终再 canonical parse 一次，确保 AST 与 parser contract 一致。

#### Fitness 为什么会 stale

GP fitness 是 `candidate.delta_objective`，即相对某个 frozen pool 的边际价值，不是表达式固有分数。若 pool 从 (P_v) 变成 (P_{v+1})：

\[
\Delta(c\mid P_v)\ne\Delta(c\mid P_{v+1}).
\]

所以 [`propose()`](../src/rlalpha/search/gp.py#L116-L129) 发现 `context.pool_version` 变化时：

```python
for individual in population:
    individual.fitness = -inf
stale = every population hash
```

接下来优先每次返回至多 8 个 `generator="gp_rescore"` 的 population 个体。coordinator 只有在该 hash 已经属于 `seen` 时才把它认作免费 rescore：已见个体可绕过 duplicate 且不增加 valid-unique budget；初始 population 或补入但从未市场计算的新个体虽带 `gp_rescore` generator，仍按首次 valid unique 消耗预算。

理想情况下 stale list 清空后才继续繁殖；但 rescore group 自己也进入 pool admission。如果其中一个个体让 pool version 再变化，下一次 `propose()` 会再次把**全 population** 设为 stale，尚未完成的一轮重评分从头按当前 population 重建。这解释了 GP 的 raw proposals 可远大于 valid-unique budget；当前实现保证 fitness 相对于最近观察到的 pool 刷新，而不保证每次 pool version 后总能完整无中断地扫完 128 个个体。

#### Population 更新

[`observe()`](../src/rlalpha/search/gp.py#L131-L145)：

1. 只从 `market_evaluated` outcome 取 delta fitness；
2. 更新 population 中已有 hash；
3. 把有效且不在 population 的 pending offspring 加入；
4. 旧 population + additions 按 fitness 降序；
5. canonical hash 去重并截断到 population size；
6. 不足时补随机 AST，补入者 fitness 为 `-inf`；
7. 清空 pending。

checkpoint 保存全部表达式、fitness、pool version、stale list 和 RNG。`retained_hashes` 通知 coordinator 不要从内存 signal cache 清除仍在 population 中的 signal。

**配置审计摘要**：`population_size`、`tournament_size` 被使用；`elitism` 只保存在实例属性中，selection 和 checkpoint state 都不读取它；四类概率和 `offspring_per_pool_batch` 没有从 YAML 读取。

### 5.3 Base LLM

[`BaseLLMSearcher`](../src/rlalpha/search/base_llm.py#L53-L123) 使用固定本地 Qwen3.5-2B 和 vLLM，只生成、不训练。

#### 模型发现与加载

[`resolve_model_path()`](../src/rlalpha/search/base_llm.py#L36-L50) 优先使用 `config.model.path`；否则在 `RLALPHA_MODEL_SEARCH_ROOT` 或 `/data/shared/huggingface` 下寻找唯一候选 Qwen3.5-2B。自动发现分支要求 `config.json` 和至少一个 safetensors；显式 `model.path` 分支只预检 `config.json`，权重/tokenizer 缺失会延后到 Transformers/vLLM loader 才失败。

当前模型：

```text
repository  Qwen/Qwen3.5-2B
revision    15852e8c16360a2fea060d615a32b45270f8a8fc
path        /data/shared/huggingface/Qwen3.5-2B
dtype       bfloat16
```

[`configure_packaged_cuda_toolchain()`](../src/rlalpha/search/base_llm.py#L16-L33) 优先环境内 bundled CUDA，设置 `CUDA_HOME/PATH`，把 FlashInfer workspace 放在 `/tmp`，并关闭 vLLM FlashInfer sampler 以避免宿主 header/compiler 不匹配。

vLLM 参数：

```text
max_model_len             config.rollout.max_model_len，当前 4096
gpu_memory_utilization    RLALPHA_VLLM_MEMORY_UTILIZATION，默认 0.18
enforce_eager             true
trust_remote_code         true
```

#### Prompt 的完整信息边界

[`build_messages()`](../src/rlalpha/search/prompts.py#L25-L49) 返回 system/user 两条 message。system 只要求一个精确表达式块；user 包含：

- 当前 exploration hint；
- 六个 feature；
- operator family、固定 windows/constants；
- depth/nodes/lookback；
- 三个合法例子；
- 常见非法 token：`SMA`、`SIGMA`、`Ema`、`$window`、`$vol`、infix arithmetic 等；
- 当前 train pool version/objective；
- 当前 pool 每条 formula 和 ridge weight；
- “提出 unique complement”的指令。

五个 hint 循环：

```text
momentum → mean reversal → volatility
→ price-volume interaction → multi-scale structure
```

`SearchContext` 还包含 valid budget 和最近 64 条 outcome summary，但 prompt 当前没有渲染这两项。prompt 不包含 reward name、validation 或 test；三个 reward 的 prompt template 完全相同。

#### Structured sampling

每个 group 构造 8 个 prompt，hint 按 `raw_completions + index` 轮换。每个 prompt 有独立 RNG seed：

```text
temperature=1.0, top_p=1.0, top_k=20
presence_penalty=2.0, repetition_penalty=1.0
max_tokens=128, thinking=false
```

[`DSL_GRAMMAR`](../src/rlalpha/search/prompts.py#L11-L22) 作为 vLLM `StructuredOutputsParams(grammar=...)`。每条结果仍经 `parse_llm_response()`；token 数和 generate wall time 分别累计到 `total_tokens/gpu_seconds`。

`observe()` 是 no-op。模型不会吸收 reward；只有下一轮 prompt 中变化的 pool/objective 能提供间接反馈。

`total_tokens` 只累计 completion token ids，不含 prompt token；`gpu_seconds` 是 `generate()` 调用的 wall-clock 区间，不含首次模型加载。它们是内部一致的成本 proxy，不是 tokenizer billed tokens 或硬件 active-time 计量。

### 5.4 Staged GRPO LLM

当前主类是 [`VerlGRPOStageCoordinator`](../src/rlalpha/search/grpo/stage_coordinator.py)。它和 Base LLM 共享 Qwen path、prompt、grammar，并在一个常驻 Verl/Ray/vLLM 生命周期内完成全部 rollout 和 update。

#### 5.4.1 一次 proposal/update 的时序

```text
SearchContext(pool version v)
  │
  ├─ build one prompt + one exploration hint
  ├─ tokenize → input_ids [1, P]
  └─ HF model.generate, num_return_sequences=8
          │
          ├─ generated        [8, P+Rmax]
          ├─ response_ids     [8, Rmax]
          ├─ response_mask    [8, Rmax]
          └─ decode/parse → 8 Candidate
                  │
                  ▼
SearchCoordinator
  ├─ evaluate signal [T_train, N]
  ├─ validity / exact duplicate
  └─ frozen pool delta → shaped_reward [8]
                  │
                  ▼
observe(outcomes)
  ├─ scalar reward placed on last response token
  │      token_rewards [8, Rmax]
  ├─ Verl group-normalized outcome advantage
  │      advantages [8, Rmax]
  ├─ teacher-forced logits [8, Rmax, vocab]
  ├─ generated-token log_probs [8, Rmax]
  ├─ advantage-weighted sequence objective
  └─ LoRA backward / grad clip / AdamW step
```

#### 5.4.2 模型与 LoRA

`_load()` 使用：

```text
AutoProcessor
AutoModelForImageTextToText
dtype=bfloat16
attention=sdpa
gradient checkpointing=true
device=cuda
```

首次运行通过 PEFT 注入：

```text
r=16
lora_alpha=32
target_modules=all-linear
lora_dropout=0
bias=none
task_type=CAUSAL_LM
```

resume 时先加载 base model，再从 checkpoint `adapter/` 以 `is_trainable=True` 恢复。优化器是对所有 `requires_grad` 参数的 AdamW，learning rate (10^{-6})；scheduler 为恒等 `LambdaLR`。

#### 5.4.3 XGrammar 与生成停止

模型加载后，代码从 tokenizer 和模型 vocab 编译 `DSL_GRAMMAR`。每个 microbatch 新建 grammar logits processor，因为 matcher 带逐序列状态。

另有两个 logits processor：

- `PresencePenalty`：对 response 中已经出现的 token logit 减 2；
- `StopAfterGrammar`：matcher terminated 后把该行除 EOS 外的 logits 设为 `-inf`。

生成参数当前硬编码：

```text
do_sample=true
temperature=1.0
top_p=1.0
top_k=20
repetition_penalty=1.0
max_new_tokens=128
num_return_sequences=microbatch slice
```

总 rollout 永远是 8；`RLALPHA_GRPO_MICROBATCH` 只决定把 8 条分几次 generate/forward，不改变同 prompt 的逻辑 group。

GRPO 的 `total_tokens` 同样只数 non-pad response token；`gpu_seconds` 累加 generation 与训练 forward/backward/step 的 wall time，但不含 `_load()`。因此 Base/GRPO 的 token 字段口径相近，GPU seconds 覆盖的工作却不同，不能直接当作严格硬件利用率比较。

#### 5.4.4 Reward 到 token reward

coordinator 返回：

```python
rewards = np.asarray([outcome.shaped_reward for outcome in outcomes])  # [8]
```

[`compute_score()`](../src/rlalpha/search/grpo/verl_reward_function.py) 通过 Verl 自定义 reward hook 返回每个 completion 的 sequence reward，并附带可恢复的 domain diagnostics。

例外是 response mask 全 0 的空 completion：函数找不到可放置位置，会让该行 token reward 总和为 0，即使 coordinator outcome 是 parse failure/-1；该 rollout 对当前 update 没有负 reward gradient。正常 grammar generation 通常至少生成 tag/token，但 reward bridge 本身没有为零长度响应补哨兵位置。

#### 5.4.5 Verl advantage 的精确公式

当前环境使用 Verl commit `4a2cba76...`，`compute_grpo_outcome_advantage()` 接收：

```text
token_level_rewards  [8, Rmax]
response_mask        [8, Rmax]
index                zeros(8)，表示八条属于同一 prompt group
```

先对 token 维求和得到 (r_i)，再按 group 计算均值和 `torch.std`。`torch.std` 默认是 sample standard deviation：

\[
\bar r=\frac1{8}\sum_i r_i,
\qquad
s_r=\sqrt{\frac1{7}\sum_i(r_i-\bar r)^2}.
\]

\[
A_i=\frac{r_i-\bar r}{s_r+10^{-6}}.
\]

最后：

\[
A_{i,t}=A_i\cdot response\_mask_{i,t},
\]

也就是同一 rollout 的每个有效 response token 都获得相同 advantage。八个 reward 标准差不超过 `1e-12` 时，代码增加 `zero_group_variance`；此时 Verl 计算出的 numerator 也是零，梯度更新基本为零。

#### 5.4.6 当前 policy loss

teacher-forced forward 输入完整 `generated [8,P+Rmax]`。取：

```python
logits = output.logits[:, P-1:-1]
targets = generated[:, P:]
log_probs = log_softmax(logits).gather(targets)
```

每条 sequence objective：

\[
q_i=\frac{\sum_t A_{i,t}\log\pi_\theta(a_{i,t}|x_i,a_{i,<t})}
{\max(1,\sum_t mask_{i,t})}.
\]

整体 loss：

\[
L=-\frac1{8}\sum_{i=1}^{8}q_i.
\]

microbatch 时每个 chunk 仍除以完整 rollout group 8，所有 chunk backward 累积后统一：

```text
clip_grad_norm=1.0
optimizer.step()
scheduler.step()
```

#### 5.4.7 这是不是完整 GRPO/PPO

**当前实现**没有：

- old-policy probability ratio；
- PPO clipping；
- reference model；
- KL loss；
- entropy bonus；
- multi-epoch minibatch replay。

它确实使用了 GRPO 的“同 prompt 多 rollout、组内 outcome reward 去均值/除标准差” advantage，但优化器直接最大化当前策略下生成序列的 advantage-weighted log probability。准确描述应是：

> GRPO-style group-normalized outcome policy gradient with LoRA，而不是完整 Verl PPO/GRPO trainer。

YAML 的 `actor.use_kl_loss: true` 和 `kl_loss_coef: 0.001` 当前没有进入 loss。

#### 5.4.8 Frozen-pool stage

类常量：

```python
rollout_group = 8
admission_group_interval = 8
```

因此：

```text
1 proposal group = 1 prompt × 8 rollouts × 1 LoRA update
1 stage          = 8 proposal groups = 64 rollouts × 8 updates
1 stage admission <= 1 candidate
```

coordinator 在前 7 组只累积 `pending_entries/pending_scores`，第 8 组才调用一次 `PoolManager.consider_group()`。64 条候选的 scores 都是在同一 pool baseline 上预计算，虽然模型在八组之间持续更新。

第八组中的顺序是：

```text
propose against pool v
→ score against pool v
→ coordinator flush admission（pool 可能变成 v+1）
→ searcher.observe 第八组并更新 LoRA
→ 下一 propose 看到 context.pool_version 变化，groups_in_stage 清零
```

预算在 stage 中间耗尽时，`run_search()` 只对 admission interval=1 的 proposer 做最终 flush；GRPO 的 incomplete pending stage 不被强行 admission。

每个 optimizer update 对应一个八 completion 的 frozen-pool group；pool 是否更新不影响下一次 update 的推进。在线 dataset 只在 optimizer 成功后提交该组的 domain transition。

#### 5.4.9 Checkpoint 与 resume

每个 group 结束后写：

```text
checkpoints/stage_XXXX/
├── group_YY.json              pool version/reward/loss/outcomes/microbatch
├── adapter/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── README.md
├── trainer_state.pt           optimizer/scheduler/torch/cuda RNG
├── stage_state.json           stage/update/pool/token/GPU/device 摘要
└── gpu-boundary.csv           仅 boundary 时
```

Searcher state 另保存 Python `random.Random` state、stage、updates、groups、pool version、zero variance、tokens、GPU seconds 和最近可恢复 checkpoint path。恢复模型时再加载 adapter、optimizer、scheduler、CPU/CUDA RNG；coordinator checkpoint 同时恢复 pool、seen、pending candidates、candidate archive 和 cached signals。

每组 rollout、reward diagnostics 与 pending spec 使用追加式 journal；optimizer、scheduler、RNG 和 LoRA state 由 Verl checkpoint 保存。[写入字段](../src/rlalpha/search/grpo/stage_coordinator.py)

恢复采用两阶段协议：reward 先产生 pending transition，optimizer 完成后才写 domain journal；只有和同一步模型 checkpoint 配对的 snapshot 可恢复，后续未配对 journal 自动忽略。[保存调用顺序](../src/rlalpha/search/grpo/stage_coordinator.py)

---

## 6. 统一搜索生命周期

### 6.1 `run_search()` 的初始化

入口 [`run_search()`](../src/rlalpha/search/run.py#L61-L139) 依次执行：

1. 读取 experiment YAML 和 paths；
2. 读取当前 method 的 search YAML；
3. LLM 方法额外读取 model YAML；
4. 合并 `method_config + model_config + method/reward/seed/budget`；
5. 创建 `runs_root/experiment/method/reward/seed_N`；
6. 保存 `pip freeze`、GPU start snapshot、resolved config；
7. 加载 train 和 validation panel；
8. 用 reward name 构造 objective；
9. 创建容量 20 的 `PoolManager`；
10. 创建具体 searcher 和统一 coordinator；
11. 根据 checkpoint 或 `continue_seed_zero_from` 恢复；
12. 循环直到 valid-unique budget 耗尽；
13. validation 选择 snapshot；
14. 写 final pool、metrics、manifest、GPU end snapshot。

[`objective_for()`](../src/rlalpha/search/run.py#L26-L36) 的选择是源码分支，不读取 `configs/reward/*.yaml`。

### 6.2 一次 `SearchCoordinator.run_group()`

完整顺序：

```text
1. context()
   ├─ 对当前 pool 重算 PoolScore/weights
   ├─ pool formula/weights/version
   ├─ train objective、budget、最近 64 records
   └─ assert_train_only_context

2. searcher.propose(context, 8)
   └─ raw_proposals += returned count

3. 对每条 candidate 顺序处理
   ├─ parse/type invalid
   ├─ exact duplicate / GP rescore exception
   ├─ group 中途 budget exhausted
   ├─ memory cache → disk cache → evaluator
   ├─ validity against current pool
   ├─ valid unique budget += 1
   └─ 构造 PoolEntry

4. pool.score_candidates(all valid entries)
   └─ 所有人对同一个 baseline 独立打分

5. outcomes 写回 delta/shaped reward

6. append pending entries/scores
   └─ interval 到达才 flush admission

7. searcher.observe(outcomes)
   └─ GP 更新 population / GRPO 梯度更新

8. 同步 tokens/GPU seconds

9. cache retention
   ├─ pool hashes
   ├─ pending hashes
   └─ searcher.retained_hashes（GP population）

10. append candidate records + atomic checkpoint
```

exact duplicate 检查按候选顺序更新 `seen`，所以同一 proposal group 中相同 hash 的第一条可以执行，后续条目会被判 duplicate。near-duplicate 只对 admission 前的 current pool 检查，不对同组其他尚未 admission 的候选检查。

`seen.add(hash)` 发生在 signal evaluate/validity 之前：一个 AST-valid 但 evaluation/coverage 失败的公式以后再出现，会变成 `exact_duplicate`；`node=None` 的 parse failure 则不会加入 `seen`，同一坏文本可重复获得 -1。GP 的 `gp_rescore` 是唯一可能绕过 seen/pool duplicate 的分支，但条件是 hash 已在 `seen`；这类真正的 rescore 仍重做 signal validity 和 candidate delta，只是不增加 valid-unique budget。

### 6.3 Admission 周期对比

| 方法 | 每次 propose | admission interval | 一次 frozen admission 覆盖 | proposer 是否学习 reward |
|---|---:|---:|---:|---|
| Random | 8 | 1 group | 最多 8 个有效候选 | 否 |
| GP | 8 | 1 group | 最多 8 个，包括 rescore | fitness/population |
| Base LLM | 8 个不同 prompt | 1 group | 最多 8 个 | 否；仅看到新 pool context |
| GRPO LLM | 同 prompt 8 rollouts | 8 groups | 最多 64 个 | 每组一次 LoRA update |

所有方法每次 `consider_group()` 最多 admission 一个，所以相同 valid-unique budget 不代表相同最大 pool update 次数。

### 6.4 Budget ledger

[`BudgetLedger`](../src/rlalpha/search/models.py#L56-L78) 记录：

```text
limit
raw_proposals
valid_unique_evaluations
duplicates
invalid
tokens
gpu_seconds
```

循环条件只看 `valid_unique_evaluations >= limit`。如果本组开始时还有 1 个预算，候选按顺序执行到第一个 valid unique 后 ledger exhausted；本组后续候选仍产生 outcome，但被标记 `budget_exhausted`，不会市场计算。GRPO 仍会对完整八条 outcome 更新，其中被预算截断者 reward 为默认 0。

### 6.5 Signal cache 与 checkpoint

coordinator 的 `signals` 是进程内完整 signal dict；`SignalCache` 另有最多 64 条的内存 LRU，以及 `run_dir/cache/signals/<hash>.npy`。普通 evaluator 结果先进入非永久 cache；每组后只有 pool、pending、GP retained hashes 被保留并持久化，其余释放。

checkpoint 包含：

```text
ledger / seen / searcher state
pool version / pool entries / pool history
pending entries / groups since admission
```

不直接把 signal 数组序列化进 JSON；resume 按 hash 读 `.npy`，缺失时从 expression 重新 evaluate 并永久写 cache。`candidates.jsonl` 每次由内存 records 全量原子重写，最终再转换为 `candidates.parquet`。

原子性是“单文件”而非 bundle 级：`save_checkpoint()` 先写 `checkpoint.json`，再写 `candidates.jsonl`。两次 rename 之间崩溃时，恢复出的 ledger/pool 可能比 candidate archive 多一组记录；缺失记录不会从 checkpoint 自动重建。GRPO 还有上一节所述的 model/optimizer 与外层 JSON 不同事务问题。

**潜在风险**：resume 没有像 test transaction 那样校验 config/code/panel fingerprint，disk signal cache 的 key 也只有 `expr_hash`。在同一个 experiment/cell 路径下改数据或 evaluator 后继续 `--resume`，可能把旧 `.npy` signal、旧 pool 和新执行代码混在一起；`resolved_config.yaml` 还会在 load checkpoint 前被新配置覆盖。正式改变输入时应使用新 experiment id，当前 checkpoint 不能被当成 content-addressed artifact。[恢复顺序](../src/rlalpha/search/run.py#L61-L94)

### 6.6 Validation snapshot 与 final pool

每次 `coordinator.run_group()` 后，如果 pool version 与 `last_version` 不同：

1. 取当前 pool expressions；
2. 在 validation panel 重新执行表达式；
3. 用同名 reward 新建 validation objective；
4. 保存 train PoolScore、validation score、pool version 和当时 budget；
5. append 到 `checkpoints/snapshots.json`。

validation 不写回 pool，也不调用 searcher。搜索 prompt 仍只含 train objective。

预算结束后选择：

```python
max(snapshots,
    key=(validation.objective,
         -len(expressions),
         -pool_version))
```

因此最终优先 validation objective；平局偏好更小、再偏好更早的 pool。`final_pool.json` 是历史最佳 snapshot，不是 terminal training state 的别名。

当前 1000-budget 产物中，Base LLM/R0 terminal version=78、selected version=37；GRPO/R0 terminal version=26、selected version=25，直观证明二者可以不同。

---

## 7. Validation 与 Test Evaluation

### 7.1 Validation 与 test 的角色不同

validation 在搜索过程中反复读取，但只用于 snapshot selection；test 由 [`finalize_experiment()`](../src/rlalpha/evaluation/finalize.py#L206-L242) 在 experiment 级统一打开。

```text
train       proposer context、candidate reward、pool admission
validation  immutable pool snapshot selection
test        final pools 全部冻结后的唯一最终评估
```

`assert_train_only_context()` 通过把 context 序列化为小写字符串，拒绝 `validation_ic/validation_metric/test_ic/test_metric/test_return` 等词，见 [`leakage/guards.py`](../src/rlalpha/leakage/guards.py#L8-L15)。这是 prompt/context guard，不替代数据 split 本身的隔离。

### 7.2 Experiment-level transaction

finalization 先收集：

```text
runs_root/<experiment>/*/*/seed_*/final_pool.json
```

没有 pool 直接失败。`scope_input_hash` 覆盖：

- experiment id；
- 每个 `final_pool.json` 的路径、size、mtime、SHA-256；
- panel `index.json/build_manifest/risk_build_manifest`；
- data store、DSL evaluator/parser、factor calculator/combiner、neutralizer、portfolio、statistics、finalize 等 evaluation 源码指纹。

事务文件 `test_finalization.json` 首先写 `status=started`。以后再次调用：

- hash 相同且 complete：直接读旧 summary；
- hash 变化：抛出 `frozen experiment inputs changed after test finalization started`；
- started/failed 且 hash 相同：继续 cell finalization。

每个 cell 自己还有 schema version 6 的 input hash 和 `test/finalization.json`。这使“改了 evaluation 代码却静默复用旧 test 指标”无法发生。

### 7.3 跨方法共同 test universe

为避免不同表达式的缺失模式制造不可比样本，experiment 先取所有 final pool expression 的并集：

```python
shared_trade_mask = test.common_mask.copy()
for expression in union(all final pool formulas):
    shared_trade_mask &= isfinite(evaluate(expression))
```

mask packbits 后计算 `universe_hash`，并写 `test_universe.json`：formula 数、date/asset shape、eligible observations 和逐日 eligible 数。

**设计含义**：任一方法选中的一个高缺失表达式都会缩小所有方法的共同 test universe。这提高了横向可比性，但可能让最终样本由最严格的表达式决定；报告结果应和 `eligible_by_date` 一起审阅。

### 7.4 Final weights 在哪里拟合

[`finalize_cell()`](../src/rlalpha/evaluation/finalize.py#L102-L203) 首先加载 train、validation、test，并把同一表达式的 train/validation signal 沿 time 拼接：

```text
fit_signal_j  [T_train + T_validation, N]
fit_mask      [T_train + T_validation, N]
fit_exposure  [T_train + T_validation, N, 22]
fit_label     [T_train + T_validation, N]
```

`_standardized_residual_signals()`：

1. 对每个 raw signal 每日标准化；
2. 每天 `column_stack` 所有 (M) 个因子；
3. common 要求所有 (M) 个 signal 和 22 exposure 同时有限；
4. 一次 multivariate QR solve 残差化全部 factor columns。

这与搜索期 R1/R2 还有一个 sample-universe 差别：`RewardObjective._neutralized_inputs()` 分别对每个 signal 残差化，每个因子可有自己的 finite sample；final evaluation 则先 `column_stack` 全部因子，要求同一股票的 (M) 个 signal 同时有限后再做多目标回归。pool 中任一因子的缺失会收缩所有因子的 final-fit residualization universe。

label 单独残差化。最终用 [`RidgeCombiner(1e-3)`](../src/rlalpha/evaluation/finalize.py#L123-L130) 在 train+validation residual signal/label 上拟合 weights。这里调用的是 `fit()` 而非 `fit_prestandardized()`，所以每个 residual signal 在构造 ridge moments 前还会再做一次 daily winsorize/z-score。

**潜在风险**：test 路径把 `_standardized_residual_signals()` 的 residual 直接加权，没有复现 ridge fit 内部的第二次标准化。因此当前权重是在“二次标准化后的 train+validation residual”上拟合，却应用于“一次标准化后再 residualize 的 test signal”。这是当前实现的 representation mismatch，不应把两端误写成完全相同的变换。[fit 与 test combine 对比](../src/rlalpha/evaluation/finalize.py#L121-L142)

**当前实现**：无论 pool 是按 R0、R1 或 R2_LCB 搜出的，test 使用的最终 weights 都是 risk-neutral train+validation weights。R0 的实验含义是“搜索 admission objective 为 raw IC”，不是“最终 test combination 完全 raw”。

### 7.5 Test raw signal 与 risk-neutral signal

固定 weights 后，在 test 上有两条并行路径。

风险中性路径：

```text
raw test signals
→ daily standardize
→ jointly residualize against 22 exposures
→ weighted sum with fit weights
→ correlate with residualized 20-day label
```

raw 路径：

```text
raw test signals
→ daily standardize, no neutralization
→ same fit weights
→ correlate with raw 20-day label
```

因此 `raw_ic` 表示 neutralization 前的信号/label 关系，但权重仍来自 risk-neutral fit。

[`_daily_correlations()`](../src/rlalpha/evaluation/finalize.py#L71-L81) 每天计算：

- Pearson：原始数值 correlation；
- rank：先 `scipy.stats.rankdata`，再 Pearson，即含平均 tie rank 的 Spearman correlation。

`test/rnic_daily.parquet` 包含：

```text
date
raw_ic
raw_rank_ic
rnic
rank_rnic
```

`metrics.json` 汇总 raw IC、RNIC、rank RNIC；raw rank IC 当前只保留 daily series，没有独立 summary 字段。

### 7.6 统计摘要

[`series_summary()`](../src/rlalpha/evaluation/statistics.py#L28-L42) 对有限 daily series 输出：

```text
n
mean
sample std (ddof=1)
Newey-West HAC SE, lag 20
HAC t = mean / SE
20-day moving-block bootstrap 95% CI
```

[`moving_block_bootstrap()`](../src/rlalpha/evaluation/statistics.py#L8-L25)：

1. block length 截到 `[1,T]`；
2. 可选 contiguous block start 为 `0,...,T-block`；
3. 有放回抽 `ceil(T/block)` 个 block；
4. 拼接后截到 T；
5. 计算 2,000 个 bootstrap mean 的 2.5%/97.5% quantile。

默认 seed=0，所以相同输入可复现。neutralization retention：

\[
retention=\frac{|mean(RNIC)|}{|mean(raw\ IC)|},
\]

raw mean 近零时为 NaN。`average_pair_correlation` 是对 train+validation residual signal pair 把二维 finite common positions flatten 后计算一次绝对 correlation，再对 pairs 平均；它不是平均 daily pair correlation。

### 7.7 Portfolio 时间轴与四 sleeve

[`PortfolioBacktester`](../src/rlalpha/evaluation/portfolio.py#L82-L131) 默认：

```text
rebalance_days=5
holding_days=20
sleeves=20/5=4
execution_delay=1
pnl_delay=1
activation_delay=2
```

signal day (t) 的 target：

- 在 (t+1) 记录该 sleeve 从旧 target 到新 target 的 turnover；
- 在 (t+2) 替换 sleeve weight 并开始 PnL；
- 每 5 日更新下一个 sleeve；
- 同一 sleeve 每 20 日轮换回来，相当于 20 日持有。

两个 portfolio 都以 risk-neutral `combined` 为排序 score；`raw_combined` 只用于 raw IC，不进入回测。`eligible` 只在 target 形成日筛选股票：持有期间若股票后来不再满足当日 membership/共同 mask，代码不会强制日内清仓，而是等该 sleeve 下一次轮换；只有 realized return 缺失时按下述规则记 0 贡献。

每日实际 portfolio weight 是四个 sleeve 的均值。刚启动时尚未填满四个 sleeve，所以 gross exposure 从 0 逐步爬升到 1。

PnL 使用 `daily_total_return`，不是 20 日 label：

\[
R^{gross}_t=\sum_{i:r_{t,i}\ finite}w_{t,i}r_{t,i}.
\]

held position 的 return 缺失时，该位置从 dot product 排除，数值上相当于贡献 0，同时累计 `missing_held_returns` 审计计数。

### 7.8 Dollar-neutral target

[`dollar_neutral_target()`](../src/rlalpha/evaluation/portfolio.py#L8-L20) 在 eligible 且 score 有限的股票中 stable sort：

```text
count = floor(20% * eligible), at least 1
bottom count  equal-weight short, total -0.5
top count     equal-weight long,  total +0.5
```

eligible 少于 10 时返回全零 target。long/short 分别等权，不做 volatility scaling 或行业约束。

### 7.9 Fully-neutral QP

[`project_fully_neutral()`](../src/rlalpha/evaluation/portfolio.py#L23-L69) 在共同 eligible universe 上定义 nonnegative `plus/minus`：

\[
w=plus-minus.
\]

若 exposure 有 22 列，QP 排除 intercept，只约束后 21 列；net neutrality 由 gross constraints 保证：

\[
\sum plus=0.5,
\qquad
\sum minus=0.5,
\qquad
X_{1:}^Tw=0.
\]

另有：

\[
0\le plus_i,minus_i\le0.02.
\]

score 高于或等于中位数者才可做多，低于中位数者才可做空。目标函数：

\[
\min_w\|w-w^{dollar}\|_2^2+10^{-4}\|w\|_2^2.
\]

它不会把 support 限制在原 top/bottom 20%；为了满足 21 个风险约束，优化器可以在各自 score 半区使用更多股票。

求解依次尝试 OSQP、CLARABEL。若 eligible 少于 `ceil(1/max_weight)=50`，或两个 solver 都不返回 optimal/optimal_inaccurate，则返回 `None`。backtester 此时不清仓，也不改用当日 dollar-neutral target，而是让该 sleeve保持旧 weight，并把当前 signal day 标记 infeasible。代码接受 `optimal_inaccurate`，对返回解只记录约束残差，没有再用 tolerance 做一次拒绝检查。

审计字段包含 solver/status、net、gross、max weight、max residual risk exposure 和 signal day。

### 7.10 成本与 portfolio metrics

测试固定 0bps 和 10bps one-way：

\[
R^{net}_t=R^{gross}_t-\frac{cost\_bps}{10000}\cdot turnover_t.
\]

turnover 是被替换 sleeve 的 `sum(abs(new-old)) / 4`，记录在执行日 (t+1)。[`portfolio_metrics()`](../src/rlalpha/evaluation/portfolio.py#L134-L151) 输出：

```text
annual_return       arithmetic daily mean × 252，不是 CAGR
annual_volatility   sample std × sqrt(252)
sharpe              mean/std × sqrt(252)
max_drawdown        cumprod(1+return) 相对 running max
average_turnover
average_gross
average_net
infeasible_days
missing_held_returns
```

另从每日 weight 和全部 22 exposure 计算 realized exposures：

\[
E^{portfolio}_{t,k}=\sum_iw_{t,i}X_{t,i,k}.
\]

若任一实际持仓缺 exposure，该天所有 realized exposure 设 NaN 并记录。`max_realized_risk_exposure` 包括 intercept 列，所以同时覆盖 net exposure 和行业/style exposure。

### 7.11 Cell 与 experiment 输出

单 cell 搜索目录：

```text
method/reward/seed_N/
├── resolved_config.yaml
├── manifest.yaml
├── checkpoint.json
├── candidates.jsonl
├── candidates.parquet
├── train_metrics.json
├── validation_metrics.json
├── final_pool.json
├── cache/signals/*.npy
├── checkpoints/snapshots.json
├── environment/{pip-freeze,gpu-start,gpu-end}.csv|txt
├── logs/search.log
└── test/
    ├── finalization.json
    ├── metrics.json
    ├── rnic_daily.parquet
    ├── dollar_neutral_daily.parquet
    ├── fully_neutral_daily.parquet
    └── exposures.parquet
```

GRPO 的 `checkpoints/stage_*` 结构见 5.4.9。

experiment 级 evaluation/report 输出：

```text
test_finalization.json
test_universe.json
evaluation_summary.json
search_efficiency.{csv,parquet}
pool_quality.{csv,parquet}
portfolio_results.{csv,parquet}
paired_comparisons.{csv,parquet}
cross_method_summary.{csv,parquet}
report.md
```

[`build_report()`](../src/rlalpha/reporting/build.py#L82-L187) 的 paired comparisons 包括：同 reward 下每种方法减 Random，以及 GRPO/R1、GRPO/R2_LCB 分别减 GRPO/R0。RNIC 差使用同日差值的 HAC/block bootstrap；Sharpe 差使用 fully-neutral 10bps 日收益的 paired 20-day block resampling。

多 seed 时先取两边共有的 seed 集合，再在每个日期分别对这些 seed 求平均，最后对两条“跨 seed 平均后的日序列”做差/重采样；并不是把所有 seed-day 当成独立 observation。`cross_method_summary` 的 seed CI 则基于 cell-level test RNIC 的 Student-t interval，只有一个 seed 时上下界都退化为该单值。

机器可读字段可按下面定位：

| Artifact | 字段 |
|---|---|
| `test/finalization.json` | `status, input_hash, metrics_hash`（started 时尚无 metrics hash） |
| `test/metrics.json` | `input_hash, pool_version, pool_size, expressions, ridge_weights, average_pair_correlation, raw_ic, rnic, rank_rnic, neutralization_retention, portfolios, limitations` |
| `test/rnic_daily.parquet` | `date, raw_ic, raw_rank_ic, rnic, rank_rnic` |
| `test/{dollar_neutral,fully_neutral}_daily.parquet` | `date, gross_return, turnover, missing_held_returns, infeasible, net_return_0bps, net_return_10bps` |
| `test/exposures.parquet` | `date, portfolio, missing_held_exposures` 加全部 22 个 exposure name |
| `test_universe.json` | `scope_input_hash, universe_hash, formula_count, dates, assets, eligible_observations, eligible_by_date` |
| `evaluation_summary.json` | 以 cell 相对路径为 key；value 为 `status+metrics` 或 `status+error` |
| `search_efficiency` | `method,reward,seed,raw_proposals,valid,unique,admitted,pool_size,tokens,gpu_hours,wall_hours` |
| `pool_quality` | train/validation objective、test raw IC/RNIC/HAC/CI、pair correlation、retention、pool size |
| `portfolio_results` | method/reward/seed、portfolio/cost、return/Sharpe/drawdown/turnover/gross/net、risk/failure/missing counters |
| `paired_comparisons` | comparison、matched seeds/dates、delta RNIC/HAC/CI、delta 10bps fully-neutral Sharpe/CI、interpretation |
| `cross_method_summary` | method/reward、seed 数、test RNIC mean/std/seed CI、平均 HAC t 和 pool size |

`search_efficiency.valid` 是 `candidates.parquet.valid` 的行数，GP rescore 也可能计入；`unique` 才是 ledger 的 `valid_unique_evaluations`。这两个字段名很接近，但预算解释必须使用后者。[report 聚合](../src/rlalpha/reporting/build.py#L91-L119)

---

## 8. 实验配置、调度与运行

### 8.1 CLI 命令图

[`cli.py`](../src/rlalpha/cli.py#L23-L105) 注册：

```text
rlalpha doctor
rlalpha data validate
rlalpha data build
rlalpha risk build
rlalpha factor eval
rlalpha search run
rlalpha matrix run
rlalpha evaluate run
rlalpha report build
```

可使用 console script `rlalpha`，也可统一写 `python -m rlalpha.cli`。以下命令以后一种形式为准。

### 8.2 环境与 data/risk 准备

```bash
conda activate rlalpha
cd /home/sunyuxiang/rl_alpha/ours

python -m pip install -e .
python -m pip install -r requirements-llm.lock

python -m rlalpha.cli doctor \
  --config configs/experiment/preliminary_screen.yaml

python -m rlalpha.cli data validate \
  --config configs/data/sp500.yaml

python -m rlalpha.cli data build \
  --config configs/data/sp500.yaml

python -m rlalpha.cli risk build \
  --config configs/data/sp500.yaml
```

`doctor` 只读检查 data discovery、关键 package versions、唯一完整模型、OSQP/CLARABEL 和 GPU，见 [`doctor.py`](../src/rlalpha/doctor.py#L28-L70)。data build 必须先于 risk build，因为后者读取 panel index 和 membership Zarr。

单表达式 smoke：

```bash
python -m rlalpha.cli factor eval \
  --expr 'CSRank(Delta($close,20))' \
  --split train
```

### 8.3 模型与 GRPO smoke

Base LLM structured generation：

```bash
CUDA_VISIBLE_DEVICES=4 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
RLALPHA_VLLM_MEMORY_UTILIZATION=0.18 \
python scripts/smoke_model.py --n 500 --seed 2026 \
  --output /data/sunyuxiang/rl_alpha/runs/base_llm_smoke_new_id/result.json
```

GRPO 常驻 cell 的至少两次真实 market-reward update：

```bash
CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/smoke_grpo.py --updates 2 \
  --run-dir /data/sunyuxiang/rl_alpha/runs/grpo_persistent_smoke_new_id
```

`smoke_grpo.py` 使用真实训练 panel、reward/pool 内核和一次 Ray/Verl/vLLM
生命周期；只有 actor 确实变化、PPO/KL/entropy 指标有效、LoRA checkpoint
包含 optimizer/scheduler/RNG 且最多保留两份时才通过。

### 8.4 单 cell

```bash
python -m rlalpha.cli search run \
  --method random \
  --reward r0 \
  --seed 0 \
  --budget 1000 \
  --experiment-id debug_random_r0 \
  --config configs/experiment/preliminary_screen_1000.yaml
```

合法 method：`random/gp/base_llm/grpo_llm`；reward：`r0/r1/r2_lcb`。`--resume` 默认 true；Typer bool option 的具体正反 flag 可通过 `python -m rlalpha.cli search run --help` 查看当前版本渲染。

### 8.5 Screening 与 confirmatory

1000-budget 全矩阵：

```bash
python -m rlalpha.cli matrix run \
  --config configs/experiment/preliminary_screen_1000.yaml \
  --experiment-id preliminary_screen_1000_v2
```

正式 5000 screening：

```bash
python -m rlalpha.cli matrix run \
  --config configs/experiment/preliminary_screen.yaml \
  --experiment-id preliminary_screen_5000

python -m rlalpha.cli evaluate run \
  --experiment-id preliminary_screen_5000 \
  --config configs/experiment/preliminary_screen.yaml

python -m rlalpha.cli report build \
  --experiment-id preliminary_screen_5000 \
  --config configs/experiment/preliminary_screen.yaml
```

confirmatory 当前配置是六组 method/reward × 三个 seeds、每 cell 20,000：

```bash
python -m rlalpha.cli matrix run \
  --config configs/experiment/confirmatory.yaml \
  --experiment-id confirmatory

python -m rlalpha.cli evaluate run \
  --experiment-id confirmatory \
  --config configs/experiment/confirmatory.yaml

python -m rlalpha.cli report build \
  --experiment-id confirmatory \
  --config configs/experiment/confirmatory.yaml
```

confirmatory 的 seed 0 从 `experiment.continue_seed_zero_from` 指定 experiment 复制 checkpoint/snapshots，再把 ledger limit 提高到 20,000；seed 1/2 从头开始。若正式 screening ID 是 `preliminary_screen_5000`，该字段也必须同步指向它。

### 8.6 Matrix cell 展开

[`_run_matrix_unlocked()`](../src/rlalpha/matrix/runner.py#L55-L158) 支持两种配置形状：

```yaml
methods × rewards × seeds       # Cartesian product
```

或：

```yaml
cells: [[method,reward], ...]
seeds: [...]
```

preliminary 是 4×3×1=12 cells；confirmatory 是 6×3=18 cells。每个 cell 启动独立 Python subprocess：

```text
python -m rlalpha.cli search run
  --method ... --reward ... --seed ... --budget ...
  --experiment-id ... --config ...
```

stdout/stderr 合并追加到 `logs/search.log`。

### 8.7 CPU/GPU 调度

当前物理 GPU 映射：

| Method | GPU | Free-memory threshold | vLLM utilization |
|---|---:|---:|---:|
| Base LLM | 4 | 14 GiB | 0.18 |
| GRPO | 2/3 | 34/28 GiB | 0.18/0.15 |
| Random/GP | CPU | N/A | N/A |

GRPO GPU 由 `(seed + reward_offset) % 2` 在 2/3 间交错，reward offset 为 R0=0、R1=1、R2=2。CPU 最多 `max_cpu_jobs=4`；每个 CPU cell 设置 OMP/MKL/OpenBLAS/Numba threads，当前为 8。

runner 轮询 `nvidia-smi` free memory，目标 GPU 已被本 runner/external cell 占用或未达 threshold 时等待，不会跨 GPU 建 distributed collective。

### 8.8 OOM、失败隔离与 runner restart

只把包含以下文本的 GRPO failure 视为可自动降 microbatch 的 CUDA OOM：

```text
cuda out of memory
torch.outofmemoryerror
cublas_status_alloc_failed
```

microbatch 依次 `8→4→2→1`，rollout group 仍为 8。普通错误将 cell 标记 failed，但其他 running/pending cells 继续。再次运行同一 matrix 时，complete 跳过，failed/missing 重新启动。

若 runner 自己重启，发现某 cell `status=running` 且 PID 存活，会作为 external process 监控；PID 消失后以 `train_metrics.valid_unique_evaluations >= budget` 判定已完成，否则重新排队。

experiment 根目录的 `matrix_runner.lock` 使用 timeout=0；第二个 runner 会立即报“another matrix runner owns lock”，避免双重调度。

**`--no-resume` 边界**：matrix 层的 `resume=False` 只让 runner 忽略旧 `cell_state.json`；它启动 child command 时没有附加 `--no-resume`，而 `search run` 默认仍是 resume。因此旧 cell 目录里只要有 `checkpoint.json`，`matrix run --no-resume` 启动的子进程依然会加载它。[child command](../src/rlalpha/matrix/runner.py#L107-L115)

即使直接调用 `search run --no-resume`，同一 run dir 下旧 `checkpoints/snapshots.json` 仍会被无条件读入，disk signal cache 也不会清空；最终 snapshot selection 可能混入旧 run。需要真正从头跑时应使用新的 experiment id，当前两个 `--no-resume` flag 都不能在复用目录时保证完全 fresh search。[snapshot 加载](../src/rlalpha/search/run.py#L95-L108)

### 8.9 一键 orchestrator

[`scripts/run_full_experiment.sh`](../scripts/run_full_experiment.sh#L1-L45) 循环执行：

```text
preliminary matrix until all status complete
→ preliminary evaluate until transaction complete
→ preliminary report
→ confirmatory matrix until complete
→ confirmatory evaluate/report
→ full_experiment_complete.json
```

```bash
bash scripts/run_full_experiment.sh
```

每次 matrix/evaluate 失败后等待 60 秒重试；单次命令尾部 `|| true` 使 orchestrator 自己不因暂时失败退出。

### 8.10 仓库内其他脚本

除 orchestrator 外，[`scripts/`](../scripts) 中还有五个薄封装/诊断脚本。它们不会创建新的搜索算法：

| 脚本 | 行为 | 与 CLI 的关系 |
|---|---|---|
| [`validate_data.py`](../scripts/validate_data.py) | 对 `/data/sunyuxiang/rl_alpha` 做 QA，失败 exit 2 | 等价于固定路径版 `data validate` |
| [`build_panel.py`](../scripts/build_panel.py) | 用硬编码 raw/processed 路径只建 panel | 等价于 `data build`，**不**建 risk exposures |
| [`inspect_environment.py`](../scripts/inspect_environment.py) | 用 preliminary config 调 `run_doctor()` | 固定配置版 `doctor` |
| [`locate_model.py`](../scripts/locate_model.py) | 扫描 `--root`，要求恰好一个完整 Qwen3.5-2B | 模型定位诊断；0/多个候选 exit 2 |
| [`smoke_factor.py`](../scripts/smoke_factor.py) | 在 `[300,20]` synthetic feature 上执行一个复杂 DSL | 不读真实 panel 的快速 kernel smoke |

正式运行建议优先使用 Typer CLI，因为它接受 config/path override；这些脚本中的绝对路径是当前机器辅助入口，不是可移植实验接口。`smoke_model.py`、`smoke_grpo.py` 和 `run_full_experiment.sh` 已分别在 8.3、8.9 说明。

### 8.11 排障路径

某 cell 失败时依次查看：

```text
<experiment>/matrix_state.json
<cell>/cell_state.json
<cell>/logs/search.log
<cell>/train_metrics.json
<cell>/checkpoint.json
<cell>/environment/gpu-start.csv
```

GRPO 再查看最近 `checkpoints/stage_*/stage_state.json`、`group_*.json` 和 `gpu-boundary.csv`。判断“搜索完成”应看 ledger valid-unique 与目标 budget；判断“可用于正式结论”还必须看 `test_finalization.json`、`evaluation_summary.json` 和 report artifacts。

---

## 9. 配置消费审计

### 9.1 配置加载规则

[`load_yaml()`](../src/rlalpha/config.py#L44-L54) 支持 `defaults` 相对路径递归 deep merge；同名 scalar/list 由后者覆盖，dict 递归合并。`load_paths()` 只提取 YAML 的 `paths`，再应用：

```text
RLALPHA_CODE_ROOT
RLALPHA_RAW_DATA_ROOT
RLALPHA_PROCESSED_ROOT
RLALPHA_CACHE_ROOT
RLALPHA_RUNS_ROOT
RLALPHA_MODEL_SEARCH_ROOT
```

`PathsConfig(extra="forbid")` 只约束 paths 对象；experiment/search/model 的其他字段是普通 dict，没有统一 schema validation。

| Path field | 当前运行用途 |
|---|---|
| `code_root` | 定位 method/model YAML、subprocess cwd、repo manifest |
| `raw_data_root` | discovery/QA/panel/risk raw input |
| `processed_root` | panel/risk artifact 与 `PanelStore` |
| `runs_root` | cell、matrix、test、report 输出 |
| `cache_root` | doctor 检查可写；当前 search signal cache 实际放在各 `run_dir/cache`，未用全局目录 |
| `model_search_root` | doctor 扫描；searcher 的无显式 path fallback 直接读同名环境变量或硬编码默认，不读取这个 PathsConfig 字段 |
| `alphagen_root/quantevolver_root` | 只进入 repo version manifest，不参与候选生成或 reward |

### 9.2 Data/reward/evaluation 配置

| 配置字段 | 声明 | 当前消费情况 | 真实来源 |
|---|---|---|---|
| `data.warmup_start` | 2008-01-02 | 未传入 builder/store | 原始数据自然从 2008 开始 |
| `data.train/validation/test` | 三段日期 | 未传入 split | `data/splits.py` 硬编码 |
| `data.horizon_trading_days` | 20 | 未传入 panel | `range(2,22)` 硬编码 |
| `data.execution` | next_close | 文档性 | label/backtester 源码语义 |
| `fundamental_lag_months` | 6 | 未从 YAML 读取 | `DateOffset(months=6)` |
| `fundamental_max_age_months` | 18 | 未从 YAML 读取 | builder 硬编码 18 |
| `reward.name/neutralized` | R0/R1/R2 | reward YAML 未加载 | CLI reward string 分支 |
| `reward.hac_lag` | 20 | YAML 未加载 | `objective_for(...hac_lag=20)` |
| `reward.critical_value` | 1.645 | YAML 未加载 | R2 constructor default |
| `evaluation.ridge_lambda` | 0.001 | eval YAML 未加载 | finalize 硬编码 `1e-3` |
| `evaluation.hac_lag` | 20 | eval YAML 未加载 | statistics default |
| `rebalance_days/holding_days` | 5/20 | eval YAML 未加载 | `PortfolioBacktester(5,20)` |
| `sleeves` | 4 | 不直接读取 | `holding/rebalance` 推导 |
| `one_way_cost_bps` | `[0,10]` | eval YAML 未加载 | finalize 硬编码 tuple |
| `fully_neutral_max_weight` | 0.02 | eval YAML 未加载 | finalize 硬编码 0.02 |
| bootstrap samples/block | YAML 未声明 | 不可配置 | 2,000 / 20 日硬编码默认 |
| execution/PnL delay | YAML 未声明 | 不可配置 | `PortfolioBacktester` 默认 1/1 |
| long-short quantile | YAML 未声明 | 不可配置 | `dollar_neutral_target` 默认 0.20 |
| QP penalty/solver | YAML 未声明 | 不可配置 | `1e-4`；OSQP 后 CLARABEL |

**当前事实**：这些 YAML 可表达设计意图，但修改它们多数不会改变当前运行。正式实验若改变参数，应同时确认 Python 执行路径，而不是只提交 YAML。

### 9.3 Search 配置

| Method/field | 当前是否消费 | 备注 |
|---|---|---|
| Random `max_depth` | 是 | 传给 `RandomSearcher` |
| Random `max_nodes` | 否 | AST 固定上限 21 |
| Random `group_size/budget` | method YAML 否 | 实际来自 experiment/CLI |
| GP `population_size` | 是 | 默认 128 |
| GP `tournament_size` | 是 | 默认 5 |
| GP `elitism` | 仅实例属性 | selection 与 checkpoint 未引用该字段 |
| GP 四类概率 | 否 | 源码 draw threshold 硬编码 |
| GP `offspring_per_pool_batch` | 否 | group size 来自 experiment |
| Base `group_size` | 否 | experiment `proposal_group_size` |
| Base `temperature/response_length` | 否 | sampling 1.0/128 硬编码 |
| GRPO `staged_frozen_pool` | 否 | 类行为始终 staged |
| GRPO `group_size` | 否 | `rollout_group=8` |
| GRPO `prompt_groups_per_stage` | 否 | `admission_group_interval=8` |
| Pool `min_delta` | 无声明 | `PoolManager` 默认 `1e-5` |
| Reward ridge | 无声明 | `RewardObjective` 默认 `1e-3` |
| Signal validity thresholds | 无声明 | coverage .8、days 252、assets 100、variable .8、pool corr .95 |

Random 的 `max_depth` 只控制生成递归深度；`sample_ast()` 自身只复核 nodes/lookback，不再调用 `validate_limits(depth=6)`。因此把 YAML `max_depth` 改到 6 以上可能让 Random 直接产生 parser 标准之外的深 AST，而 `max_nodes` 仍固定 21。当前声明值 6 没触发这个缺口。[采样检查](../src/rlalpha/dsl/grammar.py#L9-L31)

Experiment 中实际消费：

```text
methods/rewards 或 cells
seeds
valid_unique_budget
proposal_group_size
pool_capacity
max_cpu_jobs
cpu_threads_per_job
continue_seed_zero_from
```

`auto_start_expensive_jobs` 当前没有被 matrix runner 读取。

`proposal_group_size` 对 Random/GP/Base 是普通 batch size，但 GRPO `propose()` 强制 `n==8`；把 experiment 值改成其他数字会让 GRPO cell 失败，而不是自动采用 YAML 中的 `rollout.n`。

### 9.4 Model/actor 配置

| 字段 | Base LLM | GRPO | 说明 |
|---|---|---|---|
| `model.path` | 使用 | 使用 | 本地模型路径 |
| `model.repository/revision` | 未用 | 未用 | 仅随 resolved config/manifest 留档，loader 不校验 revision |
| `model.fingerprint.*` | 未用 | 未用 | loader 不重算 weights/config/tokenizer SHA-256 |
| `local_files_only/trust_remote_code` | 行为硬编码 true | 行为硬编码 true | 字段值本身未统一读取 |
| `use_remove_padding` | 未用 | 未用 | 不进入当前实现 |
| `enable_gradient_checkpointing` | N/A | 行为硬编码 enable | 字段值未读取 |
| `rollout.name=vllm` | Base 确实 vLLM | GRPO 不是 vLLM | GRPO 用 HF generate |
| `rollout.n` | group 从 experiment | 类常量 8 | 字段未直接读取 |
| `rollout.temperature/response_length` | 未读取，硬编码 1.0/128 | 未读取，硬编码 1.0/128 | 当前声明值恰好相同 |
| `rollout.max_model_len` | 使用 | 不使用 | Base vLLM only |
| `actor.use_dynamic_bsz` | N/A | 未用 | microbatch 来自环境变量 |
| `actor.use_kl_loss/coef` | N/A | 未用 | 当前 loss 无 KL |
| `actor.learning_rate` | N/A | 硬编码同值 | 改 YAML 不生效 |
| LoRA rank/alpha/targets | N/A | 硬编码同值 | 改 YAML 不生效 |

### 9.5 建议未来修复与当前事实分离

本文不修改实现，但维护者若要让配置成为可信实验接口，优先级可按：

1. 为 experiment/data/reward/eval/model 建 Pydantic schema；
2. 让 builder、objective、evaluation 和 searcher 从 resolved config 获取所有参数；
3. 把 hardcoded 默认值保留为 schema defaults，而不是散落在函数体；
4. 在 manifest 中记录“resolved effective values”，而非仅记录 YAML；
5. 为每个字段增加“改 YAML 会改变行为”的测试；
6. 若保留当前 GRPO loss，重命名/文档化算法边界；若要完整 GRPO，则显式接入 old/reference policy、ratio/clipping/KL。

这些是建议，不是当前已实现功能。

---

## 10. 当前实验状态与已知边界

> **快照口径**：本节只描述 2026-08-09、commit `52d5048` 时工作区中的 artifact。它不是代码的永久性质；续跑、重新筛选或 test finalization 后应同步更新本节。

### 10.1 `preliminary_screen` 实际完成了什么

[`output/preliminary_screen/matrix_state.json`](../output/preliminary_screen/matrix_state.json) 中有 `4 methods × 3 rewards × 1 seed = 12` 个 cell，快照时全部为 `status="complete"`。但这里的“完成”要与实验名字分开理解：12 个 state record 的 `budget` 都是 **1000**，每个 cell 的 [`resolved_config.yaml`](../output/preliminary_screen/random/r0/seed_0/resolved_config.yaml) 也记录 `valid_unique_budget: 1000`。因此它们是 1000-valid-unique screening 结果，并不是计划中的 5000-budget preliminary 结果。

每个 cell 均已生成 `final_pool.json`，且所选 validation snapshot 都含 20 个因子。下面的数据来自各 cell 的 `checkpoint.json`、`train_metrics.json` 与 `validation_metrics.json`；`selected/terminal` 表示“validation 选中的 pool version / 搜索结束时 train pool version”。

| Method | Reward | Raw proposals | Valid unique | Duplicate | Invalid | Selected/terminal version | Validation objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_llm` | R0 | 1,608 | 1,000 | 165 | 436 | 37 / 78 | 0.083158 |
| `base_llm` | R1 | 2,136 | 1,000 | 196 | 935 | 23 / 64 | 0.061990 |
| `base_llm` | R2_LCB | 1,744 | 1,000 | 111 | 629 | 34 / 58 | 0.048936 |
| `gp` | R0 | 9,640 | 1,000 | 851 | 2,157 | 120 / 122 | 0.083972 |
| `gp` | R1 | 8,264 | 1,000 | 907 | 2,198 | 89 / 89 | 0.066577 |
| `gp` | R2_LCB | 9,224 | 1,000 | 662 | 1,922 | 60 / 117 | 0.053910 |
| `grpo_llm` | R0 | 1,696 | 1,000 | 258 | 433 | 25 / 26 | 0.072562 |
| `grpo_llm` | R1 | 1,784 | 1,000 | 162 | 621 | 23 / 27 | 0.057449 |
| `grpo_llm` | R2_LCB | 1,904 | 1,000 | 285 | 616 | 20 / 28 | 0.050579 |
| `random` | R0 | 3,008 | 1,000 | 1,090 | 914 | 50 / 81 | 0.088097 |
| `random` | R1 | 3,488 | 1,000 | 1,278 | 1,210 | 67 / 77 | 0.068239 |
| `random` | R2_LCB | 3,464 | 1,000 | 1,270 | 1,192 | 72 / 76 | 0.046577 |

表中 `raw ≠ valid + duplicate + invalid` 并不一定是账错：同一个 raw proposal 可能因本组 budget 已满而未进入 market evaluation；GP 还会在 pool 改变后重评 population，其中 hash 已见的真正 rescore 不计入 valid-unique budget，却会形成额外 valid outcome。准确的计数口径以 [`BudgetLedger`](../src/rlalpha/search/models.py#L55) 和 [coordinator 分支](../src/rlalpha/search/coordinator.py#L51-L91) 为准。

GRPO 的三个终态 checkpoint 还分别保留了未构成完整 64-candidate stage 的 `pending`：R0 为 16、R1 为 37、R2_LCB 为 24。它们已经获得 outcome/reward，并计入 1000 budget，但因为没有到 admission boundary，不会再改变 pool。这正是第 6.3 节所说的“budget 停止”和“pool 更新周期”不是同一边界。

### 10.2 Validation snapshot 不是 terminal train pool

上述 12 个 cell 中，仅 `gp/R1` 的 selected version 与 terminal version 相同，其余都不同。以 [`random/R0/final_pool.json`](../output/preliminary_screen/random/r0/seed_0/final_pool.json) 为例，搜索最终曾走到 version 81，但最终文件保存的是 validation objective 最好的 version 50。

这一区别是刻意的 model selection 行为：

```text
train reward 驱动 pool version 0 → ... → V_terminal
                          │
每次 admission 后在 validation 重算 objective
                          │
                          └── argmax snapshot = V_selected → final_pool.json
```

因此分析搜索动力学应读 `checkpoint.json.pool/history`；分析最终 test 输入应读 `final_pool.json`。把 `final_pool.pool_version` 当作“算法一共成功 admission 多少次”会得出错误结论。[snapshot 产生与落盘代码](../src/rlalpha/search/run.py#L93-L131)体现了这两个状态的分离。

同理，`final_pool.valid_unique_evaluations` 是该 snapshot 当时的 budget 位置，不是整次 run 的最终 1000；最终预算看 `train_metrics.valid_unique_evaluations` 或 `checkpoint.ledger`。例如 random/R0 的 selected version 50 出现在第 201 个 valid unique，而 run 最终走到 1000/version 81。

### 10.3 尚未完成的实验阶段

快照时 `preliminary_screen/` 下没有实验级 `test_finalization.json`、`test_universe.json`、`evaluation_summary.json` 或 `report.md`，cell 下也没有 `test/`，说明正式 preliminary test transaction/report 尚未完成；`output/confirmatory/` 也尚未生成。换句话说，目前可以讨论的是 train search 和 validation selection，不能把这 12 个 cell 的 validation objective 当成最终 test 结论。

`output` 本身是指向 `/data/sunyuxiang/rl_alpha/runs` 的符号链接，所以仓库内看到的相对路径与大盘实际存储位置是同一份数据，不是副本。迁移仓库或在另一台机器复现时必须重建该链接或显式传 `--output-root`。

### 10.4 1000/5000 复用陷阱

[`run_matrix()`](../src/rlalpha/matrix/runner.py#L60) 对已有 cell 的跳过条件只检查：

```python
existing.get("status") == "complete"
```

它不会比较旧 state 中的 `budget` 与本次 experiment YAML 的 `valid_unique_budget`。[`run_full_experiment.sh`](../scripts/run_full_experiment.sh#L11-L15) 的 `matrix_complete()` 同样只检查所有 record 的 `status == complete`。所以：

1. 先用 experiment id `preliminary_screen` 跑完 1000；
2. 再用相同 id 和 5000 配置调用 `matrix run`；
3. runner 会把 12 个旧 cell 全部跳过；
4. 后续 evaluation 可能把 1000-budget pool 当作 5000-budget 结果。

**当前安全做法**是让不同 budget 使用不同 `experiment-id`，或在确认不再需要旧结果后显式选择新的空 output 目录；不要仅覆盖 YAML。若 confirmatory 需要续接 seed 0，应同步检查 `continue_seed_zero_from` 指向的是期望 budget 的 experiment。[配置展开位置](../src/rlalpha/matrix/runner.py#L24)没有替调用者做这项语义校验。

### 10.5 结果解释边界

以下限制会影响“高 IC 是否能转化为可交易收益”的解释：

- **Borrowability**：原始借券文件只有当前快照，不能构造完整历史可借 universe；当前 panel membership 并未加入历史 borrowability 约束。因此 short leg 是研究性模拟，不是可直接执行的可借组合。
- **逐池 support 差异**：每个 cell 直接使用自己的 final pool complete-case support；报告 valid days/observations 解释样本量差异，不再让一个 cell 的缺失值改变其他方法的评价样本。
- **缺失 held return**：持仓日回报为 NaN 时，PnL 中按 0 处理，同时在 `missing_held_returns` 中计数；这避免整天 PnL 变 NaN，却隐含“缺失等于零收益”的估值假设。[实现](../src/rlalpha/evaluation/portfolio.py#L111)
- **Fully-neutral 不可行**：eligible 不足、solver 报错或状态非 optimal 时，代码保持该 sleeve 的旧权重，并将 signal day 标为 `infeasible`；它不 fallback 到当日 dollar-neutral target。不能只看 fully-neutral 收益，还应报告 stale-sleeve/fallback rate。[实现](../src/rlalpha/evaluation/portfolio.py#L121-L130)
- **R0 也不是无条件 raw**：R0 不做风险残差化，但仍使用 `common_mask`，而这个 mask 要求 22 维 exposure 全部有限。R0 与“只看 membership/price 的纯 raw universe”并不相同。[mask 定义](../src/rlalpha/data/store.py#L35)
- **最终权重改变 R0 语义**：test 阶段统一用 train+validation 的 risk-neutral ridge covariance 拟合 pool weights；R0 搜索期的 raw-IC 最优权重不会原样带入 test。[实现](../src/rlalpha/evaluation/finalize.py#L120-L130)
- **Partial experiment 可被 finalize**：`finalize_experiment()` glob 当前存在的 `final_pool.json`，但不核对 experiment YAML 中应有的全部 cells。因此缺 cell 时仍可能建立一笔“部分实验”transaction；它一旦 complete，后来补入缺失 pool 又会改变 scope hash并触发 frozen-input error，而不是自动扩展 transaction。[发现 cell 的代码](../src/rlalpha/evaluation/finalize.py#L205-L219)
- **Report 也允许 partial**：`build_report()` 只扫描当前已有的 `*/test/metrics.json`；没有检查 `test_finalization.status` 或期望 seed 数，甚至零 cell 也会写一个空 `report.md`。[扫描入口](../src/rlalpha/reporting/build.py#L82-L91)
- **Transaction 的 panel hash 是间接的**：scope hash 指纹化 `index.json/build_manifest.yaml/risk_build_manifest.yaml`，不直接 hash 巨大的 Zarr chunks。正常 builder 会同步 manifest；若有人绕过 builder 手工改 Zarr 而不改 manifest，input-change guard 检测不到。[fingerprint 列表](../src/rlalpha/evaluation/finalize.py#L31-L34)
- **Validation 多次查看**：每次成功 admission 都在同一 validation split 上选最优 snapshot，没有为尝试次数做 multiple-testing correction。它符合当前 model-selection 设计，但 validation objective 会有选择偏差，不能当 test estimate。
- **Wall time 是单次进程口径**：`train_metrics.wall_seconds` 从当前 `run_search()` invocation 开始计时；resume/OOM 前已耗时间不会累加。matrix `cell_state.wall_seconds` 也只记最后一次 attempt，report 的 GPU time有 checkpoint 累积，而 wall time 不是完整累计成本。
- **硬编码与命名边界**：第 9 章列出的 YAML 未消费字段、GP 概率硬编码、GRPO 无 ratio/clipping/reference KL，都是比较结果时必须随实验共同披露的实现条件。
- **WMA 的真实语义**：`WMA(x,w)` 的当前 observation 权重为 `w`、越旧越小；允许至多约 20% 缺失，并用有限 observation 对应权重重新归一化。这里的第二个参数是 window，不是任意权重向量。[实现](../src/rlalpha/dsl/evaluator.py#L70-L79)

### 10.6 快照状态检查清单

继续实验前建议依次确认，而不是只看目录是否存在：

```bash
jq 'to_entries | map({cell: .key, status: .value.status, budget: .value.budget})' \
  output/preliminary_screen/matrix_state.json

jq '{selected_pool_version: .pool_version, pool_size: (.expressions | length)}' \
  output/preliminary_screen/random/r0/seed_0/final_pool.json

jq '{terminal_pool_version: .pool_version, pool_size}' \
  output/preliminary_screen/random/r0/seed_0/train_metrics.json

find -L output/preliminary_screen -maxdepth 1 \
  -name 'test_finalization.json' -o -name 'evaluation_summary.json'
```

第一条确认 status 与实际 budget；中间两条分别读取 selected/terminal pool version 并确认容量；最后一条确认 experiment-level test 是否真正落盘。

---

## 11. 测试、维护与源码索引

### 11.1 核心保证到测试的映射

当前默认测试基线为 **`52 passed, 2 deselected`**。[`pyproject.toml`](../pyproject.toml#L30) 的默认 marker 表达式排除 `gpu`、`slow` 与 `real_data`，所以这个数字验证的是快速 CPU/synthetic 路径，不等于大模型、真实全量 Parquet 和 GPU 路径都已经在本次运行中重测。

| 保证 | 主要测试 | 具体覆盖 |
|---|---|---|
| Data contract 与时间边界 | [`test_data_contracts.py`](../tests/unit/test_data_contracts.py) | 价格调整、退市收益、membership 边界、`t+2:t+21` label、CCM 优先级、六个月 lag；未直接测 discovery schema/18 月 expiry |
| 泄漏防线 | [`test_sentinels.py`](../tests/leakage/test_sentinels.py) | future feature/label/context sentinel、fundamental 可用日、只读状态 guard |
| DSL 语义 | [`test_dsl.py`](../tests/unit/test_dsl.py) | parse/canonical/hash、depth/nodes、protected ops、rolling、NumPy/Torch 一致性、subtree cache、validity |
| Prompt 合同 | [`test_prompts.py`](../tests/unit/test_prompts.py) | pool/context/hint 的 prompt 内容及格式 |
| 风险模型 | [`test_risk.py`](../tests/unit/test_risk.py) | FF12 映射、22 维 exposure、横截面标准化、残差正交性 |
| Reward 与 pool | [`test_rewards_pool.py`](../tests/unit/test_rewards_pool.py) | fixed-universe ridge、R0/R1/R2、Newey–West、满池替换、group admission |
| Search/resume | [`test_search.py`](../tests/unit/test_search.py) | Random RNG 恢复、GP fitness invalidation、delta 重算、staged admission、OOM microbatch、matrix resume |
| GRPO online/Verl | [`test_verl_grpo_adapter.py`](../tests/unit/test_verl_grpo_adapter.py)、[`test_verl_stage_coordinator.py`](../tests/unit/test_verl_stage_coordinator.py) | 同组重复 reward 复用、一次 trainer 多 update、在线 dataset callback、paired resume 与旧语义拒绝；真实模型 loss 由独立 GPU smoke 验收 |
| Evaluation | [`test_evaluation.py`](../tests/unit/test_evaluation.py) | QP 基本约束、sleeve 延迟、turnover/cost、paired 统计、风险暴露摘要；未测 infeasible/missing-return 分支 |
| 组合链 synthetic | [`test_synthetic_pipeline.py`](../tests/integration/test_synthetic_pipeline.py) | DSL → validity → R0/R1/R2 → one-group pool admission；不落 final artifact |
| Reporting | [`test_reporting.py`](../tests/integration/test_reporting.py) | experiment summary 与报告文件生成 |
| 真实数据装载 | [`test_real_panel_store.py`](../tests/integration/test_real_panel_store.py) | 真实 Zarr split、shape/mask；默认 deselect |
| GPU acceptance 占位 | [`test_gpu_smokes.py`](../tests/integration/test_gpu_smokes.py) | marker test 主动 `skip`，只提示应另跑 `scripts/smoke_grpo.py`；它本身不执行 GPU |

测试是局部 contract，而不是统计结论证明。例如 `test_risk.py` 能证明 residual 与给定 exposure 数值正交，不能证明 Balanced-22 是唯一正确的资产定价模型；`test_search.py` 能证明恢复后 RNG 连续，不能证明某个随机 seed 的 pool 有稳定 out-of-sample alpha。

### 11.2 修改后最小回归集合

| 修改区域 | 至少重跑 |
|---|---|
| `data/`、split 或 label | `test_data_contracts.py test_sentinels.py test_real_panel_store.py` |
| `dsl/` 或 factor cache | `test_dsl.py test_synthetic_pipeline.py` |
| `risk/` | `test_risk.py test_rewards_pool.py test_evaluation.py` |
| `rewards/`、combiner、pool | `test_rewards_pool.py test_search.py test_synthetic_pipeline.py` |
| Random/GP/Base | `test_search.py test_prompts.py`，Base 再跑 GPU smoke |
| GRPO | `test_verl_grpo_adapter.py test_verl_stage_coordinator.py test_search.py test_gpu_smokes.py` |
| Evaluation/reporting | `test_evaluation.py test_reporting.py test_synthetic_pipeline.py` |
| Matrix/CLI/scripts | `test_search.py`，再用新的 experiment id 做 smoke |

可复制命令：

```bash
# 默认快速基线
conda run -n rlalpha pytest -q

# Data + leakage
conda run -n rlalpha pytest -q \
  tests/unit/test_data_contracts.py tests/leakage/test_sentinels.py

# Reward/search/evaluation 主链
conda run -n rlalpha pytest -q \
  tests/unit/test_rewards_pool.py tests/unit/test_search.py \
  tests/unit/test_verl_grpo_adapter.py tests/unit/test_verl_stage_coordinator.py \
  tests/unit/test_evaluation.py

# 需要真实 artifact 的显式测试
conda run -n rlalpha pytest -q -m real_data

# GPU acceptance 不能由 pytest 占位测试代替
CUDA_VISIBLE_DEVICES=2 conda run -n rlalpha \
  python scripts/smoke_grpo.py --updates 2
```

### 11.3 维护本文时的审计顺序

代码变更后，建议用下面的顺序更新本文，避免只改叙述而遗漏执行事实：

1. 从 [`cli.py`](../src/rlalpha/cli.py) 找公开参数与默认值；
2. 从 [`run.py`](../src/rlalpha/search/run.py) 或 [`finalize.py`](../src/rlalpha/evaluation/finalize.py) 沿调用链确认参数真正传到哪里；
3. 搜索配置 key 的读取位置，区分“YAML 中存在”和“Python 已消费”；
4. 检查 dataclass/state dict 是否新增字段，确认 checkpoint 向后兼容；
5. 用一个新 smoke experiment 读取 `resolved_config`、`checkpoint`、`final_pool` 和 transaction；
6. 运行对应局部测试，再运行默认全量测试；
7. 更新本节日期、commit、artifact 数字和已知边界。

如果算法公式改变，文档至少要同步更新四处：公式、shape/mask、checkpoint state、测试映射。只更新“方法介绍”而不更新恢复语义，会让读者无法复现实验。

---

## 附录 A：关键状态类型与 dataclass 字段

### A.1 Data、factor 与 pool

| 类型 | 核心字段 | 角色 |
|---|---|---|
| [`SplitPanel`](../src/rlalpha/data/store.py#L18) | `dates, permnos, features, daily_return, label, membership, exposures, target_slice` | 一个 split 连同 252 日历史前缀；计算可看历史，评分只取 target |
| [`ValidityResult`](../src/rlalpha/dsl/validity.py#L10) | `valid, reason, coverage, valid_days, variable_day_rate, max_pool_correlation` | signal-valid gate 的可审计结果 |
| [`FactorSignal`](../src/rlalpha/factors/calculator.py#L11) | `values, expr_hash, expression` | 因子值与 identity 的轻量包装；主搜索路径直接使用 ndarray/`PoolEntry` |
| [`PoolScore`](../src/rlalpha/factors/records.py#L6) | `objective, mean_ic, daily_ic, weights, standard_error` | 一整个 pool 的 objective 与组合权重 |
| [`CandidateScore`](../src/rlalpha/factors/records.py#L15) | `candidate_hash, pool_score, delta_objective, shaped_reward, replaced_hash` | 候选相对冻结 pool 的反事实分数 |
| [`PoolEntry`](../src/rlalpha/factors/records.py#L24) | `expression, expr_hash, signal, metadata` | pool 内可持久化公式及内存 signal |
| [`Admission`](../src/rlalpha/factors/pool.py#L9) | `admitted, candidate_hash, replaced_hash, delta, pool_version` | 一次 group 最多一项的状态转换 |

### A.2 Search 与 evaluation

| 类型 | 核心字段 | 角色 |
|---|---|---|
| [`Candidate`](../src/rlalpha/search/models.py#L10) | `node, generator, parents, raw_text` | 搜索器的提议；invalid LLM 文本也能保留 |
| [`CandidateOutcome`](../src/rlalpha/search/models.py#L26) | `valid, reason, market_evaluated, delta_objective, shaped_reward, metadata` | 一条候选最终审计记录 |
| [`SearchContext`](../src/rlalpha/search/models.py#L41) | `pool_version/formulas/weights, train_objective, budget, history_summary` | 传给搜索器、也进入 LLM prompt 的冻结上下文 |
| [`BudgetLedger`](../src/rlalpha/search/models.py#L55) | `raw_proposals, valid_unique_evaluations, duplicates, invalid, tokens, gpu_seconds` | 不同搜索法的统一预算/成本口径 |
| [`VerlGRPOStageCoordinator`](../src/rlalpha/search/grpo/stage_coordinator.py) | `stage, updates, pool_version, paired_optimizer_step` | 常驻 rollout/LoRA 与两阶段 domain commit 状态 |
| [`PortfolioResult`](../src/rlalpha/evaluation/portfolio.py#L72) | `weights, gross_returns, turnover, missing_held_returns, infeasible, audits` | 一个 portfolio variant 的完整日频执行记录 |

这些字段中，`signal`、模型对象与 optimizer tensor 不直接放进 JSON；checkpoint 只保存可序列化状态或另存 PyTorch 文件。恢复逻辑必须同时找齐通用 checkpoint 与搜索器私有 checkpoint，不能只复制 `checkpoint.json`。

---

## 附录 B：Artifact schema 与定位

### B.1 Data artifacts

| 路径 | 内容 | 主要消费者 |
|---|---|---|
| `<processed_root>/panel/index.json` | dates、permnos、shape、feature names、axes | `PanelStore.index`、transaction hash；fingerprint 在 manifest |
| `panel/features.zarr/{open,high,low,close,volume,return}` | `[T,N]` float32 feature | DSL evaluator |
| `panel/returns.zarr/daily_total_return` | `[T,N]` realized one-day return | portfolio PnL |
| `panel/returns.zarr/forward_return_20d` | `[T,N]` `t+2:t+21` label | reward/evaluation IC |
| `panel/membership.zarr/membership` | `[T,N]` bool historical S&P 500 | common mask |
| `panel/risk_exposures.zarr/exposures` | `[T,N,22]` float32 Balanced-22 | common mask、R1/R2、fully-neutral QP |
| `panel/qa_report.json` | schema/row/date/duplicate/coverage 检查 | data build 审计 |

### B.2 Cell artifacts

| 文件 | 关键字段/内容 | 何时看它 |
|---|---|---|
| `resolved_config.yaml` | 合并后的 experiment/method/model 配置 | 先确认声称运行了什么 |
| `cell_state.json` | running/complete/failed、时间、error | 定位 cell 级状态 |
| `candidates.jsonl/.parquet` | 每个 outcome、reason、reward、metadata | 分析生成质量与失败率 |
| `checkpoint.json` | ledger、pool、history、searcher state | 恢复与 terminal train state |
| `checkpoints/snapshots.json` | 每次 pool version 改变后的 train/validation snapshot | validation model selection |
| `checkpoints/stage_XXXX/adapter/` | LoRA adapter files | GRPO 模型恢复 |
| `checkpoints/stage_XXXX/trainer_state.pt` | optimizer、scheduler、Torch CPU/CUDA RNG | GRPO 可训练状态恢复 |
| `checkpoints/stage_XXXX/group_YY.json` | 单组 responses、outcomes、reward、loss、microbatch | GRPO 学习轨迹 |
| `checkpoints/stage_XXXX/stage_state.json` | stage/group/update/token/GPU counters | GRPO stage 恢复与审计 |
| `train_metrics.json` | terminal train objective/version/ledger | 训练汇总 |
| `validation_metrics.json` | 被选 snapshot 的 validation `PoolScore` | 最终选择摘要；完整轨迹看 `snapshots.json` |
| `final_pool.json` | selected snapshot、terminal version | test 的直接输入 |
| `manifest.yaml` | Python/package、三个 repo 状态、GPU、model config、split/convention、budget | 运行环境审计；当前 search 调用未传 `data_files`，不含 raw/panel artifact hash |

### B.3 Experiment artifacts

| 文件 | 内容 |
|---|---|
| `matrix_state.json` | 所有 cell 的调度状态、实际 budget、GPU、attempt |
| `test_finalization.json` | experiment transaction 状态、scope/finalization hashes、完成/失败 cell 数 |
| `test/metrics.json` | 逐池 support 的 valid days/observations 与 test 指标 |
| `evaluation_summary.json` | 各 cell 的 complete/failed 状态和内嵌 `metrics` |
| `search_efficiency.{csv,parquet}` | raw/valid/unique/admitted、pool size、token/GPU/wall cost |
| `pool_quality.{csv,parquet}` | train/validation/test IC、HAC t、相关性、retention |
| `portfolio_results.{csv,parquet}` | 两种 portfolio × 两档成本的收益、风险与执行指标 |
| `paired_comparisons.{csv,parquet}` | 对 Random 或 GRPO/R0 的 paired RNIC/Sharpe 差异 |
| `cross_method_summary.{csv,parquet}` | method/reward 跨 seed 汇总 |
| `report.md` | 上述机器可读表的 Markdown presentation layer |

cell test 文件由 [`finalize.py`](../src/rlalpha/evaluation/finalize.py#L103-L202) 写入，experiment/report 文件由 [`finalize.py`](../src/rlalpha/evaluation/finalize.py#L205-L242) 与 [`reporting/build.py`](../src/rlalpha/reporting/build.py#L82-L187) 写入。新增字段时应先改 schema/消费者测试，再更新此表。

---

## 附录 C：按调用链阅读源码

首次阅读时，不建议按目录字母顺序。下面这条路径能最快建立“一个候选如何变成最终 PnL”的因果链：

```text
cli.py
 ├─ data/panel.py → data/store.py
 ├─ risk/builder.py → risk/exposures.py → risk/neutralize.py
 ├─ search/run.py
 │   ├─ search/{random_search,gp,base_llm}.py
 │   ├─ search/grpo/{online_dataset,stage_coordinator,verl_reward_function,verl_trainer}.py
 │   └─ search/coordinator.py
 │       ├─ dsl/parser.py → dsl/evaluator.py → dsl/operators.py
 │       ├─ dsl/validity.py
 │       └─ factors/pool.py → rewards/{r0,r1,r2_lcb}.py
 └─ evaluation/finalize.py
     ├─ evaluation/statistics.py
     ├─ evaluation/portfolio.py
     └─ reporting/build.py
```

第二遍再读横切关注点：[`config.py`](../src/rlalpha/config.py)、[`manifest.py`](../src/rlalpha/manifest.py)、[`utils/io.py`](../src/rlalpha/utils/io.py)、[`utils/seed.py`](../src/rlalpha/utils/seed.py)、[`leakage/guards.py`](../src/rlalpha/leakage/guards.py) 与 [`matrix/runner.py`](../src/rlalpha/matrix/runner.py)。这样更容易判断某个行为属于算法本身，还是属于配置、持久化、泄漏防护或调度层。

---

## 附录 D：术语速查

| 术语 | 本项目中的精确定义 |
|---|---|
| raw proposal | 搜索器吐出的一个候选字符串/AST；未必合法、唯一或可市场求值 |
| valid unique evaluation | 首次出现、AST/signal 有效且实际计算 candidate delta 的候选；唯一预算单位 |
| group | 一次对同一 frozen pool 批量评分的候选集合 |
| admission | group 中 delta 最大且 `delta > min_delta` 的至多一个候选进入/替换 pool |
| pool version | 每次成功 admission 加 1；拒绝 group 不变 |
| terminal pool | 搜索预算耗尽时的 train pool |
| selected/final pool | 历史 pool versions 中 validation objective 最优的 snapshot |
| IC | 某日横截面 Pearson correlation，再对日期取时间序列统计 |
| rank IC | signal 与 label 分别做横截面 rank 后的 Pearson correlation |
| RNIC | signal 与 label 按同日 risk exposures 残差化后的 IC |
| retention | test IC 相对 train/validation 参考 IC 的保留比例；分母接近 0 时需谨慎解释 |
| HAC/Newey–West | 对每日 IC 均值的自相关/异方差稳健标准误估计 |
| LCB | `mean_ic - 1.645 × HAC_SE`，当前 R2_LCB objective |
| sleeve | 每 5 日再平衡、持有 20 日形成的四个错峰子组合之一 |
| transaction | 对所有 final pools、数据与关键输入 hash 固定后，只写一次的共同 test 评估 |

---

## 结语：如何正确理解这套实现

这套系统的核心不是“让某个搜索器生成看起来合理的公式”，而是把每个公式放进同一个受控链条：历史时点数据 → 受限 DSL → signal validity → frozen-pool marginal reward → validation snapshot selection → 共同 test universe → 带延迟和成本的组合评估。Random、GP、Base LLM 与 GRPO-style learner 的主要区别发生在 proposal distribution 如何形成和更新；候选的市场计算、pool admission 与最终 test transaction 则尽量共享实现。

阅读任何实验结论时，至少同时回答五个问题：有效预算是多少；reward/mask 的真实语义是什么；报告的是 terminal 还是 validation-selected pool；test 的逐池 complete-case 样本量是多少；配置字段是否真的被消费。只要其中一个答案来自“文件名看起来像”而不是源码和 artifact，这个结论就还不具备可复现性。
