# QuantEvolver / AlphaSAGE 五窗口对齐

两个 baseline 现在使用与主实验相同的 `recent_alpha_v1` 外层协议：测试年为 2021–2025；每个测试年重新初始化并独立训练；搜索期为此前两年，校准期为随后半年，测试期为下一自然年。搜索结束后冻结终态因子池，校准期只拟合一次 ridge 权重，测试年不再改公式或权重。年度测试结果由现有 rolling report 汇总为方法级五年评估。

每个窗口的固定预算均为 100 个记账步骤、每步 8 个候选，即 800 个因子配额。QuantEvolver 每个 optimizer step 仍生成 8 个 completion。AlphaSAGE 的原始训练接口逐条生成候选，因此适配层将 800 解释为 100×8 个 valid-unique 因子配额；它原有的 GFlowNet optimizer 更新频率保持 64，不把记账步骤伪装成算法更新。

没有替换 baseline reward：QuantEvolver 继续使用 `qe_native` DiCo RankIC 及原有 shaping；AlphaSAGE 继续使用其绝对 IC、diversity、SSL 与 novelty reward。变化只涉及外层日期、预算、终态池冻结、校准/测试和跨五年汇总。QuantEvolver 的 early/middle/late/full 任务结构按每个两年搜索窗等比例重映射，reward 公式和阈值不变。

运行入口：

```bash
ours/scripts/run_quantevolver_recent_alpha_rolling.sh
baseline/AlphaSAGE/run_recent_alpha_rolling.sh
```

对应配置分别为 `configs/experiment/recent_alpha_quantevolver_rolling.yaml` 和 `configs/experiment/recent_alpha_alphasage_rolling.yaml`。两者使用不同虚拟环境，应分别启动。默认各有 5 年 × 3 seed = 15 个训练/评估单元，且每个年度窗口都会创建新的模型、优化器、pool 和搜索历史。

旧的 `quantevolver_fair_250` 与 `alphasage_fair_1000` 文件仅保留历史复现实验，不再是当前对齐协议的运行入口。
