# 常驻 Verl GRPO 集成

正式 GRPO 路径由 `stage_coordinator.py`、`online_dataset.py`、
`verl_reward_function.py` 和 `verl_trainer.py` 组成。旧的自定义
REINFORCE controller、reward bridge 和未接线 adapter 已删除。

一个 GRPO cell 只创建一次 Ray、Verl workers、LoRA actor/reference 和
vLLM。在线 dataset 每个 optimizer update 后调用 `on_batch_end()`：
先提交已产生的 pending domain transition，再用最新 pool 生成下一条
prompt。同组相同表达式只评估一个确定性代表，其余 completion 复用市场
诊断但得到重复惩罚，不再复用正 reward；历史或 pool 内重复也按重复处理。

Reward 使用严格同 support 的 add-only 增量：

```text
J(P + candidate; M_(P+c)) - J(P; M_(P+c))
```

pool 满时在 21 因子 ridge 系统计算条件 saliency
`w_j² / (A⁻¹)_jj`，每个候选只检查最低三个删除项，全组最多三个方案
接受自然-support正式复核。训练 shaping 与最终是否入池分开计算。

当前 v9 使用 [expanding OOF 协议](rolling_oof.md)：LCB 默认 0.5，ridge
默认 0.01；合法但 add-only 增量不超过 min_delta 的 reward 为 0，合格
正增量按正候选中位数做 softsign 缩放，非法/重复为 -1。实际 Verl 配置
固定 `norm_adv_by_std_in_grpo=false`，advantage 只减组均值，不做第二次
标准差归一化，不引入双 reward 或额外混合参数。

Checkpoint 默认每 50 个 optimizer updates 保存，最后一步强制保存，
仅保存 LoRA model state，同时保留 LoRA optimizer、scheduler、RNG 和
同一步 domain snapshot；最多保留两份。恢复只接受
schema 9 / `fixed-universe-expanding-positive-softsign-v9`，旧 checkpoint 明确拒绝。
模型大文件只做一次 SHA 验证，attestation 缓存在
`/data/sunyuxiang/rl_alpha/cache/model_attestations`。

正式配置还限制 Ray 为 8 CPUs、8 GiB object store。GPU smoke 必须证明
连续 updates 间 Ray/vLLM PID 不变、actor 参数变化，并存在 PPO clip、
KL、entropy 指标；在这些证据通过前不得启动 3000-budget 长程实验。
