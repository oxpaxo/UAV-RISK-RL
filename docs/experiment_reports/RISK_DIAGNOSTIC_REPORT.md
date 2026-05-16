# Risk Signal Sudden-Turn 诊断报告

## 1. 诊断目的

本报告用于分析 `risk_full_rbar` 在 `eval_sudden_turn` 中 reaction_time 显著慢于 `attention_full` 的原因。

---

## 2. 已检查文件和 checkpoint

- risk checkpoints:
  `checkpoints/risk_full_rbar_s0.zip`
  `checkpoints/risk_full_rbar_s1.zip`
  `checkpoints/risk_full_rbar_s2.zip`
- attention checkpoints:
  `checkpoints/attention_full_s0.zip`
  `checkpoints/attention_full_s1.zip`
  `checkpoints/attention_full_s2.zip`
- env file:
  `envs/dynamic_obstacle_env.py`
- extractor file:
  `policies/obstacle_set_extractor.py`
- eval file:
  `eval.py`

---

## 3. 诊断脚本

新增文件：

- `scripts/diagnose_sudden_turn.py`
- `scripts/plot_risk_diagnostics.py`
- `scripts/summarize_risk_diagnostics.py`
- `scripts/run_risk_diagnostics.sh`
- `scripts/run_risk_eval_sensitivity.sh`

---

## 4. 主要诊断结果

| Method | Seed | risk@turn | max risk after turn | risk rise 0.5 ↓ | sigma trace@turn | max R_bar | w_risk max | reaction ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| risk_full_rbar | 0 | 0.0667 | 0.0757 | 0.0000 | 1.0805 | 0.4280 | 0.9488 | 0.2000 |
| risk_full_rbar | 1 | 0.0372 | 0.0386 | NaN | 1.2142 | 0.4006 | 0.9870 | 0.3500 |
| risk_full_rbar | 2 | 0.0266 | 0.0355 | NaN | 1.0545 | 0.3717 | 0.9749 | 0.5167 |
| attention_full | 0 | 0.0196 | 0.0261 | NaN | 1.0037 | 0.4093 | 0.9774 | 0.2000 |
| attention_full | 1 | 0.0007 | 0.0083 | NaN | 1.1405 | 0.2799 | 0.8222 | 0.2000 |
| attention_full | 2 | 0.0237 | 0.0478 | 0.2000 | 0.8848 | 0.3794 | 0.9320 | 0.6929 |

说明：

- `risk@turn` 和 `max risk after turn` 在所有 seed 上都偏低，绝大多数 episode 连 `0.3` 都到不了。
- `risk rise 0.5` 大量为 `NaN`，意味着 turning obstacle 的 risk 在突变后根本没有稳定升到有效威胁区间。
- `w_risk max` 和 `turning_obstacle_top1_rate_after_turn` 普遍较高，说明纯 risk 排序本身没有明显失焦。
- `R_bar` 上限普遍在 `0.28 - 0.43`，明显偏低；对 `risk_full_rbar` 而言，这会持续压低风险上下文幅度。

---

## 5. 曲线观察

典型 episode 的曲线图位于：

- `results/risk_diagnostics/plots/`

观察结论：

1. `risk_turn` 在 `turn_step` 后通常没有快速上升，很多轨迹在整个突变窗口内仍维持在很低水平。
2. `Sigma trace` 确实会在突变后上升，但上升幅度有限，而且到 2x baseline 往往需要约 `0.7s - 1.5s`。
3. `R_sum / R_bar` 整体偏低，`R_bar` 在 turn 时刻几乎始终低于 `0.1`，峰值通常也不足 `0.45`。
4. `w_risk_turn` 往往能把 turning obstacle 提到前列，说明“关注谁”不是主问题。
5. `deviation_lateral` 通常不晚于 risk 上升；也就是说，当前慢反应并不主要来自“risk 已升高但策略完全不用”，而更像是“risk 本身升不起来”。

---

## 6. 自动诊断结论

综合自动规则与人工复核，当前最可信的判断是：

- risk_i 上升慢；
- Sigma 上升偏慢；
- R_bar 压制过强；
- 未发现明显 risk 排序失败；
- “risk 已明显升高但策略不反应”不是当前主矛盾。

更具体地说：

1. turning obstacle 的 `risk_turn` 在大多数 episode 中始终过低，说明主问题首先是 risk 信号动态范围不足。
2. `Sigma trace` 会涨，但涨得不够快，也不够大，导致 multi-step risk 在突变后不能快速越过 `0.3/0.5` 阈值。
3. 即使 `R_sum` 有所上升，`R_bar = tanh(R_sum / r_ref)` 仍常年落在很低区间，对 `risk_full_rbar` 会形成额外缩放压制。
4. `turning_obstacle_top1_rate_after_turn` 很高，说明解析 risk 排序大体是对的，不是“关注错障碍物”导致的问题。

---

## 7. Eval-only sensitivity 结果

| Config | risk rise 0.5 ↓ | max risk ↑ | max R_bar ↑ | reaction ↓ | 备注 |
|---|---:|---:|---:|---:|---|
| default | 0.0000 | 0.0757 | 0.4280 | 0.2000 | 93.3% episode 仍无法升到 0.5 |
| sigma_min_0p10 | 0.0000 | 0.0757 | 0.4280 | 0.2000 | 单独提高 `sigma_min` 几乎没改善 |
| lambda_0p15 | 0.0000 | 0.0908 | 0.4284 | 0.2100 | risk 峰值略升，但 `sigma_rise_time_2x` 反而更慢 |
| lambda_0p20 | 0.0000 | 0.1020 | 0.4292 | 0.2105 | risk 峰值继续略升，但仍然很低 |
| Rgate_8 | 0.0000 | 0.0786 | 0.4725 | 0.2000 | `R_bar` 峰值提升最明显，说明 gate 确实在压制 |
| no_rbar | NaN | 0.0083 | 0.1582 | 0.2000 | 该 eval-only 结果不支持“只去 R_bar 就能救”；信号本体仍太弱 |
| sigma0p10_Rgate8_no_rbar | NaN | 0.0096 | 0.1768 | 0.2000 | 组合修正仍没让 risk 动态区间显著起来 |

说明：该表只用于信号诊断，不作为正式性能结论。

关键信息：

- 仅增大 `R_gate` 的改善最明确，说明 distance gate 在压制 risk 动态范围。
- 单独提高 `lambda_ewma` 会把峰值抬高一点，但远不足以解决问题。
- 单独提高 `sigma_min` 几乎没有效果，说明当前主要瓶颈不只是最小方差下限。
- 仅关掉 `R_bar` 并没有让 signal 自己变强，说明 `R_bar` 是“第二层压制”，不是唯一根因。

---

## 8. 下一步建议

### 如果 Sigma 上升慢

建议：

```text
lambda_ewma: 0.10 → 0.15
必要时再试 0.20，但优先 0.15
```

当前数据不支持把 `sigma_min` 单独提到 `0.10` 作为主修正，因为它几乎没有改善 `risk_max_after_turn`。

### 如果 R_bar 压制明显

建议：

```text
先做 risk_full_no_rbar 的短训验证
或使用 R_bar floor:
c = max(R_bar, 0.3) * c
```

但要注意：这一步只能作为次级修正，因为 base risk 值本身仍偏低。

### 如果 distance gate 压制明显

建议：

```text
R_gate: 5.0 → 8.0
```

当前 sensitivity 中，`Rgate_8` 是最有直接证据的修正方向。

### 如果 risk 已经升高但策略不反应

当前证据不强，不建议把它作为第一优先级。

---

## 9. 最终判断

`risk_full_rbar` 的 sudden-turn 慢反应，最主要问题不是排序错了，也不是策略完全忽视风险，而是 turning obstacle 的 `risk_i` 在突变后普遍升得太慢、太低。其根因更接近“risk signal 本体动态范围不足”，其中 `distance gate` 压制和 `EWMA` 响应偏慢是第一层问题，`R_bar` 低幅度缩放是第二层问题。  

因此，当前不建议继续直接投入“纯 risk weighting 原样放大训练”。更合理的主线是：

1. 先做短训验证：
   `R_gate=8.0`
   `lambda_ewma=0.15`
   `risk_no_rbar` 或 `R_bar floor`
2. 如果这些短训仍不能明显改善 sudden-turn 的 `risk_rise_time_0p5` 与 `reaction_time`，则主线应转向：

```text
risk-residual attention:
w_final = (1 - alpha) * w_risk + alpha * w_attention
alpha = 0.4
```

这比继续押注纯 risk weighting 更有希望。
