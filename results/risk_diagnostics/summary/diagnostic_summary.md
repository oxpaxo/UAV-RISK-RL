# Risk Diagnostic Summary

| Method | Seed | Risk0.5 Rise | Sigma2x Rise | R_bar@turn | Top1 After | Reaction |
|---|---:|---:|---:|---:|---:|---:|
| attention_full | 0 | NaN | 1.3286 | 0.0233 | 0.9667 | 0.2000 |
| attention_full | 1 | NaN | 1.4545 | 0.0007 | 0.9667 | 0.2000 |
| attention_full | 2 | 0.2000 | 0.9067 | 0.0318 | 0.9333 | 0.6929 |
| risk_full_rbar | 0 | 0.0000 | 1.0667 | 0.0627 | 0.9667 | 0.2000 |
| risk_full_rbar | 1 | NaN | 0.8000 | 0.0367 | 1.0000 | 0.3500 |
| risk_full_rbar | 2 | NaN | 0.7067 | 0.0332 | 1.0000 | 0.5167 |

## Auto Diagnosis

### attention_full s0
- 风险信号对突变响应偏慢。优先检查 sigma_min、lambda_ewma、R_gate。
- EWMA 不确定性估计响应偏慢。建议尝试 sigma_min=0.10 或 lambda_ewma=0.15/0.20。
- R_bar 对 risk context 可能存在过强压制。建议尝试 risk_full_no_rbar 或 R_bar floor。

### attention_full s1
- 风险信号对突变响应偏慢。优先检查 sigma_min、lambda_ewma、R_gate。
- EWMA 不确定性估计响应偏慢。建议尝试 sigma_min=0.10 或 lambda_ewma=0.15/0.20。
- R_bar 对 risk context 可能存在过强压制。建议尝试 risk_full_no_rbar 或 R_bar floor。

### attention_full s2
- 风险信号对突变响应偏慢。优先检查 sigma_min、lambda_ewma、R_gate。
- EWMA 不确定性估计响应偏慢。建议尝试 sigma_min=0.10 或 lambda_ewma=0.15/0.20。
- R_bar 对 risk context 可能存在过强压制。建议尝试 risk_full_no_rbar 或 R_bar floor。

### risk_full_rbar s0
- 风险信号对突变响应偏慢。优先检查 sigma_min、lambda_ewma、R_gate。
- EWMA 不确定性估计响应偏慢。建议尝试 sigma_min=0.10 或 lambda_ewma=0.15/0.20。
- R_bar 对 risk context 可能存在过强压制。建议尝试 risk_full_no_rbar 或 R_bar floor。

### risk_full_rbar s1
- 风险信号对突变响应偏慢。优先检查 sigma_min、lambda_ewma、R_gate。
- EWMA 不确定性估计响应偏慢。建议尝试 sigma_min=0.10 或 lambda_ewma=0.15/0.20。
- R_bar 对 risk context 可能存在过强压制。建议尝试 risk_full_no_rbar 或 R_bar floor。

### risk_full_rbar s2
- 风险信号对突变响应偏慢。优先检查 sigma_min、lambda_ewma、R_gate。
- EWMA 不确定性估计响应偏慢。建议尝试 sigma_min=0.10 或 lambda_ewma=0.15/0.20。
- R_bar 对 risk context 可能存在过强压制。建议尝试 risk_full_no_rbar 或 R_bar floor。
