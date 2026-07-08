# Qwen Arabic One-Token OCR Drift

- Samples: 120 one-token Arabic words rendered on white background.
- OCR success: 110 success, 10 failed.
- Direction: fixed FLORES Arabic-minus-English sentence direction.
- `norm0_raw`: raw FLORES direction; `norm1_unit`: unit-normalized FLORES direction.
- Positive signed work `P` means MLP work along Arabic-minus-English; drift score is `D=-P`.

## Best Windows

| Rank | Position | Condition | Orientation | Window | AUC | Bal. acc. | Success mean | Failed mean |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | final_m0 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 26-26 | 0.7745 | 0.8000 | -0.0012 | 0.0050 |
| 2 | final_m0 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 26-26 | 0.7745 | 0.8000 | -0.0042 | 0.0172 |
| 3 | final_m2 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 25-25 | 0.7509 | 0.7636 | 0.0015 | 0.0086 |
| 4 | final_m2 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 25-25 | 0.7509 | 0.7636 | 0.0052 | 0.0294 |
| 5 | final_m7 | `norm0_raw|real_gt3sd_layers18_32` | drift_D_minus_P | 28-28 | 0.7464 | 0.7455 | 0.0647 | 0.0713 |
| 6 | final_m7 | `norm0_raw|real_gt3sd_null_lte3sd_layers18_32` | drift_D_minus_P | 28-28 | 0.7464 | 0.7455 | 0.0647 | 0.0713 |
| 7 | final_m7 | `norm1_unit|real_gt3sd_layers18_32` | drift_D_minus_P | 28-28 | 0.7464 | 0.7455 | 0.2209 | 0.2436 |
| 8 | final_m7 | `norm1_unit|real_gt3sd_null_lte3sd_layers18_32` | drift_D_minus_P | 28-28 | 0.7464 | 0.7455 | 0.2209 | 0.2436 |
| 9 | final_m6 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | arabic_work_P | 20-26 | 0.7200 | 0.7500 | -0.1013 | -0.0927 |
| 10 | final_m6 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | arabic_work_P | 20-26 | 0.7200 | 0.7500 | -0.3457 | -0.3164 |
| 11 | final_m6 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | arabic_work_P | 19-26 | 0.7155 | 0.7227 | -0.0854 | -0.0770 |
| 12 | final_m6 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | arabic_work_P | 19-26 | 0.7155 | 0.7227 | -0.2916 | -0.2628 |
| 13 | final_m0 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 23-23 | 0.7109 | 0.7227 | -0.0002 | 0.0012 |
| 14 | final_m0 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 23-23 | 0.7109 | 0.7227 | -0.0007 | 0.0039 |
| 15 | final_m0 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 25-26 | 0.7073 | 0.7636 | 0.0299 | 0.0363 |
| 16 | final_m0 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 25-26 | 0.7073 | 0.7636 | 0.1021 | 0.1239 |
| 17 | final_m7 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | arabic_work_P | 25-25 | 0.7064 | 0.6909 | 0.0318 | 0.0336 |
| 18 | final_m7 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | arabic_work_P | 25-25 | 0.7064 | 0.6909 | 0.1086 | 0.1147 |
| 19 | final_m8 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 23-31 | 0.7055 | 0.7409 | 0.2113 | 0.2230 |
| 20 | final_m8 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 23-31 | 0.7055 | 0.7409 | 0.7215 | 0.7614 |
| 21 | final_m8 | `norm0_raw|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 27-31 | 0.7027 | 0.7318 | 0.2818 | 0.2929 |
| 22 | final_m8 | `norm1_unit|real_gt3sd_null_lte1sd_layers18_32` | drift_D_minus_P | 27-31 | 0.7027 | 0.7318 | 0.9622 | 1.0000 |
| 23 | final_m10 | `norm0_raw|real_gt3sd_layers18_32` | arabic_work_P | 31-32 | 0.7027 | 0.6818 | 0.4012 | 0.4321 |
| 24 | final_m10 | `norm0_raw|real_gt3sd_layers18_32` | arabic_work_P | 31-33 | 0.7027 | 0.6818 | 0.4012 | 0.4321 |
| 25 | final_m10 | `norm0_raw|real_gt3sd_layers18_32` | arabic_work_P | 31-34 | 0.7027 | 0.6818 | 0.4012 | 0.4321 |
| 26 | final_m10 | `norm0_raw|real_gt3sd_layers18_32` | arabic_work_P | 31-35 | 0.7027 | 0.6818 | 0.4012 | 0.4321 |
| 27 | final_m10 | `norm0_raw|real_gt3sd_null_lte3sd_layers18_32` | arabic_work_P | 31-32 | 0.7027 | 0.6864 | 0.4057 | 0.4362 |
| 28 | final_m10 | `norm0_raw|real_gt3sd_null_lte3sd_layers18_32` | arabic_work_P | 31-33 | 0.7027 | 0.6864 | 0.4057 | 0.4362 |
| 29 | final_m10 | `norm0_raw|real_gt3sd_null_lte3sd_layers18_32` | arabic_work_P | 31-34 | 0.7027 | 0.6864 | 0.4057 | 0.4362 |
| 30 | final_m10 | `norm0_raw|real_gt3sd_null_lte3sd_layers18_32` | arabic_work_P | 31-35 | 0.7027 | 0.6864 | 0.4057 | 0.4362 |
