# Qwen Counting Digit Drift Position Sweep

- Samples: 720 total, 331 correct, 389 wrong.
- Digits: [2, 3, 4, 5, 6, 7, 8, 9]; 30 samples per digit per dataset.
- Direction orientation: `to_digit`; larger score means stronger work toward the Arabic digit side.
- Vectors are LM-head/unembedding rows, matching the existing Qwen neuron-set builders.
- Layers: 0--35; windows 1--36.

## Best Windows

| Rank | Position | Condition | Window | AUC | Bal. acc. | Correct mean | Wrong mean |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | final_m1 | `per_en_to_digit:raw|left2sd` | 21-34 | 0.8488 | 0.7897 | 0.7117 | 4.6851 |
| 2 | final_m1 | `per_en_to_digit:raw|left2sd` | 22-34 | 0.8453 | 0.7916 | 0.8602 | 4.7011 |
| 3 | final_m1 | `per_en_to_digit:raw|left2sd` | 20-34 | 0.8453 | 0.7930 | 0.2989 | 4.1437 |
| 4 | final_m1 | `per_en_to_digit:raw|left3sd` | 21-34 | 0.8449 | 0.7817 | 0.0190 | 3.5401 |
| 5 | final_m1 | `per_en_to_digit:raw|left3sd` | 27-34 | 0.8436 | 0.7892 | 0.0155 | 3.3361 |
| 6 | final_m1 | `per_en_to_digit:raw|left3sd` | 22-34 | 0.8435 | 0.7804 | 0.0143 | 3.5230 |
| 7 | final_m1 | `per_en_to_digit:raw|left3sd` | 28-34 | 0.8428 | 0.7826 | 0.0221 | 3.3436 |
| 8 | final_m1 | `per_en_to_digit:raw|left3sd` | 26-34 | 0.8423 | 0.7821 | 0.0340 | 3.3433 |
| 9 | final_m1 | `per_en_to_digit:raw|left2sd` | 18-34 | 0.8411 | 0.7887 | -0.0116 | 3.7845 |
| 10 | final_m1 | `per_en_to_digit:raw|left2sd` | 19-34 | 0.8409 | 0.7887 | 0.1687 | 4.0247 |
| 11 | final_m0 | `per_en_to_digit:raw|left3sd` | 28-35 | 0.8339 | 0.7671 | 3.3659 | 8.5280 |
| 12 | final_m0 | `per_en_to_digit:raw|fixed_en_wmd_top400` | 25-33 | 0.8326 | 0.7699 | -2.4133 | -0.1615 |
| 13 | final_m0 | `per_en_to_digit:raw|fixed_en_wmd_top400` | 25-34 | 0.8326 | 0.7699 | -2.4133 | -0.1615 |
| 14 | final_m0 | `per_en_to_digit:raw|fixed_en_wmd_top400` | 25-35 | 0.8326 | 0.7699 | -2.4133 | -0.1615 |
| 15 | final_m0 | `per_en_to_digit:raw|fixed_en_wmd_top400` | 23-33 | 0.8324 | 0.7766 | -2.6725 | -0.4739 |
| 16 | final_m0 | `per_en_to_digit:raw|fixed_en_wmd_top400` | 23-34 | 0.8324 | 0.7766 | -2.6725 | -0.4739 |
| 17 | final_m0 | `per_zh_to_digit:raw|right2sd` | 23-28 | 0.8322 | 0.7659 | -0.7547 | 0.3188 |
| 18 | final_m0 | `per_en_to_digit:raw|left3sd` | 30-35 | 0.8318 | 0.7669 | 3.7081 | 8.7126 |
| 19 | final_m0 | `per_en_to_digit:raw|left3sd` | 29-35 | 0.8299 | 0.7630 | 3.4900 | 8.5001 |
| 20 | final_m0 | `per_en_to_digit:raw|left3sd` | 17-35 | 0.8295 | 0.7680 | 3.5880 | 8.3850 |
| 21 | final_m0 | `per_en_to_digit:raw|left3sd` | 31-35 | 0.8292 | 0.7618 | 3.6637 | 8.5751 |
| 22 | final_m0 | `per_zh_to_digit:raw|right2sd` | 24-28 | 0.8256 | 0.7672 | -0.6884 | 0.3211 |
| 23 | final_m1 | `per_en_to_digit:raw|fixed_zh_wmd_top1000` | 0-34 | 0.8234 | 0.7559 | 6.3198 | 9.2331 |
| 24 | final_m1 | `per_en_to_digit:raw|fixed_zh_wmd_top1000` | 1-34 | 0.8234 | 0.7559 | 6.3198 | 9.2331 |
| 25 | final_m1 | `per_en_to_digit:raw|fixed_zh_wmd_top1000` | 2-34 | 0.8234 | 0.7559 | 6.3198 | 9.2331 |
| 26 | final_m1 | `per_en_to_digit:raw|fixed_zh_wmd_top1000` | 3-34 | 0.8234 | 0.7559 | 6.3198 | 9.2331 |
| 27 | final_m1 | `per_en_to_digit:raw|fixed_zh_wmd_top1000` | 4-34 | 0.8234 | 0.7559 | 6.3198 | 9.2331 |
| 28 | final_m0 | `per_pooled_to_digit:raw|right2sd` | 24-31 | 0.8220 | 0.7569 | -0.0095 | 2.0272 |
| 29 | final_m0 | `per_zh_to_digit:raw|right2sd` | 22-28 | 0.8213 | 0.7545 | -0.6685 | 0.3422 |
| 30 | final_m5 | `per_pooled_to_digit:raw|left3sd` | 11-11 | 0.8196 | 0.7795 | -0.0128 | 0.0173 |
