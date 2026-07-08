# Gemma Counting Digit-Minus-Word Token Position Sweep

- Samples: 720 total, 295 correct, 425 wrong.
- Position modes: final_newline, assistant_role, assistant_start, user_end, user_final_content, question_mark.
- Conditions: raw_top200, unit_top200.
- Layers: 0--33; max window width 34.
- Direction: `d_k = E(k) - E(word(k))`; positive score is digit-direction work.

## Best Global Windows

| Position | Condition | Window | AUC | Correct mean | Wrong mean | Bal. acc. |
|---|---|---|---:|---:|---:|---:|
| final_newline | raw_top200 | 0-21 | 0.6921 | -0.0651 | -0.0360 | 0.6513 |
| final_newline | unit_top200 | 0-21 | 0.6909 | -0.0527 | -0.0293 | 0.6508 |
| assistant_role | raw_top200 | 20-20 | 0.7172 | -0.0142 | -0.0122 | 0.6740 |
| assistant_role | unit_top200 | 20-20 | 0.7101 | -0.0116 | -0.0101 | 0.6504 |
| assistant_start | raw_top200 | 21-21 | 0.7210 | -0.0390 | -0.0307 | 0.6790 |
| assistant_start | unit_top200 | 21-21 | 0.7165 | -0.0319 | -0.0253 | 0.6697 |
| user_end | raw_top200 | 0-18 | 0.7257 | 0.0016 | 0.0023 | 0.6688 |
| user_end | unit_top200 | 0-18 | 0.7325 | 0.0013 | 0.0019 | 0.6733 |
| user_final_content | raw_top200 | 23-27 | 0.6629 | -0.8317 | -0.7496 | 0.6522 |
| user_final_content | unit_top200 | 23-27 | 0.6479 | -0.6832 | -0.6216 | 0.6448 |
| question_mark | raw_top200 | 21-21 | 0.7641 | -0.0222 | -0.0073 | 0.7150 |
| question_mark | unit_top200 | 21-21 | 0.7638 | -0.0181 | -0.0059 | 0.7138 |
