#!/usr/bin/env bash
set -euo pipefail

bash configs/counting_gemma3_4b.sh
bash configs/counting_gemma3_12b.sh
bash configs/counting_qwen2_5_vl_3b.sh
bash configs/counting_qwen3_vl_8b.sh
bash configs/run_all_ccocr.sh
