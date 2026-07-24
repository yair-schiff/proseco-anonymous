#!/bin/bash

source ./setup_env.sh || exit
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

benchmark=${BENCHMARK:-human-eval}
gen_length=${LENGTH:-1024}
block_length=${BLOCK_LENGTH:-32}
steps=${STEPS:-1024}
apply_corrector_every_n_steps=${APPLY_CORRECTOR_EVERY_N_STEPS:-2}
max_corrector_steps_per_loop=${MAX_CORRECTOR_STEPS_PER_LOOP:-4}
early_eos_stopping=${EARLY_EOS_STOPPING:-True}
threshold=${THRESHOLD:-None}
model_path=${MODEL_PATH:?MODEL_PATH must be set}
tokenizer_path=${TOKENIZER_PATH:-"${model_path}"}
prompt_config=${PROMPT_CONFIG:-"${PWD}/llada/prompt_configs/code.yaml"}
num_gpus=${NUM_GPUS:-8}
BASE_SAVE_DIR=${BASE_SAVE_DIR:-"${PWD}/llada/outputs"}

save_dir="${BASE_SAVE_DIR}/${benchmark}/length-${gen_length}--block_length-${block_length}--steps-${steps}--early_eos_stopping-${early_eos_stopping}--apply_corrector_every_n_steps-${apply_corrector_every_n_steps}--max_corrector_steps_per_loop-${max_corrector_steps_per_loop}"

accelerate launch --num_processes "${num_gpus}" llada/eval_llada.py \
  --benchmark "${benchmark}" \
  --model_path "${model_path}" \
  --tokenizer_path "${tokenizer_path}" \
  --gen_length "${gen_length}" \
  --block_length "${block_length}" \
  --steps "${steps}" \
  --threshold "${threshold}" \
  --apply_corrector_every_n_steps "${apply_corrector_every_n_steps}" \
  --max_corrector_steps_per_loop "${max_corrector_steps_per_loop}" \
  --early_eos_stopping "${early_eos_stopping}" \
  --output_dir "${save_dir}" \
  --prompt_config "${prompt_config}"
