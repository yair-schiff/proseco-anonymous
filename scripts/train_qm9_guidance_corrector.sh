#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64000
#SBATCH -t 960:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --open-mode=append
#SBATCH --requeue

cd ../ || exit
source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

if [ -z "${PROP}" ]; then
  echo "PROP is not set"
  exit 1
fi
RUN_NAME="${RUN_NAME:-corrector_${PROP}}"

DIFFUSION="absorbing_state"
PARAMETERIZATION="d3pm"
SUBS_MASKING=True
T=0
TIME_COND=False
ZERO_RECON_LOSS=False
USE_MODEL_OUTPUTS_AS_CORRECTOR_INPUT=True
USE_MODEL_ARGMAX_OUTPUT=True
USE_WEIGHTED_CORRECTOR_LOSS=True
CORRECTOR_TRAINING_START_STEP=0
MDLM_LOSS_WEIGHT=1.0
CORRECTOR_LOSS_WEIGHT=1.0
SAMPLING_EPS_TRAINING=1e-1

srun python -u -m main \
  corrector_training=True \
  use_model_outputs_as_corrector_input=${USE_MODEL_OUTPUTS_AS_CORRECTOR_INPUT} \
  use_model_argmax_output=${USE_MODEL_ARGMAX_OUTPUT} \
  corrector_training_start_step=${CORRECTOR_TRAINING_START_STEP} \
  use_weighted_corrector_loss=${USE_WEIGHTED_CORRECTOR_LOSS} \
  mdlm_loss_weight=${MDLM_LOSS_WEIGHT} \
  corrector_loss_weight=${CORRECTOR_LOSS_WEIGHT} \
  diffusion="${DIFFUSION}" \
  parameterization="${PARAMETERIZATION}" \
  subs_masking=${SUBS_MASKING} \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=qm9 \
  data.label_col=${PROP} \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  eval.generate_samples=False \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=2048 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  backbone="dit" \
  model=small \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  training.sampling_eps_training=${SAMPLING_EPS_TRAINING} \
  training.guidance.cond_dropout=0.1 \
  callbacks='[checkpoint_every_n_steps,checkpoint_monitor,checkpoint_monitor_corrector,learning_rate_monitor]' \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  callbacks.checkpoint_monitor_corrector.start_monitor_step=${CORRECTOR_TRAINING_START_STEP} \
  training.compute_loss_on_pad_tokens=True \
  trainer.max_steps=25_000 \
  trainer.val_check_interval=1.0 \
  wandb=null \
  hydra.run.dir="${PWD}/outputs/qm9/${RUN_NAME}"
