# ProSeCo 

This repository contains code for reproducing the experiments in an anonymous paper submission.

## Code Organization

<a name="code-organization"></a>

1. ```main.py```: Routines for training (language models and classifiers)
2. ```noise_schedule.py```: Noise schedules
3. ```diffusion.py```: Forward/reverse diffusion
    - Absorbing state / uniform noise diffusion
    - AR
4. ```dataloader.py```: Dataloaders
5. ```utils.py```: LR scheduler, logging, `fsspec` handling
6. ```models/```: Denoising network architectures.
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules
8. ```scripts/```: Shell scripts for training/evaluation
9. ```guidance_eval/```: Guidance evaluation scripts
10. ```llada/```: Code to reproduce evaluation of LLaDA SFT models

### ProSeCo Training

<a name="training"></a>

To enable ProSeCo training, set the `corrector_training` flag in
[`config.yaml`](configs/config.yaml) to `True`.

Additional parameters that can be tuned include the following:

```yaml
corrector_training: True
use_weighted_corrector_loss: True
use_model_outputs_as_corrector_input: False
use_argmax_for_corrector: True
corrector_training_start_step: 0
mdlm_loss_weight: 1.0
corrector_loss_weight: 1.0
corrector_loss_errors_upweighted: False
```

### ProSeCo Sampling

<a name="inference"></a>

Below we detail the parameters one can use when applying corrector steps during
inference.
These parameters can be found under `sampling` in [`config.yaml`](configs/config.yaml):

```yaml
corrector_prior_is_argmax: True
corrector_sampling: 'argmax'
corrector_every_n_steps: 1
corrector_steps: 0
corrector_start_iter: 0
corrector_top_k: 0
```

### LLaDA experiments

<a name="llada"></a>

The repository also provides code for reproducing evaluations with a LLaDA-SFT model in the
[llada](./llada) directory.
See the [README](./llada/README.md) file there for more details.

## Getting started in this repository

<a name="getting-started"></a>

To get started, create a conda environment containing the required dependencies.

```bash
./create_env.sh
conda activate anonymous-code
```

Set local resource locations when running the corresponding experiments:

```bash
export DATA_CACHE_DIR=/path/to/cache
export QM9_DATASET_PATH=/path/to/qm9
export QM9_TOKENIZER_PATH=/path/to/qm9-tokenizer
export MODEL_PATH=/path/to/model
export TOKENIZER_PATH=/path/to/tokenizer
```

Create the following directories to store saved models and slurm logs:

```bash
mkdir outputs
mkdir watch_folder
```

## Reproducing Experiments

<a name="reproducing-experiments"></a>

Throughout, the main entry point for running experiments is the [`main.py`](./main.py) script.
We also provide sample `slurm` scripts for launching pre-training and evaluation experiments in the [`scripts/`](./scripts) directory.
