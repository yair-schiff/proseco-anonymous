#!/bin/bash

eval "$(conda shell.bash hook)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "anonymous-code" ]]; then
  conda activate anonymous-code
fi

export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"
