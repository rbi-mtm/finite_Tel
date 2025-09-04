#!/bin/bash

if [ -z "${1}" ]; then
echo "Must set number of CUDA device. Use nvidia-smi to get numbers of available devices!"
exit -1
fi

export CUDA_VISIBLE_DEVICES="$1"
echo $CUDA_VISIBLE_DEVICES
num_threads=1
export OMP_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads

PYPATH=~/miniconda3/envs/mace_env3/bin/

$PYPATH/mace_run_train \
    --name="1H_NbSe2_model_6" \
    --train_file="./1H_NbSe2_frames_train_MACE.xyz" \
    --valid_fraction=0.05 \
    --test_file="./1H_NbSe2_frames_test_MACE.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='average' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o + 128x2e' \
    --r_max=8.0 \
    --batch_size=5 \
    --max_num_epochs=2000 \
    --swa \
    --start_swa=1600 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --save_cpu

echo "Done."
