#!/bin/bash

# By setting --param_searching to 0, we can use the best \lambda for
# different SSL basic tasks. See details in src/configs.py

python ./src/train_ssl.py \
    --datapath data// \
    --seed 10 \
    --dataset cora \
    --type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 0 \
    --hidden 128 \
    --epoch 200 \
    --lr 0.01 \
    --weight_decay 5e-04 \
    --early_stopping 200 \
    --sampling_percent 1 \
    --dropout 0.5 \
    --normalization AugNormAdj --task_type semi \
    --ssl AttributeMask \
    --lambda_ 10 \
    --train_size 0 \
    --param_searching 1 \
     \

