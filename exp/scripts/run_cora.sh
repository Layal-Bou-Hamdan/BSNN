#!/bin/sh

python3 -m exp.run \
    --model=BayesBundleSheaf \
    --d=4 \
    --add_lp=1 \
    --add_hp=0 \
    --dataset=cora \
    --layers=5 \
    --hidden_channels=32 \
    --weight_decay=1e-7 \
    --sheaf_decay=1e-8 \
    --input_dropout=0.7 \
    --dropout=0.3 \
    --lr=0.02 \
    --epochs=1000 \
    --early_stopping=200 \
    --folds=10 \
    --orth=householder \
    --left_weights=True \
    --right_weights=True \
    --use_kl=True \
    --sheaf_use_deg=False \
    --permute_masks=False \
    --num_ensemble=3 \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --entity="${ENTITY}"
