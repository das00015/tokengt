#!/usr/bin/env bash

ulimit -c unlimited

fairseq-train \
--user-dir ../tokengt \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name upfd \
--dataset-source ogb \
--task graph_prediction \
--criterion l1_loss \
--arch tokengt_base \
--orf-node-id \
--orf-node-id-dim 64 \
--stochastic-depth \
--prenorm \
--num-classes 2 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 32 \
--data-buffer-size 20 \
--save-dir ./ckpts/upfd \
--tensorboard-logdir ./tb/upfd \
--no-epoch-checkpoints

#--num-classes 1 \
#--fp16 \
#--batch-size 128 \
#--dataset-name pcqm4mv2 \