#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# 删除 PORT 生成逻辑（torchrun 会自动处理）

# 使用 torchrun 启动
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NGPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py \
    --launcher pytorch \
    ${PY_ARGS}