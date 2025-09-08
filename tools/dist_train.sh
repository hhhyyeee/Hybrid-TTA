#!/usr/bin/env bash

CONFIG=local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py
GPUS=1
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/our.py $CONFIG --launcher pytorch ${@:3}
