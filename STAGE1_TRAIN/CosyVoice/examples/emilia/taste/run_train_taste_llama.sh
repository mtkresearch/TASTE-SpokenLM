#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
set -e
# please run this script with `bash run_train_taste.sh`, or the relative fpath will be incorrect.

source ../../../../path.sh

stage=3
stop_stage=3

pretrained_model_dir=$RTSLM_STORAGE_DIR/pretrained_models/CosyVoice-300M
DATA_ROOT=$RTSLM_STORAGE_DIR/data/ # change to your own
export CUDA_VISIBLE_DEVICES="2,3" # change for your own need

# prepare data for example usage
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # prepare data list (for example usage)
    ls -a $DATA_ROOT/train/*.arrow > data/train.data.list
    ls -a $DATA_ROOT/dev/*.arrow > data/dev.data.list
    ls -a $DATA_ROOT/test/*.arrow > data/test.data.list
fi

# train text only baseline
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    EXP_NAME="text-only_baseline"
    conf_fpath="./conf/$EXP_NAME.yaml"
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    _OMP_NUM_THREADS=$num_gpus
    job_id=1986
    dist_backend="nccl"
    data_loader_num_workers=2
    prefetch=8
    train_engine=torch_ddp
    model=llm
    echo "Conduct training of the text-only baseline."
    echo "Training info: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; num_gpus=$num_gpus; dataloader_num_workers=$data_loader_num_workers"
    # start training modules
    mkdir -p ./exp/$model/$train_engine/$EXP_NAME
    cp $0 ./exp/$model/$train_engine/$EXP_NAME
    NCCL_DEBUG=TRACE OMP_NUM_THREADS=$_OMP_NUM_THREADS torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 12345 \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
        $RTSLM_WORK_DIR/CosyVoice/cosyvoice/bin/train.py \
        --train_engine $train_engine \
        --config $conf_fpath \
        --train_data ./data/train.data.list \
        --cv_data ./data/dev.data.list \
        --model $model \
        --model_dir ./exp/$model/$train_engine/$EXP_NAME \
        --tensorboard_dir ./tensorboard/$model/$train_engine/$EXP_NAME \
        --ddp.dist_backend $dist_backend \
        --num_workers ${data_loader_num_workers} \
        --prefetch ${prefetch} \
        --pin_memory \
        --timeout 30 \
        --deepspeed_config ./conf/customized_ds.json \
        --deepspeed.save_states model+optimizer 
        # --checkpoint $ckpt_fpath # for resuming
fi


# train TASTE w/o quantizer
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    EXP_NAME="taste_no_vq_llama"
    # ckpt_fpath="./exp/llm/torch_ddp/text-only_baseline/checkpoint_best.pt" # initialize from text-only baseline for efficiency
    ckpt_fpath="./exp/text-only_baseline/checkpoint_llama.pt"
    conf_fpath="./conf/$EXP_NAME.yaml"

    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    _OMP_NUM_THREADS=$num_gpus
    job_id=1986
    dist_backend="nccl"
    data_loader_num_workers=2
    prefetch=8
    train_engine=torch_ddp
    model=llm
    echo "Conduct training of the taste without vq."
    echo "Training info: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; num_gpus=$num_gpus; dataloader_num_workers=$data_loader_num_workers"
    # start training modules
    mkdir -p ./exp/$model/$train_engine/$EXP_NAME
    cp $0 ./exp/$model/$train_engine/$EXP_NAME
    NCCL_DEBUG=DEBUG OMP_NUM_THREADS=$_OMP_NUM_THREADS torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
        $RTSLM_WORK_DIR/CosyVoice/cosyvoice/bin/train.py \
        --train_engine $train_engine \
        --config $conf_fpath \
        --train_data ./data/train.data.list \
        --cv_data ./data/dev.data.list \
        --model $model \
        --model_dir ./exp/$model/$train_engine/$EXP_NAME \
        --tensorboard_dir ./tensorboard/$model/$train_engine/$EXP_NAME \
        --ddp.dist_backend $dist_backend \
        --num_workers ${data_loader_num_workers} \
        --prefetch ${prefetch} \
        --pin_memory \
        --timeout 30 \
        --deepspeed_config ./conf/customized_ds.json \
        --deepspeed.save_states model+optimizer \
        --checkpoint $ckpt_fpath
fi


# train TASTE with rvq_quantizer
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    EXP_NAME="taste"
    ckpt_fpath="./exp/llm/torch_ddp/taste_no_vq/checkpoint_best.pt" # initialize from text-only baseline for efficiency
    conf_fpath="./conf/$EXP_NAME.yaml"

    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    _OMP_NUM_THREADS=$num_gpus
    job_id=1986
    dist_backend="nccl"
    data_loader_num_workers=2
    prefetch=8
    train_engine=torch_ddp
    model=llm
    echo "Conduct training of the taste with rvq."
    echo "Training info: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; num_gpus=$num_gpus; dataloader_num_workers=$data_loader_num_workers"
    if [ $train_engine == 'deepspeed' ]; then
      echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
    fi
    # start training modules
    mkdir -p ./exp/$model/$train_engine/$EXP_NAME
    cp $0 ./exp/$model/$train_engine/$EXP_NAME
    NCCL_DEBUG=DEBUG OMP_NUM_THREADS=$_OMP_NUM_THREADS torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
        $RTSLM_WORK_DIR/CosyVoice/cosyvoice/bin/train.py \
        --train_engine $train_engine \
        --config $conf_fpath \
        --train_data ./data/train.data.list \
        --cv_data ./data/dev.data.list \
        --model $model \
        --model_dir ./exp/$model/$train_engine/$EXP_NAME \
        --tensorboard_dir ./tensorboard/$model/$train_engine/$EXP_NAME \
        --ddp.dist_backend $dist_backend \
        --num_workers ${data_loader_num_workers} \
        --prefetch ${prefetch} \
        --pin_memory \
        --timeout 30 \
        --deepspeed_config ./conf/customized_ds.json \
        --deepspeed.save_states model+optimizer \
        --checkpoint $ckpt_fpath
fi