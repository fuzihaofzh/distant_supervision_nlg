#!/bin/bash
##SBATCH --nodes=1    #Use only 1 computing node
##SBATCH --gres=gpu:1   # use only 1 GPU node
##SBATCH --mem=0               # Request the full memory of the node
##SBATCH --time=24:00:00           #Timeout : 3 minute
##SBATCH --partition=infofil01

DATASET=$1
MODE=$2
VERSION=$3

if [ -z "$VERSION" ]
then
    VERSION=''
else
    VERSION=__"$VERSION"
fi
EXPNAME="$DATASET"__"$MODE""$VERSION"

LR=5e-4 #5e-4

#alias python="~/ProgramFiles/miniconda3/envs/fairseq/bin/python"
which python

echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES


export SRC=src
export TGT=tgt
export MTK=30000 
export EPOCH=103
export VALIDINTERVAL=20

PRETRAINEXP=$DATASET"__endorsement,pretrain"   
PRETRAINMODEL=checkpoint_last.pt 



if [[ "$MODE" = *endorsement* ]]; then
    if [[ "$MODE" = *pretrain* ]]; then
        export EPOCH=301
        export VALIDINTERVAL=1000
    else
        export LOADPAR=" --restore-file output/models/$DATASET""__endorsement,pretrain/checkpoint_last.pt  --reset-dataloader "
    fi
fi

if [[ "$MODE" = "attn_endorse" ]]; then
    export VALIDINTERVAL=5
    export EPOCH=20
fi




echo "EXPNAME: " $EXPNAME

mkdir -p output/models/$EXPNAME


echo "SRC: " $SRC ", TGT: " $TGT ", MTK:" $MTK



PYTHONIOENCODING=utf8 python3 ./fairseq/train.py output/data-bin/$DATASET  --optimizer adam --clip-norm 1.0 --lr $LR -s $SRC -t $TGT --label-smoothing 0.1 --dropout 0.3 --max-tokens $MTK --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001  --max-update 500000 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --save-dir output/models/$EXPNAME --tensorboard-logdir ./output/log/runs/$EXPNAME --results-dir ./output/eval/$EXPNAME --max-epoch $EPOCH --save-interval 20 --validate-interval $VALIDINTERVAL  --arch distant_transformer --criterion distant_transformer_loss --user-mode $MODE --fp16 --task nlg --reset-optimizer $LOADPAR
