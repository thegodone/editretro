#!/bin/bash

databin='/home/ubuntu/editretro/datasets/USPTO_50K/aug20/data-bin/'

exp_n=inference_full

root_dir=./results
exp_dir=$root_dir/$exp_n
model_dir=./checkpoints/
ckpt_name=checkpoint_50k.pt

aug=10
topk=10
name=single
input=smiles.csv # src

outputdir=$root_dir/output
mkdir -p $outputdir

#export PYTHONPATH=$PYTHONPATH:/home/ubuntu/editretro



CUDA_VISIBLE_DEVICES="0" CUDA_LAUNCH_BLOCKING=1 fairseq-interactive \
        --user-dir editretro  \
        $databin \
        -s src -t tgt \
        --input $input \
        --task translation_retro \
        --path $model_dir/$ckpt_name \
        --iter-decode-max-iter 10 \
        --iter-decode-eos-penalty 0 \
        --beam 1 --remove-bpe \
    --init-src \
    --buffer-size 3000 \
    --batch-size 1 \
        --TOPK $topk \
        --inference-with-augmentation \
        --aug $aug \
        --print-step --retain-iter-history >$outputdir/${name}.txt

python ./utils/get_ranked_topk.py \
        -output_file $outputdir/${name}.txt \
        -save_file $outputdir/ranked_output.txt \
        -augmentation $aug \
        -beam_size $topk \
        -n_best $topk \
        -score_alpha 0.1 \
        -output_edit_step

