#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_TextGraph.py --exp_name Synthtext --max_epoch 5 --batch_size 12 --lr 0.0001 --gpu 1 --input_size 512 --save_freq 1 --viz --num_workers 16 
#--resume pretrained/synthtext_pretrain/textgraph_vgg_48000.pth
