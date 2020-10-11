#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_TextGraph.py --exp_name TD500 --max_epoch 2000 --batch_size 4 --gpu 1 --input_size 640 --optim SGD --lr 0.01 --start_epoch 0 --viz --net vgg --start_epoch 0 --save_freq 50 
#--resume pretrained/mlt2017_pretain/textgraph_vgg_200.pth 
