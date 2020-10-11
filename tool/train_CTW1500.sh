#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_TextGraph.py --exp_name Ctw1500 --max_epoch 600 --batch_size 6 --gpu 0 --input_size 640 --optim SGD --lr 0.001 --start_epoch 0 --viz --net vgg 
#--resume pretrained/mlt2017_pretain/textgraph_vgg_100.pth
