#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textsnake.py --exp_name Icdar2015 --max_epoch 300 --batch_size 4 --gpu 0 --input_size 800 --optim SGD --lr 0.001 --start_epoch 1131 --num_workers 12 --viz --net vgg --resume pretrained/icdar15_pretain/textsnake_vgg_1130.pth 
