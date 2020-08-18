#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textsnake.py --exp_name TD500 --max_epoch 2000 --batch_size 4 --gpu 1 --input_size 640 --optim SGD --lr 0.01 --start_epoch 0 --viz --net vgg --resume pretrained/mlt2017_pretain/textsnake_vgg_200.pth --start_epoch 0 --save_freq 50
