#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textsnake.py --exp_name Totaltext --max_epoch 600 --batch_size 4 --gpu 0 --input_size 800 --optim SGD --lr 0.001 --viz --num_workers 12 --net vgg --resume pretrained/totaltext_pretain/textsnake_vgg_300.pth --start_epoch 300
