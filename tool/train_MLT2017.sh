#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_TextGraph.py --exp_name MLT2017 --max_epoch 400 --batch_size 6 --gpu 0 --input_size 640 --optim SGD --lr 0.01 --start_epoch 0 --num_workers 12 --viz --net resnet50 --save_freq 2
