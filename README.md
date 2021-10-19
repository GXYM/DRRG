 This is an implementation of “[Deep relational reasoning graph network for arbitrary shape text detection](http://arxiv.org/abs/2003.07493)”.
![](https://github.com/GXYM/DRRG/blob/master/result/img2_0.png)
## Prerequisites  
**python 3.7**;  
**PyTorch 1.2.0**;   
**Numpy >=1.16**;   
**CUDA 10.1**;  
**GCC >=9.0**;   
**NVIDIA GPU(with 10G or larger GPU memory for inference)**;   

 ## Compile  
```
cd ./csrc and make
cd ./nmslib/lanms and make
```  
## Data Links
Note:  download the data and put it under the data file  
1. [CTW1500](https://drive.google.com/file/d/1A2s3FonXq4dHhD64A2NCWc8NQWMH2NFR/view?usp=sharing)   
2. [TD500](https://drive.google.com/file/d/1ByluLnyd8-Ltjo9AC-1m7omZnI-FA1u0/view?usp=sharing)  
3. [Total-Text](https://drive.google.com/file/d/17_7T_-2Bu3KSSg2OkXeCxj97TBsjvueC/view?usp=sharing)  


## Train
```
cd tool
sh train_CTW1500.sh # run or other shell script 

```   
you should  modify the relevant training parameters according to the  environment， such as gpu_id and input_size:  
```
#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_TextGraph.py --exp_name Ctw1500 --max_epoch 600 --batch_size 6 --gpu 0 --input_size 640 --optim SGD --lr 0.001 --start_epoch 0 --viz --net vgg 
# --resume pretrained/mlt2017_pretain/textgraph_vgg_100.pth ### load the pretrain model,  You should change this path to your own 
```

## Eval
First, you can modify the relevant parameters in the [config.py](https://github.com/GXYM/DRRG/tree/master/util/config.py) and [option.py](https://github.com/GXYM/TextPMs/blob/master/util/option.py)
```
python  eval_TextGraph.py # Testing single round model 
or 
python  batch_eval.py #  Testing multi round models 
```   

## Qualitative results(![view](https://github.com/DRRG/DRRG/blob/master/result))  
![screenshot1](https://github.com/GXYM/DRRG/blob/master/result/screenshot_1.png)

![screenshot](https://github.com/GXYM/DRRG/blob/master/result/screenshot_22.png)  

## References  
@InProceedings{Zhang_2020_CVPR,
author = {Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng},
title = {Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}  
## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/DRRG/blob/master/LICENSE.md) file for details

