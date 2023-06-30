 This is an implementation of “[Deep relational reasoning graph network for arbitrary shape text detection](http://arxiv.org/abs/2003.07493)”.
![](https://github.com/GXYM/DRRG/blob/master/result/img2_0.png)

## News
- [x]  Our new work at [https://github.com/GXYM/TextBPN-Plus-Plus](https://github.com/GXYM/TextBPN-Plus-Plus).  
- [x]  This project is reproduced in [MMOCR](https://github.com/open-mmlab/mmocr).  
- [x]  This project is reproduced in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_drrg_en.md).
- [x]  This project is reproduced by Paddle implementation in  [DRRG_Paddle](https://github.com/zhiminzhang0830/DRRG_Paddle). Description of reproduce is in [Paddle AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4595172)   



## Prerequisites  
**python 3.7**;  
**PyTorch 1.2.0**;   
**Numpy >=1.16**;   
**CUDA 10.1**;  
**GCC >=9.0**;  
**opencv-python < 4.5.0**  
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

## Models
 *  The trained models of Total-Text, CTW-1500 model, MSRA-TD500, MLT2017, Icdar2015 all in here.   
 [Google Drive](https://drive.google.com/file/d/1xH-jfhTO7grgk-P3kjzBDWstjjb0PYJY/view?usp=share_link) or [Baidu Drive](https://pan.baidu.com/s/1dDZwkK3PDJh0Mr903kWD8g) (download code: cfat)


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
```
@inproceedings{DBLP:conf/cvpr/ZhangZHLYWY20,
  author       = {Shi{-}Xue Zhang and
                  Xiaobin Zhu and
                  Jie{-}Bo Hou and
                  Chang Liu and
                  Chun Yang and
                  Hongfa Wang and
                  Xu{-}Cheng Yin},
  title        = {Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection},
  booktitle    = {2020 {IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2020, Seattle, WA, USA, June 13-19, 2020},
  pages        = {9696--9705},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2020},
  doi          = {10.1109/CVPR42600.2020.00972},
}

@inproceedings{DBLP:conf/iccv/Zhang0YWY21,
  author    = {Shi{-}Xue Zhang and
               Xiaobin Zhu and
               Chun Yang and
               Hongfa Wang and
               Xu{-}Cheng Yin},
  title     = {Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection},
  booktitle = {2021 {IEEE/CVF} International Conference on Computer Vision, {ICCV} 2021, Montreal, QC, Canada, October 10-17, 2021},
  pages     = {1285--1294},
  publisher = {{IEEE}},
  year      = {2021},
}

@article{zhang2023arbitrary,
  title={Arbitrary shape text detection via boundary transformer},
  author={Zhang, Shi-Xue and Yang, Chun and Zhu, Xiaobin and Yin, Xu-Cheng},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}

@article{DBLP:journals/pami/ZhangZCHY23,
  author       = {Shi{-}Xue Zhang and
                  Xiaobin Zhu and
                  Lei Chen and
                  Jie{-}Bo Hou and
                  Xu{-}Cheng Yin},
  title        = {Arbitrary Shape Text Detection via Segmentation With Probability Maps},
  journal      = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume       = {45},
  number       = {3},
  pages        = {2736--2750},
  year         = {2023},
  url          = {https://doi.org/10.1109/TPAMI.2022.3176122},
  doi          = {10.1109/TPAMI.2022.3176122},
}

@article{zhang2022kernel,
  title={Kernel proposal network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Yang, Chun and Yin, Xu-Cheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}

```
## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/DRRG/blob/master/LICENSE.md) file for details

