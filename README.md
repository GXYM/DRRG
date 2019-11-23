# Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection
![](https://github.com/anoycode22/DRRG/blob/master/result/img2_0.png)
## 1.Prerequisites  
**python 3.7**;  
**PyTorch 1.2.0**;   
**Numpy >=1.16**;   
**CUDA 10.1**;  
**GCC >=9.0**;   
**NVIDIA GPU(with 10G or larger GPU memory for inference)**;   
## 2.Description  
* Generally, this code has following features:  
  1.Just include complete inference code  
  2.Support TD500 and CTW1500 datasets  
## 3.Parameter setting 
* **CTW1500**: follow the [model/Ctw1500/ctw1500_test.txt](https://github.com/anoycode22/DRRG/model/TD500/ctw1500_test.txt)
* **TD500**: follow the [model/TD500/TD500_test.txt](https://github.com/anoycode22/DRRG/model/Ctw1500/TD500_test.txt)

## 4.Pretrained Models
 *  CTW1500 pretrained model: [CTW1500](https://drive.google.com/open?id=1cyAW7X4LESCJV6pEcSWw3BnXOnZSSPPC)
 *  TD500 pretrained model: [TD500](https://drive.google.com/open?id=1WKFJsotug9qeuMxqnmgBbMPDR6CaujsM)
 
## 5.Running tests
* **Preparation**  
1. git clone https://github.com/anoycode22/DRRG.git  
2. put your test images in "data/TD500/Test" or data/ctw1500/test/text_image
3. put the pretrained model into ["model/TD500/"](https://github.com/anoycode22/DRRG/tree/master/model/TD500) or ["model/Ctw1500"](https://github.com/anoycode22/DRRG/tree/master/model/Ctw1500)
4. cd ./csrc and make
5. cd ./nmslib/lanms and make

* **CTW1500**  
1. set the parameter in [config](https://github.com/anoycode22/DRRG/tree/master/util/config.py) according to [model/Ctw1500/ctw1500_test.txt](https://github.com/anoycode22/DRRG/model/TD500/ctw1500_test.txt)
2. python eval_TextGraph.py --exp_name Ctw1500 --test_size \\(512, 1024\\)

 * **TD500**  
 1. set the parameter in [config](https://github.com/anoycode22/DRRG/tree/master/util/config.py) according to [model/TD500/TD500_test.txt](https://github.com/anoycode22/DRRG/model/Ctw1500/TD500_test.txt)
 2. python eval_TextGraph.py --exp_name TD500 --test_size \\(512, 640\\)

## 6.Qualitative results(![view](https://github.com/anoycode22/DRRG/blob/master/result))  
![](https://github.com/anoycode22/DRRG/blob/master/result/screenshot_1.png)

![](https://github.com/anoycode22/DRRG/blob/master/result/screenshot_22.png)
