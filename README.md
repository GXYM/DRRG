# Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection
## 1.Prerequisites  
**python 3.7**;  
**PyTorch 1.2.0**;   
**Numpy >=1.16.4**;   
**CUDA 10.1**;   
**NVIDIA GPU(with 8G or larger GPU memory for inference)**;   
## 2.Description  
Generally, this code has following features:  
  1.Just include complete inference code  
  2.Support TD500 and CTW1500 dataset  
## 3.Parameter setting 
* **CTW1500**: follow the [model/Ctw1500/ctw1500_test.txt](https://github.com/anoycode22/DRRG/model/TD500/ctw1500_test.txt)
* **TD500**: follow the [model/TD500/TD500_test.txt](https://github.com/anoycode22/DRRG/model/Ctw1500/TD500_test.txt)

## 4.Running tests
* **Clone Project**  
git clone https://github.com/princewang1994/TextSnake.pytorch.git

* **CTW1500**  
1. set the Parameter in [config](https://github.com/anoycode22/DRRG/tree/master/util/config.py) according to [model/Ctw1500/ctw1500_test.txt](https://github.com/anoycode22/DRRG/model/TD500/ctw1500_test.txt)
 2. python eval_TextGraph.py --exp_name Ctw1500 --test_size (512, 1024)

 * **TD500**  
 1. set the Parameter in [config](https://github.com/anoycode22/DRRG/tree/master/util/config.py) according to [model/TD500/TD500_test.txt](https://github.com/anoycode22/DRRG/model/Ctw1500/TD500_test.txt)
 2. python eval_TextGraph.py --exp_name TD500 --test_size (512, 640)

## 5.Pretrained Models
 *  CTW1500 pretrained model: 
 *  TD500 pretrained model: 
## 6.Qualitative results
![td500_0](https://github.com/anoycode22/DRRG/tree/master/result/2.jpg)

![td500_1](https://github.com/anoycode22/DRRG/tree/master/result/9.jpg)

![ctw1500_0](https://github.com/anoycode22/DRRG/tree/master/result/1157.jpg)

![ctw1500_1](https://github.com/anoycode22/DRRG/tree/master/result/1157_0.jpg)

![ctw1500_2](https://github.com/anoycode22/DRRG/tree/master/result/1410.jpg)

![ctw1500_3](https://github.com/anoycode22/DRRG/tree/master/result/1410_0.jpg)

![ctw1500_4](https://github.com/anoycode22/DRRG/tree/master/result/1165.jpg)

![ctw1500_5](https://github.com/anoycode22/DRRG/tree/master/result/1165_00.jpg)
  


