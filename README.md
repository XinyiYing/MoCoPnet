# Local Motion and Contrast Priors Driven Deep Network for Infrared Small Target Super-Resolution
Pytorch implementation of local motion and contrast prior driven deep network (MoCoPnet). [<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9796529">PDF</a>] <br><br>

## Overview
<img src="https://raw.github.com/XinyiYing/MoCoPnet/master/images/1.PNG" width="1024"/><br>

## Requirements
- Python 3
- pytorch >= 1.6
- numpy, PIL

## Datasets

### Training & test datasets

#### Download [SAITD](https://www.scidb.cn/en/detail?dataSetId=808025946870251520&dataSetType=journal) dataset. 
SAITD dataset is a large-scale high-quality semi-synthetic dataset of infrared small target.
We employ the 1st-50th sequences with target annotations as the test datasets and the remaining 300 sequences as the training datasets. 

#### Download [Hui](https://www.scidb.cn/en/detail?dataSetId=720626420933459968&dataSetType=journal) and [Anti-UAV](https://anti-uav.github.io/dataset/). 
Hui and Anti-UAV datasets are used as the test datasets to test the robustness of our MoCoPnet to real scenes. In Anti-UAV dataset, only the sequences with infrared small target (i.e., The target size is less than 0.12% of the image size) are selected as the test set (21 sequences in total). Note that, we only use the first 100 images of each sequence for test to balance computational/time cost and generalization performance.

For simplicity, you can also Download the test datasets in https://pan.baidu.com/s/1oobhklwIChvNJIBpTcdQRQ?pwd=1113 and put the folder in code/data.

#### Data format: 

1. The training dataset is in `code/data/train/SAITD`. 
```
train
  └── SAITD
       └── 1
              ├── 0.png
              ├── 1.png
              ├── ...
       └── 2
              ├── 00001
              ├── 00002
              ├── ...		
       ...
```
2. The test datasets are in `code/data/test` as below:
```
 test
  └── dataset_1
         └── scene_1
              ├── 0.png  
              ├── 1.png  
              ├── ...
              └── 100.png    
               
         ├── ...		  
         └── scene_M
  ├── ...    
  └── dataset_N      
```

## Results

### Quantitative Results of SR performance 

<img src="https://raw.github.com/XinyiYing/MoCoPnet/master/images/2.PNG" width="1024" />

<img src="https://raw.github.com/XinyiYing/MoCoPnet/master/images/3.PNG" width="1024"/>

### Qualitative Results of SR performance 
<img src=https://raw.github.com/XinyiYing/MoCoPnet/master/images/4.PNG>

<img src=https://raw.github.com/XinyiYing/MoCoPnet/master/images/5.PNG>

### Quantitative Results of detection

<img src="https://raw.github.com/XinyiYing/MoCoPnet/master/images/6.PNG" width="1024" />

<img src="https://raw.github.com/XinyiYing/MoCoPnet/master/images/7.PNG" width="1024"/>

<img src=https://raw.github.com/XinyiYing/MoCoPnet/master/images/10.PNG>

<img src=https://raw.github.com/XinyiYing/MoCoPnet/master/images/11.PNG>

### Qualitative Results of detection

<img src=https://raw.github.com/XinyiYing/MoCoPnet/master/images/8.PNG>

<img src=https://raw.github.com/XinyiYing/MoCoPnet/master/images/9.PNG>

## Citiation
```
@article{MoCoPnet,
  author = {Ying, Xinyi and Wang, Yingqian and Wang, Longguang and Sheng, Weidong and Liu, Li and Lin, Zaiping and Zhou, Shilin},
  title = {Local Motion and Contrast Priors Driven Deep Network for Infrared Small Target Superresolution},
  journal={Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year = {2022},
}
```

## Contact
Please contact us at ***yingxinyi18@nudt.edu.cn*** for any question.

