# Local Motion and Contrast Priors Driven Deep Network for Infrared Small Target Super-Resolution
Pytorch implementation of local motion and contrast prior driven deep network (MoCoPnet). [<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9796529">PDF</a>] <br><br>

## Overview
<img src="https://raw.github.com/XinyiYing/MoCoPnet/master/images/1.PNG" width="550"/><br>

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
Table 1. PSNR/SSIM achieved by different methods.

<img src="https://github.com/XinyiYing/MoCoPnet/blob/master/images/2.PNG" width="1100" />

Table 2. SNR and CR results of different methods achieved on super-resolved LR images and super-resolved HR images.

<img src="https://github.com/XinyiYing/MoCoPnet/blob/master/images/3.PNG" width="550"/>

### Qualitative Results of SR performance 
<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/4.PNG>

Figure 1. Visual results of different SR methods on LR images for 4x SR.

<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/5.PNG>

Figure 2. Visual results of different SR methods on LR images for 4x SR.

### Quantitative Results of detection

Table 3. Quantitative results of Tophat, ILCM, IPI achieved on super-resolved LR images.

<img src="https://github.com/XinyiYing/MoCoPnet/blob/master/images/6.PNG" width="1100" />

Table 4. Quantitative results of Tophat, ILCM, IPI achieved on super-resolved HR images.

<img src="https://github.com/XinyiYing/MoCoPnet/blob/master/images/7.PNG" width="550"/>


<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/10.PNG>

Figure 3. ROC results of Tophat, ILCM and IPI achieved on super-resolved LR images.

<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/11.PNG>

Figure 4. ROC results of Tophat, ILCM and IPI achieved on super-resolved HR images.

### Qualitative Results of detection

<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/8.PNG>

Figure 5. Qualitative results of super-resolved LR image and detection results.

<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/9.PNG>

Figure 6. Qualitative results of super-resolved HR image and detection results.


## Citiation
```
@article{MoCoPnet,
  author = {Ying, Xinyi and Wang, Yingqian and Wang, Longguang and Sheng, Weidong and Liu, Li and Lin, Zaiping and Zhou, Shilin},
  title = {Local Motion and Contrast Priors Driven Deep Network for Infrared Small Target Super-Resolution},
  journal={Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year = {2022},
}
```

## Contact
Please contact us at ***yingxinyi18@nudt.edu.cn*** for any question.

