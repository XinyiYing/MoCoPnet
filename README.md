# Deformable 3D Convolution for Video Super-Resolution
Pytorch implementation of local motion and contrast prior driven deep network (MoCoPnet). [<a href="https://arxiv.org/abs/2201.01014">PDF</a>] <br><br>

## Overview

### Architecture of MoCoPnet
<img src="https://github.com/XinyiYing/MoCoPnet/blob/main/images/1.PNG" width="550"/><br>

## Requirements
- Python 3
- pytorch >= 1.6
- numpy, PIL

## Datasets

### Training & test datasets

1. Download [SAITD](https://www.scidb.cn/en/detail?dataSetId=808025946870251520&dataSetType=journal) dataset (a large-scale high-quality semi-synthetic dataset). Note that, we employ the 1st-50th sequences with target annotations as the test datasets and the remaining 300 sequences as the training datasets. 

2. Download [Hui](https://www.scidb.cn/en/detail?dataSetId=720626420933459968&dataSetType=journal) and [Anti-UAV](https://anti-uav.github.io/dataset/). Note that, Hui and Anti-UAV is used as the test dataset to test the robustness of our MoCoPnet to real scenes. In Anti-UAV dataset, only the sequences with infrared small target (i.e., The target size is less than 0.12% of the image size) are selected as the test set (21 sequences in total). Note that, we only use the first 100 images of each sequence for test to balance computational/time cost and generalization performance.

For simplicity, you can also Download the Hui and Anti-UAV datasets in https://pan.baidu.com/s/1PKZeTo8HVklHU5Pe26qUtw (Code: 4l5r) and put the folder in code/data.

3. Data format: 

1) The training dataset is in `code/data/SAITD`. 
```
data
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
2) The test datasets are in `code/data` as below:
```
 data
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

### Qualitative Results of detection

<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/8.PNG>

Figure 3. Qualitative results of super-resolved LR image and detection results.

<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/9.PNG>

Figure 4. Qualitative results of super-resolved HR image and detection results.

<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/10.PNG>

Figure 5. ROC results of Tophat, ILCM and IPI achieved on super-resolved LR images.

<img src=https://github.com/XinyiYing/MoCoPnet/blob/master/images/11.PNG>

Figure 6. ROC results of Tophat, ILCM and IPI achieved on super-resolved HR images.

## Citiation
```
@article{D3Dnet,
  author = {Ying, Xinyi and Wang, Yingqian and Wang, Longguang and Sheng, Weidong and Liu, Li and Lin, Zaipin and Zhou, Shilin},
  title = {MoCoPnet: Exploring Local Motion and Contrast Priors for Infrared Small Target Super-Resolution},
  journal={arXiv preprint arXiv:2201.01014},
  year = {2020},
}
```

## Contact
Please contact us at ***yingxinyi18@nudt.edu.cn*** for any question.

