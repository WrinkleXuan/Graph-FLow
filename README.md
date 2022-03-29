# Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation


## Introduction
This repository contains the code  of our paper Graph Flow: Cross-layer Graph Flow Distillation for Dual Efficient Medical Image Segmentation.

The paper is under review in IEEE Transactions on Medical Imaging.

The paper is an extension version of our CoCo DistillNet which is published in the proceeding of the 2021 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM).

We will collate our whole code as soon as possible.

## Visulatiztion on GastricCancer and Synpase
![kd_visualization_new](https://user-images.githubusercontent.com/84963829/160541226-c0ea02c5-5995-4116-906d-3e756e82156a.png)

## Performance on GastricCancer and Synapse 
![image](https://user-images.githubusercontent.com/84963829/160541432-460433ce-2240-49ab-a096-f7e4a2f9662a.png)

## Requirments

* Python 3.6
* Pytorch 1.7.1
* Two NVIDIA TITAN XP GPUs


## Acknowledgement
The codebase of semantic segmentation is succeed from the work of my senior schoolmates [Accurate Retinal Vessel Segmentation in Color Fundus Images via Fully Attention-Based Networks](https://ieeexplore.ieee.org/abstract/document/9210783). 

The codebase of kd is heavily borrowed from [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) and [structure_knowledge_distillation](https://github.com/irfanICMLL/structure_knowledge_distillation) .

The pre-processed Synapse is from [Transunet](https://github.com/Beckschen/TransUNet).

Thanks for their excellent works.
