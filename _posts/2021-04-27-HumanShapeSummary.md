---
layout:     post
title:      人体姿态估计与三维重建综述
subtitle:   Human Pose and Shape
date:       2021-04-27
author:     WY
header-img: img/404-bg.jpg
catalog: 	 true
tags:
    - Humam Pose and Shape
    - 人体姿态与重建
---

# 人体姿态估计与三维重建综述

## 基于深度学习的人体三维重建

### 基于参数化人体模型的姿态估计  

#### _基于参数化可驱动的人体模型一般包括LBS,SMPL(SMPL-X,SMPL-H),SCAPE,还有最新的STAR等等。目前主流模型还是SMPL以及其变体.基于学习的方式完成end-to-end的从单张图片估计SMPL模型的方法最早为2018年CVPR的HMR，逐渐发展到一些利用多视角、视频帧间信息的做法。模型的精度和泛化性也在逐步提高._

**本节介绍从HMR开始的人体3D姿态估计与重建的学习算法**。

#### HMR
##### HMR的backbone采用Resnet50从单张图片提取1024维特征，经过全连接层迭代三次输出pose+shape+camera参数，一共85维。相当于估计出一个SMPL模型。采用loss包括2Djoint，3Djoint，以及gt的SMPL参数的二范数。只靠带有ambiguity的joint约束很难得到符合正常人体姿态先验的结果，于是HMR采用GAN的方式避免这一点。训练时采用K+2个Discriminator,(K个关节点独立判断(旋转矩阵),1个全局kinetics判断,1个10维shape参数判断)。HMR通过实验证明没有paired的3D监督， 依靠2Dlable的数据量扩充也可以提升预测准确度。2D数据集：LSP,COCO,MPII。3D数据集：Human3.6M,MPIINF-3DHP;MoSH(带有SMPL参数label)
![hmr](/img/humanshapesum/hmr.png)
####

