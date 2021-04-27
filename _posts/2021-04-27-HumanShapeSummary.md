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
_______
## 基于深度学习的人体三维重建

### 基于参数化人体模型的姿态估计

基于参数化可驱动的人体模型一般包括LBS,SMPL(SMPL-X,SMPL-H),SCAPE,还有最新的STAR等等。目前主流模型还是SMPL以及其变体.基于学习的方式完成end-to-end的从单张图片估计SMPL模型的方法最早为2018年CVPR的HMR，逐渐发展到一些利用多视角、视频帧间信息的做法。模型的精度和泛化性也在逐步提高.

**本节介绍从HMR开始的人体3D姿态估计与重建的学习算法**。

#### HMR
HMR的backbone采用Resnet50从单张图片提取1024维特征，经过全连接层迭代三次输出pose+shape+camera参数，一共85维。相当于估计出一个SMPL模型。采用loss包括2Djoint，3Djoint，以及gt的SMPL参数的二范数。只靠带有ambiguity的joint约束很难得到符合正常人体姿态先验的结果，于是HMR采用GAN的方式避免这一点。训练时采用K+2个Discriminator,(K个关节点独立判断(旋转矩阵),1个全局kinetics判断,1个10维shape参数判断)。HMR通过实验证明没有paired的3D监督， 依靠2Dlable的数据量扩充也可以提升预测准确度。2D数据集：LSP,COCO,MPII。3D数据集：Human3.6M,MPIINF-3DHP;MoSH(带有SMPL参数label)
![hmr](/img/humanshapesum/hmr.png)
#### SPIN
SPIN的核心思路是采用了回归网络加优化的组合形式来达到参数网络和迭代方法的自提高。通过将网络估计的模型参数作为初值送给SMPLify(去除碰撞检测)做优化，优化依据包括关节点位置，pose,shape的prior等等。作者认为这种做法使得regression和optimaization能形成一个self-improving cycle(自提高循环?)。因此网络loss为regressoin和优化结果的L2Loss。训练过程会预先通过SMPLify得到初始结果，制作成一个表，当新优化的shape比表中误差低时则更新"最优解"，这一过程在整个训练中都持续进行。
![spin](/img/humanshapesum/spin.png)
#### I2L-Meshnet
I2L-MeshNet提出lixel(线素)的概念，想解决的问题是一般从图片回归模型参数的方法面临的高度非线性，从像素到Mesh的预测破坏了像素间的空间位置关系.  I2L转而预测Mesh每个顶点在像素1D热图上的逐像素可能性。网络包含PoseNet和MeshNet。PoseNet估计关节点位置并产生三维空间的gaussian heatmap与同尺寸的imgfeat作为MeshNet的输入。输出是N通道的heatmap，通过在x,y,x+y等维度取平均等操作回归N个顶点的相对坐标loss包括joint,Mesh顶点坐标，normal,edge长度的误差。
![I2L](/img/humanshapesum/I2L.png)
#### VIBE
VIBE的思路是采用视频序列的信息去估计一个时间段的人体姿态序列。以前的基于视频的方法面临的问题是缺少野外带
3D标注的数据集，VIBE使用mocap数据集作为真实动作序列的来源，基于GAN的方式训练，生成器、鉴别器都使用GRU的结构，前者生成动作序列，后者判断生成结果是否真实。
![vibe](/img/humanshapesum/VIBE.png)
值得注意的是鉴别器采用了自注意力机制，给不同帧以不同的权重来判断动作是否符合真实分布，相比采用pooling,平均等方式聚合多帧信息的方式，作者认为注意力机制能够考虑到每帧的细节，而不是模糊的特征。
论文提到相比temporal-HMR，它的结果平滑性稍低于temporal-HMR，但在精度上有提高。
![vibe-D](/img/humanshapesum/VIBE-D.png)
### SMPL+offset的带衣物形变重建
#### Tex2Shape
Tex2Shape的目的是从单张图片恢复出同SMPL拓扑却带有衣物褶皱细节的体型出来。输入从原图转化为UV texture map(这一步借助densepose完成uv预测和转化)，然后输入采用U-Net结构的网络得到normal和displacenment。
训练也使用了patchgan的思路判断输出是否代表真正的局部衣服细节。此外还有β网络经过卷积和全连接层得到体型参数，一起作为最终输出的参数。
数据处理中重要的步骤是扫描结果对齐。方法是使用真实的mesh和纹理贴图，渲染成图片后经由openpose检测关节点，优化SMPL的pose参数对齐动作，然后优化shape使得SMPL尽量贴合真实三维表面，但这一步得到的还是光滑的SMPL模型，所以还需最后一步优化每个顶点位置得到高频的纹理细节。
Lossf方面，对于normal和displacement更关注结构而非准确率，所以采用了MS-SSIM(多尺度结构相似性)作为度量，而对于体型参数采用L2loss。
![tex2shape](/img/humanshapesum/tex2shape.png)
#### HMD
HMD的思路是coarse-to-fine得预测模型顶点的偏移，从初级的joint到anchor最后到顶点，网络预测逐层的deform值，最终得到带形变的mesh。在vertex阶段，训练了一个shading-net从图片和当前的深度图预测一个refined的深度图，
使用了少量的带有深度gt的数据作为训练。并且这里使用了IID分解，目的是使得最终IID分解的误差最小。
![hmd](/img/humanshapesum/HMD.png)

### model-free的人体三维重建
#### DeepHuman
DeepHuman从单张图片和初始的SMPL估计预测出整个人的占用场和法向量图。具体做法是利用初始的SMPL估计得到一个semantic volume,和投影得到semantic map,
对于几何重建，使用了一个vol2vol的网络输出一个占用场，encoder阶段嵌入se map和原图的特征信息。得到占用场后，投影计算normal map,经过refine网络得到更具细节信息法向图来优化mesh。
vol2vol采用了一个Volumetric Feature Transformation的结构，作者认为这个结构的好处是相比于latent code再反卷积的形式，它保留了关于shape的原始信息，以及效率和试应不同分辨率的灵活性上有所提高。
#### PiFuHD







