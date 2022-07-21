## ORB-SLAM3

### Abstract

### I. INTRODUCTION

​	

### II. RELATED WORK

#### A. Visual SLAM

​	单目视觉SLAM首先被**MonoSLAM**解决，其使用扩展卡尔曼滤波(EKF)。

​	**PTAM**将跟踪与建图分为两个线程，并且引入关键帧以及后端优化部分。基于关键帧的SLAM会更加准确以及稳定。

​	**LSD-SLAM**使用了滑窗BA以及双串口优化和共视图。

​	**ORB-SLAM**提取ORB特征，构建共视图来限制跟踪与建图的复杂度，且有回环检测模块以及重定位功能(DBoW2)。

​	**SVO (semi-direct visual odometry)**提取FAST角点，且使用直接法来跟踪相邻帧图像中的特征点以及具有非零强度梯度的像素点，用来优化相机位姿。SVO短期数据准确度较高，长期准确度不足。

​	**DSO (Direct Sparse Odometry)**能够在特征较差的环境中准确计算位姿，在模糊或者低纹理的情况下增强鲁棒性。



#### B. Visual-Inertial SLAM

​	**MSCKF (Multi-state constraint kalman filter)**使用非线性卡尔曼滤波融合技术，利用特征边缘化来进行复杂度的降低。

​	**OKVIS**是第一个基于关键帧和BA优化的紧耦合VIO系统。支持单目以及双目相机。

​	**ROVIO**使用直接数据关联为 EFK 提供光度误差。

​	**ORB-SLAM-VI**展示了一个VI-SLAM系统，可以做到利用短期、中期、长期数据的地图融合，包括IMU预积分的BA后端优化。但是，其IMU初始化速度非常慢，长达15 s 。

​	**VINS-Mono**具有回环检测功能、四自由度的PG优化以及地图融合功能。特征跟踪使用的是稀疏光流法跟踪(LK光流法)。

​	**VI-DSO**是DSO的扩展，优化部分包括IMU观测误差以及高梯度像素点的光度误差，其有较高的准确度。随着高梯度像素信息的成功利用，纹理较差的场景区域的鲁棒性也得到了提升。其基于视觉惯性的BA初始化(优化)方式，需要20 ~ 30s 来收敛至 1%的误差。

​	**BASALT**是双目惯性里程计系统，其为BA提供VI里程计中的非线性因子。

​	**Kimera**是一个新颖且优秀的 **公制-语义建图系统(metric-semantic mapping system)**。其恢复公制(一般指恢复尺度的米制单位)的部分涉及到PG优化以及回环检测，可以达到与**VINS-Fusion**相似的精准度。

​	**ORB-SLAM3**提出一个新颖且快速的基于**最大后验估计(MAP)**的初始化方法，其适当考虑视觉和惯性传感器的不确定性，且估计误差5%情况下耗时2 s，1%的误差耗时15 s。



#### C. Multi-map SLAM























