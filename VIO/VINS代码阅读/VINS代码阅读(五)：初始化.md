# VINS代码阅读(五)：初始化

https://blog.csdn.net/tmfighting/article/details/125350294

## 一、概述

​	在processImage函数中，若当前未初始化，则需要先进行初始化操作。

​	**初始化目的：**计算出尺度信息、初始bias信息、重力加速度信息。纯视觉主要计算尺度信息，VIO都要计算。

## 二、代码流程

### 2.1 附：标定外参

​	**当ESTIMATE_EXTRINSIC参数为2时，程序中进行外参标定。**不过推荐先自行对传感器标定，之后再令该参数为1，优化外参。当外参优化程度较好时，可以考虑把该参数改为0，即固定外参。

### 2.2 初始化入口

​	初始化入口位于processImage函数中，即图像数等于滑窗大小时，进行初始化。

**初始化有三种情况，针对传感器：**

- 单目+IMU
- 双目+IMU
- 双目

我们详细看一下单目+IMU以及双目+IMU的情况

### 2.3 单目+IMU

很明显，我们看到初始化过程中涉及到的函数为**initialStructure()**

```c++
// monocular + IMU initilization
if (!STEREO && USE_IMU)
{
    if (frame_count == WINDOW_SIZE)
    {
        bool result = false;
        if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
        {
            result = initialStructure();
            initial_timestamp = header;   
        }
        if(result)
        {
            optimization();
            updateLatestStates();
            solver_flag = NON_LINEAR;
            slideWindow();
            ROS_INFO("Initialization finish!");
        }
        else
            slideWindow();
    }
}
```



接下来，**看initialStructure函数**

#### 2.3.1 判断IMU数据是否足够后续优化

```c++
//check imu observibility
{
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    //遍历所有图像
    //计算线加速度总值
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
        double dt = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        sum_g += tmp_g;
    }
    Vector3d aver_g;
    //计算均值
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    //计算方差
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
        double dt = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        //cout << "frame g " << tmp_g.transpose() << endl;
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));
    //ROS_WARN("IMU variation %f!", var);
    
    //若方差小于0.25，则认为IMU测量量不足。
    if(var < 0.25)
    {
        ROS_INFO("IMU excitation not enouth!");
        //return false;
    }
}
```

#### 2.3.2 relativePose函数(五点法恢复位姿)

​	**根据论文部分，首先进行纯视觉初始化**：检查最新帧和所有先前帧之间的特征对应关系。 如果能在滑动窗口中找到最新帧与任何其他帧之间的稳定特征跟踪（超过 30 个跟踪的特征）和足够的视差（超过 20 个像素），**则进行5点法计算相对位姿**。之后任意设定尺度进行三角化，得到3D特征点进行PnP计算滑窗内其他帧的位姿。

​	在滑窗内寻找与当前帧的匹配特征点数较多的关键帧作为**参考帧**，并通过求解基础矩阵计算出当前帧到参考帧的T。

```c++
//
Matrix3d relative_R;
Vector3d relative_T;
int l;
if (!relativePose(relative_R, relative_T, l))
{
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
}
```



```c++
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    //在在滑动窗口内(共11帧)寻找合适的图像帧(和(新)窗口最后一帧对应特征点的平均视差>视差阈值且特征点匹配数量>20)
    //且可利用solveRelativeRT解算出R/T)
    //记录下初始化图像对的索引(窗口内最后一帧和筛选得到的第L帧)和相对姿态(relative_R/relative_T)
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        //遍历滑动窗口，找到滑动窗口中符合上述条件的帧
        //若找不到就marg最老的帧，一直更新最新帧，直到有符合的帧
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}
```



**五点法求解相对位姿：**

​	理论部分见pdf，或者见*SLAM基础知识复习文档*。

```c++
bool MotionEstimator::solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    //执行七点法RANCAC，最少15个点
    if (corres.size() >= 15)
    {
        //两帧匹配点对
        //F矩阵构建了两帧像素点间的约束
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        //求解基础矩阵，使用方法：FM_RANSAC
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        //从E矩阵恢复姿态 E=t^R
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        //算法过程中，验证的内点大于12，则认为位姿估计正确
        if(inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}
```

#### 2.3.3 全局SFM

**恢复窗口内所有图像帧位姿和landmarks**。

流程：

a.基于输入的初始化图像对和相对Pose，进行两帧内三角化triangulateTwoFrames
b.以第l帧为纯视觉坐标系原点，通过Pnp计算第l+1帧到窗口最后一帧中所有图像帧的姿态
c.以第l帧为纯视觉坐标系原点，通过Pnp计算窗口第0帧到第l-1帧中所有图像帧的姿态
d.基于滑动窗口内的所有带Pose的图像帧，三角化窗口内剩余的特征轨迹
e.执行全局BA，添加滑动窗口内各图像帧的四元数形式的旋转和向量形式的平移作为优化变量，遍历所有特征对建立残差项
f.将优化结果取出，计算滑动窗口内各帧到参考帧l的转换矩阵，并记录所有地图点sfm_tracked_points



![]()

```c++
// global sfm
//滑动窗口以及最新帧 12帧的位姿
Quaterniond Q[frame_count + 1];
Vector3d T[frame_count + 1];
map<int, Vector3d> sfm_tracked_points;
vector<SFMFeature> sfm_f;
//汇总特征管理器记录的所有图像帧的特征信息，整理得到所有特征轨迹sfm_f
//观测到的特征点在滑窗每幅图像中的2D位置
for (auto &it_per_id : f_manager.feature)
{
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto &it_per_frame : it_per_id.feature_per_frame)
    {
        imu_j++;
        Vector3d pts_j = it_per_frame.point;
        tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
} 
Matrix3d relative_R;
Vector3d relative_T;
int l;
if (!relativePose(relative_R, relative_T, l))
{
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
}
//进行特征重建SFM
// param_1(in): 待求解位姿的图像帧总数目
// param_2(out): 当前滑动窗口内的图像帧姿态，从滑动窗口内各图像帧坐标系到第l帧相机坐标系的旋转矩阵
// param_3(out): 当前滑动窗口内的图像帧平移，从滑动窗口内各图像帧坐标系到第l帧相机坐标系的平移向量
// param_4(in): 初始化图像对的左图索引，右图索引为窗口内最后一帧
// param_5，param_6(in): 初始化图像对的相对位姿，注意relative_R表示从当前最新帧到第l帧的坐标系变换relative_R和relative_T
// param_7(in,out): 特征管理器内获取的所有特征轨迹
// param_8(out): sfm计算的到的稀疏点云
GlobalSFM sfm;
if(!sfm.construct(frame_count + 1, Q, T, l,
                  relative_R, relative_T,
                  sfm_f, sfm_tracked_points))
{
    ROS_DEBUG("global SFM failed!");
    marginalization_flag = MARGIN_OLD;
    return false;
}
```



##### 2.3.3.1 三角化第l(关键)帧与最新帧中观测到的特征点

​	由前面可知，**我们用五点法求解基础矩阵F，得到了这两帧之间的相对位姿，现在我们用这个位姿进行三角化。**

​	三角化后，这两帧之间观测到的特征点都变成了3D特征点，再通过3D-2D:PnP算法计算第l+1,...,l+n帧到最新帧的相对位姿。

​	之后再三角化这些帧中观测到的特征点。

![]()



```c++
//1: trangulate between l ----- frame_num - 1
//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
for (int i = l; i < frame_num - 1 ; i++)
{
    // solve pnp
    //第一次循环 i=l 不执行如下pnp求解，只执行三角化
    if (i > l)
    {
        Matrix3d R_initial = c_Rotation[i - 1];
        Vector3d P_initial = c_Translation[i - 1];
        if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
}
```



##### 2.3.3.2 三角化l->倒数第二帧的特征点

```c++
//3: triangulate l-----l+1 l+2 ... frame_num -2
for (int i = l + 1; i < frame_num - 1; i++)
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
```



##### 2.3.3.3 求解l帧之前的PnP

第l帧的特征点已经恢复，那么还是可以用3D-2DPnP方法求解相对位姿。

```c++
//4: solve pnp l-1; triangulate l-1 ----- l
//             l-2              l-2 ----- l
for (int i = l - 1; i >= 0; i--)
{
    //solve pnp
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
        return false;
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    c_Quat[i] = c_Rotation[i];
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    //triangulate
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
}
```



##### 2.3.3.4 三角化所有特征点

```c++
//5: triangulate all other points
for (int j = 0; j < feature_num; j++)
{
    if (sfm_f[j].state == true)
        continue;
    if ((int)sfm_f[j].observation.size() >= 2)
    {
        Vector2d point0, point1;
        int frame_0 = sfm_f[j].observation[0].first;
        point0 = sfm_f[j].observation[0].second;
        int frame_1 = sfm_f[j].observation.back().first;
        point1 = sfm_f[j].observation.back().second;
        Vector3d point_3d;
        triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
        sfm_f[j].state = true;
        sfm_f[j].position[0] = point_3d(0);
        sfm_f[j].position[1] = point_3d(1);
        sfm_f[j].position[2] = point_3d(2);
        //cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
    }		
}
```



##### 2.3.3.5 滑动窗口全局范围BA

设置优化参数

```c++
//full BA
ceres::Problem problem;
ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
//cout << " begin full BA " << endl;
for (int i = 0; i < frame_num; i++)
{
    //double array for ceres
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    c_rotation[i][0] = c_Quat[i].w();
    c_rotation[i][1] = c_Quat[i].x();
    c_rotation[i][2] = c_Quat[i].y();
    c_rotation[i][3] = c_Quat[i].z();
    //优化参数：位姿：四元数+平移向量
    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[i], 3);
    //不优化第l帧与最后一帧的位姿
    if (i == l)
    {
        problem.SetParameterBlockConstant(c_rotation[i]);
    }
    if (i == l || i == frame_num - 1)
    {
        problem.SetParameterBlockConstant(c_translation[i]);
    }
}
```

添加残差：边->位姿节点，重投影误差

```c++
for (int i = 0; i < feature_num; i++)
{
    if (sfm_f[i].state != true)
        continue;
    for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
    {
        int l = sfm_f[i].observation[j].first;
        ceres::CostFunction* cost_function = ReprojectionError3D::Create(
            sfm_f[i].observation[j].second.x(),
            sfm_f[i].observation[j].second.y());

        problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
                                 sfm_f[i].position);	 
    }

}
```

进行优化

```c++
ceres::Solver::Options options;
options.linear_solver_type = ceres::DENSE_SCHUR;
//options.minimizer_progress_to_stdout = true;
options.max_solver_time_in_seconds = 0.2;
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
//std::cout << summary.BriefReport() << "\n";
if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
{
    //cout << "vision only BA converge" << endl;
}
else
{
    //cout << "vision only BA not converge " << endl;
    return false;
}
for (int i = 0; i < frame_num; i++)
{
    q[i].w() = c_rotation[i][0]; 
    q[i].x() = c_rotation[i][1]; 
    q[i].y() = c_rotation[i][2]; 
    q[i].z() = c_rotation[i][3]; 
    q[i] = q[i].inverse();
    //cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
}
for (int i = 0; i < frame_num; i++)
{

    T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    //cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
}
for (int i = 0; i < (int)sfm_f.size(); i++)
{
    if(sfm_f[i].state)
        sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
}
return true;
```

#### 2.3.4 最后进行PnP

​	对所有图像帧(包括滑动窗口外的图像帧)，给定初始的R/T，然后执行solvePnp进行优化，这个过程修正了all_image_frame中存储的每一帧的R/T(使用相机到imu的标定外参做了转换)。

**注意这里对all_image_frame进行处理**。

```c++
//solve pnp for all frame
// 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化
map<double, ImageFrame>::iterator frame_it;
map<int, Vector3d>::iterator it;
frame_it = all_image_frame.begin( );
// 遍历所有frame数据
// 这些frame数据有三种情况
// a.时间戳小于滑动窗口第一帧时间戳
// b.时间戳在滑动窗口内第i/i+1帧之间
// c.时间戳大于滑动窗口最后一帧时间戳(不可能发生，因为滑动窗口内最后一帧就是当前最新帧)
for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
{
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    // 若当前frame在滑动窗口内(即关键帧keyframe)，则直接使用construct中更新的Q/T
    if((frame_it->first) == Headers[i].stamp.toSec())
    {
        frame_it->second.is_key_frame = true;
        frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
        frame_it->second.T = T[i];
        i++;
        continue;
    }
    if((frame_it->first) > Headers[i].stamp.toSec())
    {
        i++;
    }
    // 以距离当前普通帧时序上最近的滑动窗口内对应关键帧的Pose作为初值，执行Pnp求解当前普通帧的位姿
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = - R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);
    // 标记当前图像帧为普通帧
    frame_it->second.is_key_frame = false;
    // 建立特征点3d位置和对应2d位置的向量，用以Pnp计算位姿
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto &id_pts : frame_it->second.points)
    {
        int feature_id = id_pts.first;
        for (auto &i_p : id_pts.second)
        {
            it = sfm_tracked_points.find(feature_id);
            if(it != sfm_tracked_points.end())
            {
                Vector3d world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                Vector2d img_pts = i_p.second.head<2>();
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
    if(pts_3_vector.size() < 6)
    {
        cout << "pts_3_vector size " << pts_3_vector.size() << endl;
        ROS_DEBUG("Not enough points for solve pnp !");
        return false;
    }
    if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
    {
        ROS_DEBUG("solve pnp fail!");
        return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp,tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    // R_pnp表示当前普通帧的相机坐标系到视觉坐标系(第l帧的相机坐标系)的旋转矩阵
    // T_pnp表示当前普通帧的相机坐标系到视觉坐标系(第l帧的相机坐标系)的平移向量
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    // frame_it->second.R表示从当前帧的IMU坐标系到视觉坐标系(第l帧相机坐标系)的旋转矩阵
    // 因为VINS-MONO中相机和IMU之间的外参平移项被忽略
    // 因此frame_it->second.Ｔ(T_pnp)表示从当前帧的IMU坐标系到视觉坐标系(第l帧相机坐标系)的平移向量
    frame_it->second.R = R_pnp * RIC[0].transpose();
    frame_it->second.T = T_pnp;
}

```



计算完这些后，会执行一次optimization()。

### 2.4 双目+IMU（未完成）

### 



