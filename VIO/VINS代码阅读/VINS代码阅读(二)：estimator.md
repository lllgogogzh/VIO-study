# VINS代码阅读(二)：estimator

## 一、概述

​	这里主要是状态估计器的代码。

## 二、IMU数据入口 imputIMU函数

```c++
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    //构造pair，时间和加速度角速度对。压入accBuf和gyrBuf。
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}
```

### 1、fastPredictIMU函数

快速预测函数，简单计算PVQ
$$
dt=t_{now}-t_{last}——时间间隔\\
acc_0=lastQ*(lastacc_0-lastB_a)-g——上时刻加速度\\
gyr=\frac{1}{2}(lastgyr_0-lastB_g+angular-lastB_g)——当前角速度(上时刻和现在均值)\\
lastQ=lastQ*\Delta q——现在旋转姿态\\
acc_1=lastQ(acc_{meas}-lastB_a)-g——当前加速度\\
acc_{now}=\frac{1}{2}(acc_0+acc_1)——取均值\\
P=P_{last}+V_{last}*dt+\frac{1}{2}acc_{now}*dt^2\\
V=V_{last}+acc_{now}*dt\\
再更新测量值即可。当前的变为之前的，之后下一次的变为当前的。
$$
*公式对应代码*

```c++
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}
```

## 三、图片入口函数 imputImage

```c++
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    if(_img1.empty())	//单目
        featureFrame = featureTracker.trackImage(t, _img);
    else				//双目
        featureFrame = featureTracker.trackImage(t, _img, _img1);

    if (SHOW_TRACK)
    {
        //展示轨迹
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }
    
    if(MULTIPLE_THREAD)  
    {     
        //多线程
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}
```

## 四、测量处理入口 processMeasurements()

```c++
void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if(!featureBuf.empty())
        {
            feature = featureBuf.front();
            curTime = feature.first + td;
            while(1)
            {//等待imu，如果不用imu或者imu已经好了，那么直接break
                //IMUAvailable是来判断imu是不是已经好了的函数
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }//end while(1)
            mBuf.lock();
            if(USE_IMU)//获取IMU信息
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();//弹出特征信息
            mBuf.unlock();

            if(USE_IMU)
            {
                if(!initFirstPoseFlag)//第一帧imu pose
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);//处理IMU数据
                }
            }
            mProcess.lock();
            processImage(feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}
```

### 1、获取IMU信息函数 IMUAvailable()

参数：t0 之前时间；t1 现在时间；accVector要获取的加速度；gyrVector要获取的角速度。

```c++
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        //没有数据
        printf("not receive imu\n");
        return false;
    }
    
	//我们想取得t0~t1时间内所有imu数据
    //如果最新时刻的imu数据的时间还小于t1
    //说明我们等待的时间还不够，需要等待
    if(t1 <= accBuf.back().first)
    {
        //imu最新数据时间已经在t1之后，我们要把t0~t1时间的数据搞出来
        while (accBuf.front().first <= t0)
        {
            //舍弃t0之前的
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            //保存t0~t1的
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        //再压入最后一个
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}
```

### 2、处理每时刻的IMU数据 processIMU

```c++
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        //当前帧数的预积分值是否为空，若为空则重建一个
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);//预积分
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);//预积分

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}
```

### 3、处理图像特征(未完成)





























