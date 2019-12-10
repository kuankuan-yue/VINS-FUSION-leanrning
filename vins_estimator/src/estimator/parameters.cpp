/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX; //滑动窗口中,是删除最老帧,还是倒数第二针,是通过新帧和倒数第二帧的视差决定,也就是这个最小像素值,还需要除以焦距
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD; //acc偏执的阈值？
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME; //max solver itration time (ms), to guarantee real time 最大求解器迭代时间（毫秒），以确保实时
int NUM_ITERATIONS; //max solver itrations, to guarantee real time
int ESTIMATE_EXTRINSIC; // 外部参数校准开关
int ESTIMATE_TD; // 校准时间的开关
int ROLLING_SHUTTER; // 这个并没有赋值！！
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL; // 相片的行和列
double TD; // 输入的相机和imu时间的差值
int NUM_OF_CAM;//相机数目
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt; // 这个也没有赋值
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK; // 这个并没有赋值！！
std::vector<std::string> CAM_NAMES;
int MAX_CNT; //max feature number in feature tracking
int MIN_DIST; //min distance between two features 
double F_THRESHOLD; // ransac threshold (pixel)
int SHOW_TRACK; //publish tracking image as topic
int FLOW_BACK; // perform forward and backward optical flow to improve feature tracking accuracy


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

// 读取参数
void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r"); // r 以只读方式打开文件，该文件必须存在。
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK(); //中止程序执行，提供有用信息，提示是在那个文件中失败的
        return;         
    }
    fclose(fh); //打开文件之后一定记得关闭

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
        // XML / YAML文件存储类，封装了将数据写入文件或从文件读取数据所需的所有信息。
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    // yaml格式文件的参数传递参数传递
    fsSettings["image0_topic"] >> IMAGE0_TOPIC;//字符型变量用的是>>
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];//数字型变量用的是=
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
        //USE_IMU: 1
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
            //IMU_TOPIC: /mynteye/imu/data_raw
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv"; //输出VIO的数据
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
        //result path /home/kuan/2data/mynt_data/s1030/cali_test/1030/test8//vio.csv
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out); //输出操作
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];

    //没有外部校准参数，在程序中进行校准
    if (ESTIMATE_EXTRINSIC == 2)//
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());//装有旋转矩阵的VECTOR
        TIC.push_back(Eigen::Vector3d::Zero());// 装有位移的vector
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        // 外部校准参数开关为1，有粗略估计值，对其进行优化
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");
                //[ WARN] [1574931791.323889401]:  fix extrinsic param 

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);//讲opencv中的Mat格式转化为eigen中的matrix格式
        RIC.push_back(T.block<3, 3>(0, 0));//从（0.0）位置开始取一个3*3的矩阵
        TIC.push_back(T.block<3, 1>(0, 3));//从（0.3）位置开始取一个3*1的向量
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);
        //camera number 2

    //相机数目不等于1，也不等于2，则报错
    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }

    // 为寻找相机参数文件所在的位置做准备
    int pn = config_file.find_last_of('/');// 找到最后一个/
    std::string configPath = config_file.substr(0, pn);// 返回0到pn子字符串
    
    // 设置相机参数
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);//放的是含有相机参数的文件，因为有两个相机，所以用了vector

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;//启动双目
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        
        cv::Mat cv_T;//保存body_T_cam1为matrix格式
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }


    INIT_DEPTH = 5.0;//初始深度
    BIAS_ACC_THRESHOLD = 0.1;//acc偏执的阈值？
    BIAS_GYR_THRESHOLD = 0.1;

    //是否在线估计时间
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
        //[ INFO] [1574931791.324048298]: Unsynchronized sensors, online estimate time offset, initial td: 0
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);
        //[ INFO] [1574931791.324074965]: ROW: 480 COL: 752 

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
