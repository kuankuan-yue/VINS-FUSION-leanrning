/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    // 是为了将不同的损失函数_cost_function以及优化变量_parameter_blocks统一起来再一起添加到marginalization_info中
    ResidualBlockInfo(
        ceres::CostFunction *_cost_function, 
        ceres::LossFunction *_loss_function, 
        std::vector<double *> _parameter_blocks, 
        std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}
    /*这一步是为了将不同的损失函数_cost_function以及优化变量_parameter_blocks统一起来再一起添加到marginalization_info中。
    _loss_function:
    是核函数，在VINS-mono的边缘化中仅仅视觉残差有用到couchy核函数，
    _drop_set:
    另外会设置需要被边缘话的优化变量的位置_drop_set，这里对于不同损失函数又会有不同：
    对于先验损失，其待边缘化优化变量是根据是否等于para_Pose[0]或者para_SpeedBias[0]，也就是说和第一帧相关的优化变量都作为边缘化的对象。    
    对于IMU，其输入的_drop_set是vector{0, 1}，也就是说其待边缘化变量是para_Pose[0], para_SpeedBias[0]，也是第一政相关的变量都作为边缘化的对象，
    这里值得注意的是和后端优化不同，这里只添加了第一帧和第二帧的相关变量作为优化变量，因此边缘化构造的信息矩阵会比后端优化构造的信息矩阵要小对于视觉，
    其输入的_drop_set是vector{0, 3}，也就是说其待边缘化变量是para_Pose[imu_i]和para_Feature[feature_index]，从这里可以看出来在VINS-mono的边缘化操作中
    会不仅仅会边缘化第一帧相关的优化变量，还会边缘化掉以第一帧为起始观察帧的路标点。
    原文链接：https://blog.csdn.net/weixin_44580210/article/details/95748091*/


    void Evaluate();

    ceres::CostFunction *cost_function;//(其中parameter_block_sizes每个优化变量块的变量大小，以IMU残差为例，为[7,9,7,9]) 
    ceres::LossFunction *loss_function; //优化变量数据
    std::vector<double *> parameter_blocks;//待marg的优化变量id
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;//残差，IMU:15×1,视觉:2×1 VectorXd residuals;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// 边缘化类,需要marg掉的信息,
// 但是感觉更像是信息矩阵
// 这里需要参考崔华坤的博文:https://mp.weixin.qq.com/s/9twYJMOE8oydAzqND0UmFw
class MarginalizationInfo
{
  public:
    MarginalizationInfo(){valid = true;};
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;

    //添加参差块相关信息（优化变量，待marg的变量）
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info); 

    //计算每个残差对应的雅克比，并更新 parameter_block_data
    // 得到每次IMU和视觉观测(cost_function)对应的参数快(parameter_blocks),雅克比矩阵(jacobians),残差值(residuals)
    void preMarginalize();

    // 多线程进行marg,这里边执行了舒尔布操作.pos为所有变量维度，ｍ为需要marg掉的变量，ｎ为需要保留的变量
    void marginalize();
    
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; //所有观测项
    int m, n;//m为要边缘化的变量个数，n为要保留下来的变量个数

    /*先说变量，这里有三个unordered_map相关的变量分别是：
    parameter_block_size、parameter_block_idx、parameter_block_data，
    他们的key都同一是long类型的内存地址，而value分别是，各个优化变量的长度，
    各个优化变量在id以各个优化变量对应的double指针类型的数据。*/
    // 这几个变量参考崔华坤的博文:https://mp.weixin.qq.com/s/9twYJMOE8oydAzqND0UmFw
    std::unordered_map<long, int> parameter_block_size; // 需要marg掉的变量的大小,<优化变量内存地址,优化变量的长度>global size
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; // <优化变量内存地址,在矩阵中的id>local size
    std::unordered_map<long, double *> parameter_block_data;//<优化变量内存地址,数据>

    /*对应的有三个vector相关的变量分别是：
    keep_block_size、keep_block_idx、keep_block_data，
    他们是进行边缘化之后保留下来的各个优化变量的长度，各个优化变量在id以各个优化变量对应的double指针类型的数据*/
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;//边缘化之后从信息矩阵恢复出来雅克比矩阵
    Eigen::VectorXd linearized_residuals;//分别指的是边缘化之后从信息矩阵恢复出来残差向量
    const double eps = 1e-8;
    bool valid;

};

/* 先验残差的相关内容
包括了先验残差的损失函数 MarginalizationFactor
Evaluate 计算所有状态变量构成的残差和雅克比矩阵 */
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
