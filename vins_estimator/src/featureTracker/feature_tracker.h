/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

// 特征跟踪的类
//FIXME: 还没仔细看呢
class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask; //用于标记点的图像
    cv::Mat fisheye_mask; 
    cv::Mat prev_img, cur_img; //先前和现在的图像，在双目中，特指左目
    vector<cv::Point2f> n_pts; //从图片上返回的特征，shi-tomasi角点（Harris角点）
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug; //setPrediction生成的，暂时不知道作用
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts; //cur_pts当前帧上的特征点，双目中的左目，并且应该像素坐标
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts; //为归一化相机座标系下的座标。
    vector<cv::Point2f> pts_velocity, right_pts_velocity; //像素移动速度
    vector<int> ids, ids_right; //ids这个好像就是当前帧特征点数目的索引
    vector<int> track_cnt;//保存了当前追踪到的角点一共被多少帧图像追踪到
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map; //cur_un_pts_map中存放ids[i]和cur_un_pts[i]构成的键值对。
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;//当前右目上的点
    map<int, cv::Point2f> prevLeftPtsMap; //上一帧的左目中的点
    vector<camodocal::CameraPtr> m_camera; //相机类，双目的话有两个
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;
};
