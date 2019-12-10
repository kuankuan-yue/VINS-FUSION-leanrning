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

#include "feature_tracker.h"


// 好像是返回时候在图像边界内
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);//cvRound返回跟参数最接近的整数值，即四舍五入；
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 按照status中的状态，删除v中对应元素
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 构造函数
FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

//TODO: 这里应该可以做动态检测
// 把追踪到的点进行标记
// 设置遮挡部分（鱼眼相机）
// 对检测到的特征点按追踪到的次数排序
// 在mask图像中将追踪到点的地方设置为0，否则为255，目的是为了下面做特征点检测的时候可以选择没有特征点的区域进行检测。
// 在同一区域内，追踪到次数最多的点会被保留，其他的点会被删除
void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255)); // 标记点的图像

    // 保存长时间跟踪到的特征点 prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;// 帧id，点位置和点id
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));//把追踪得到的点track_cnt放入cnt_pts_id

    // sort 对给定区间的所有元素进行排序，按照点的跟踪次数，从多到少进行排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)//对于所有追踪到的点
    {
        //将跟踪到的点按照跟踪次数重新排列，并返回到forw_pts，ids，track_cnt
        if (mask.at<uchar>(it.second.first) == 255)//如果在特征点出的灰度值等于255
            // at 获取像素点的灰度值或者RGB值，可以通过image.at<uchar>(i,j)的方式轻松获取。检测新建的mask在该点是否为255
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);

            
            /*在已跟踪到角点的位置上，将mask对应位置上设为0,
            意为在cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
            进行操作时在该点不再重复进行角点检测，这样可以使角点分布更加均匀
            TODO:讲动态的特征点,也设置为这个
            图片，点，半径，颜色为0表示在角点检测在该点不起作用,粗细（-1）表示填充*/
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
        // TODO:这里应该输出一下mask  看看效果
        // cv::imshow ( "mask", mask );      // 用cv::imshow显示图像
        // cv::waitKey ( 0 );                  // 暂停程序,等待一个按键输入
    }
}

// 据算二维坐标的均方根距离
double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 对图片进行一系列操作，返回特征点featureFrame。
// 其中还包含了：图像处理、区域mask、检测特征点、计算像素速度等
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r; // 定义一个时间，为了计时
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    /*//TODO: 这里使用来做图像处理的！！！
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));//createCLAHE 直方图均衡
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();

    // --------------如果上一帧有特征点，就直接进行LK追踪
    if (prev_pts.size() > 0) 
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        if(hasPrediction)
        {
            cur_pts = predict_pts; //当前的点等于上一次的点
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1); 
            //迭代算法的终止条件
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                
            int succ_num = 0;//成功和上一阵匹配的数目
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10)//小于10时，好像会扩大搜索，输入的基于最大金字塔层次数为3
               cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        }
        else 
            //如果没有进行预测的话，直接是基于最大金字塔层次数为3
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        
        
        // reverse check 方向检查
        if(FLOW_BACK)//
        {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;
            //注意！这里输入的参数和上边的前后是相反的
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
            for(size_t i = 0; i < status.size(); i++)
            {
                //如果前后都能找到，并且找到的点的距离小于0.5
                if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }
        
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))// 如果这个点不在图像内，则剔除
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    // --------------对所有的帧，先计算一个mask，然后检测器特征点是否够多，不够多则提取新的，存入到cur_pts
    if (1)
    {
        // rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;

        // 如果当前图像的特征点cur_pts数目小于规定的最大特征点数目MAX_CNT，则进行提取
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size()); // 在图像上所需要提取的特征点最大值
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            /* goodFeaturesToTrack
            _image：8位或32位浮点型输入图像，单通道
            _corners：保存检测出的角点
            maxCorners：角点数目最大值，如果实际检测的角点超过此值，则只返回前maxCorners个强角点
            qualityLevel：角点的品质因子
            minDistance：对于初选出的角点而言，如果在其周围minDistance范围内存在其他更强角点，则将此角点删除
            _mask：指定感兴趣区，如不需在整幅图上寻找角点，则用此参数指定ROI
            blockSize：计算协方差矩阵时的窗口大小
            useHarrisDetector：指示是否使用Harris角点检测，如不指定，则计算shi-tomasi角点
            harrisK：Harris角点检测需要的k值 */
            cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
            // mask 这里肯定是指定感兴趣区，如不需在整幅图上寻找角点，则用此参数指定ROI
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        // 对于所有新提取出来的特征点，加入到cur_pts
        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        //printf("feature cnt after add %d\n", (int)ids.size());
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]); //当前帧不失真的点
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // ---------------如果是双目的
    if(!_img1.empty() && stereo_cam) 
        // 把左目的点在右目上找到，然后计算右目上的像素速度。
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();

        // --------------------如果当前帧非空
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n"); //在右侧图像上追踪特征

            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft; //左右目的状态
            vector<float> err;

            /*光流跟踪是在左右两幅图像之间进行cur left ---- cur right
            prevImg	第一幅8位输入图像 或 由buildOpticalFlowPyramid()构造的金字塔。
            nextImg	第二幅与preImg大小和类型相同的输入图像或金字塔。
            prevPts	光流法需要找到的二维点的vector。点坐标必须是单精度浮点数。
            nextPts	可以作为输入，也可以作为输出。包含输入特征在第二幅图像中计算出的新位置的二维点（单精度浮点坐标）的输出vector。当使用OPTFLOW_USE_INITIAL_FLOW 标志时，nextPts的vector必须与input的大小相同。
            status	输出状态vector(类型：unsigned chars)。如果找到了对应特征的流，则将向量的每个元素设置为1；否则，置0。
            err	误差输出vector。vector的每个元素被设置为对应特征的误差，可以在flags参数中设置误差度量的类型；如果没有找到流，则未定义误差（使用status参数来查找此类情况）。
            winSize	每级金字塔的搜索窗口大小。
            maxLevel	基于最大金字塔层次数。如果设置为0，则不使用金字塔（单级）；如果设置为1，则使用两个级别，等等。如果金字塔被传递到input，那么算法使用的级别与金字塔同级别但不大于MaxLevel。
            criteria	指定迭代搜索算法的终止准则（在指定的最大迭代次数标准值（criteria.maxCount）之后，或者当搜索窗口移动小于criteria.epsilon。）
            flags 操作标志，可选参数：
            OPTFLOW_USE_INITIAL_FLOW：使用初始估计，存储在nextPts中；如果未设置标志，则将prevPts复制到nextPts并被视为初始估计。
            OPTFLOW_LK_GET_MIN_EIGENVALS：使用最小本征值作为误差度量（见minEigThreshold描述）；如果未设置标志，则将原始周围的一小部分和移动的点之间的 L1 距离除以窗口中的像素数，作为误差度量。
            minEigThreshold	
            算法所计算的光流方程的2x2标准矩阵的最小本征值（该矩阵称为[Bouguet00]中的空间梯度矩阵）÷ 窗口中的像素数。如果该值小于MinEigThreshold，则过滤掉相应的特征，相应的流也不进行处理。因此可以移除不好的点并提升性能。 */
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);

            // 反向跟踪reverse check cur right ---- cur left
            if(FLOW_BACK)
            {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // 仅保留左右目都有的点only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */


            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);//右目的归一化平面点
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }

    // 展示轨迹 featureFrame是在这构造的
    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);


    // ----------------------构建特征点featureFrame，并加入特征点
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;
    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;//左目的特征点
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        // 特征点的id，相机id（0或1） 和 xyz_uv_velocity（特征点空间坐标，像素坐标和像素速度）
    }

    // ---------
    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id =  1;//右目的特征点
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }
    // printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

//使用F矩阵进行拒绝 删除一些点 涉及到FM_RANSAC,拒绝的是一些光流跟踪错误的点,
// TODO: 这里可能涉及到动态检测
void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p; //应该是像素坐标
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        // Estimation of fundamental matrix using the RANSAC algorithm
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

// 读取相机内参，生成相机类m_camera
void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
            //[ INFO] [1574931791.324250184]: reading paramerter of camera 4VINS_test/0config/yaml_mynt_s1030/params_left_my.yaml
        //定义了一个camera类
        camodocal::CammeraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

// 将像素座标系下的座标，转换为归一化相机座标系下的座标 即un_pts为归一化相机座标系下的座标。
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b); 
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

// 其为当前帧相对于前一帧 特征点沿x,y方向的像素移动速度
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

//在imTrack图像上画出特征点
void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;

    // ------------将两幅图像进行拼接
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
        // 图像凭借hconcat（B,C，A）; // 等同于A=[B  C]
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);
        //将imTrack转换为彩色

    // -------------在左目图像上标记特征点
    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);//FIXME: 这个是画圈的颜色问题
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    // -------------在右目图像上标记特征点
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;//计算在凭借的图像上，右目特征点的位置
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //画出左右目的匹配直线 curLeftPtsTrackRight找不到啊！！
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            // 在imTrack上，从curLeftPts到mapIt->second画箭头
        }
    }

    //TODO: 这个需要画图看一下 draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));//调整图像的大小
}

//把上一帧3d点预测到归一化平面，预测方法好像就是直接把3D点投影下来。
// 填充predict_pts
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);//prev_pt 上一帧的特征点
    }
}

// 移除野点
void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

// imTrack实在drawTrack中画出来的
cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
    
}