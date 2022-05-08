#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/CompressedImage.h"
#include <iostream>
#include <math.h>
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include<opencv2/opencv.hpp>
 #include <image_transport/image_transport.h>
 #include <cv_bridge/cv_bridge.h>
 using namespace std;

/* Camera Calibration Stuff: 
Opened Kinect with serial 000440615312
Color camera: 
type = 4
cx = 641.786133
cy = 366.616272
fx = 608.228882
fy = 608.156616
k1 = 0.411396
k2 = -2.710792
k3 = 1.644717
k4 = 0.291122
k5 = -2.525489
k6 = 1.563623
codx = 0.000000
cody = 0.000000
p2 = 0.000494
p1 = 0.000089
metric_radius = 0.000000
rotation = [
 0.999956  0.009388  0.000255 ;
-0.009373  0.995875  0.090252 ;
 0.000593 -0.090251  0.995919 
];
translation = [
-32.135025
-2.009181
 3.814234

];

Depth camera: 
type = 4
cx = 163.385284
cy = 157.269714
fx = 252.774521
fy = 252.861252
k1 = 0.291972
k2 = -0.109157
k3 = -0.005082
k4 = 0.633168
k5 = -0.085589
k6 = -0.030972
codx = 0.000000
cody = 0.000000
p2 = -0.000053
p1 = 0.000118
metric_radius = 0.000000
rotation = [
 1.000000  0.000000  0.000000 ;
 0.000000  1.000000  0.000000 ;
 0.000000  0.000000  1.000000 
];
translation = [
 0.000000
 0.000000
 0.000000

];

*/
float odomX, odomY, odomTheta;
void odomCallback(const nav_msgs::Odometry& msg)
{
    odomX = msg.pose.pose.position.x;
    odomY = msg.pose.pose.position.y;
    odomTheta = 2.0 * atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);
}

bool savedRGB = false;
void rgbCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr img = cv_bridge::toCvCopy(msg);
    // cout << "Got an rgb image!" << endl;
}

bool savedDepth = false;
cv::Mat depth(720, 1280, CV_16UC1);
void depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        memcpy(depth.data, msg->data.data(), msg->data.size());
        cv::imshow("depth.jpg", depth);
        cv::waitKey(10);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // cv::Mat mono8_img = cv::Mat(cv_ptr->image.size(), CV_8UC1);
    // cv::convertScaleAbs(cv_ptr->image, mono8_img, 100, 0.0);
    
    // cout << "Got a depth image!" << endl;
}

void get3DPoint(int rgbX, int rgbY){
    uint16_t pixelDepthMM = depth.at<uint16_t>(rgbX, rgbY);
    uint32_t depthMeters = pixelDepthMM / 1000;
    float normalizedX = ((float) rgbX / 1280) * 2 - 1;
    float normalizedY = ((float) rgbY / 720) * 2 - 1;
    cv::Vec3f dir{.2528, normalizedX, normalizedY};
    // dir /= dir.norm();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_feature_matcher");
    ros::NodeHandle n;
    ros::Subscriber odomSub = n.subscribe("odom", 1000, odomCallback);
    image_transport::ImageTransport it(n);
    // ros::Subscriber rgbSub = n.subscribe("camera/rgb/image_raw/compressed", 1000, rgbCallback);
    // ros::Subscriber depthSub = n.subscribe("camera/depth/image_raw/compressed", 1000, depthCallback);
    image_transport::Subscriber rgbSub = it.subscribe("camera/rgb/image_raw", 1000, rgbCallback, image_transport::TransportHints("compressed"));
    image_transport::Subscriber depthSub = it.subscribe("camera/depth/image_raw", 1000, depthCallback);
    ros::spin();
    return 0;
}