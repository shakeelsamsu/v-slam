#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/Point32.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseStamped.h"

#include <iostream>
#include <math.h>
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include <opencv2/core/eigen.hpp>
// #include "opencv2/xfeatures2d.hpp"

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

#include <unordered_map>
#include <unordered_set>

using namespace cv;
using namespace std;

const char* DATASET_PATH = "dataset/sequences/00/image_0/";
const char* IMAGE_TOPIC = "/camera/rgb/image_raw/compressed";

const int IMG_WIDTH = 1280; // 1241;
const int IMG_HEIGHT = 720; //376;

const int NUM_IMAGES = 10;

bool verbose = false;

vector<int> result_point_indices {};
vector<unordered_map<int, int>> image_mappings;
vector<Eigen::Vector3f> point_3d_locs;
vector<Eigen::Vector3f> camera_angles;
vector<Eigen::Vector3f> camera_translations;

Mat last_image;
Mat curr_image;

Mat readFromDataset(int num) {
    string filename = to_string(num);
    filename.insert(0, 6 - min(6, (int) filename.size()), '0');
    string path(DATASET_PATH);
    filename.append(".png");
    path.append(filename);
    if(verbose) cout << path << endl;
    return imread(path/*, IMREAD_GRAYSCALE */);
}

void twoFrameDetectAndCompute(Mat& img1, Mat& img2, Ptr<ORB> detector, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2) {
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
}

std::vector<DMatch> match(Ptr<DescriptorMatcher> matcher, Mat& descriptors1, Mat& descriptors2) {
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

void drawMatches(Mat& img1, Mat& img2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, std::vector<DMatch> good_matches) {
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow("Good Matches", img_matches );
    waitKey();
}

int computeKey(int x, int y) {
    return y * IMG_WIDTH + x;
}

Point2f extractKey(int key) {
    return Point2f(key % IMG_WIDTH, key / IMG_WIDTH);
}

Eigen::Vector3f camera_pose = {1000, 1000, 1000};
float camera_angle = 1000;
Eigen::Vector3f camera_ref_pose = {0, 0, 0};
float camera_ref_angle = 0;
bool odomInit = false;

// Eigen::Vector3f rotateVector(Eigen::Vector3f v, Eigen::Vector3f rotAxis, float theta);

Eigen::Vector3f get3DPoint(int rgbX, int rgbY);
ros::Publisher featurePub;
sensor_msgs::PointCloud p2;
uint32_t seq2 = 0;
void ImageCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        // imshow("ros image", image->image);
        // waitKey();

        if(image->image.empty()) return;
        if(!odomInit) return;

        last_image = curr_image;
        curr_image = image->image;

        int minHessian = 8000;

        Ptr<ORB> detector = ORB::create(minHessian);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        
        p2.points.clear();
        p2.header.seq = seq2++;
        p2.header.frame_id = "odom";
                
        if (last_image.empty()) {
            printf("last_image.empty()\n");
            printf("first image!\n");
            
            Eigen::AngleAxisf aa = Eigen::AngleAxisf(-camera_angle, Eigen::Vector3f{0.0, 1.0, 0.0}); // rotate odom around some axis
            
            camera_angles.push_back(aa.axis().normalized() * aa.angle());
            camera_translations.push_back(-camera_pose); // TODO: Fix Z
            Mat img = curr_image; // readFromDataset(0);
            std::vector<KeyPoint> keypoints;
            detector->detect(img, keypoints, noArray());

            unordered_map<int, int> first_image_mapping {};
            for(KeyPoint kp : keypoints) {
                if(verbose) cout << (int) kp.pt.y << " " << (int) kp.pt.x << endl;
                int key = computeKey((int) kp.pt.x, (int) kp.pt.y);
                if(first_image_mapping.find(key) != first_image_mapping.end()) 
                    continue;
                if(verbose) cout << "corresponding key: " << key << endl;
                Eigen::Vector3f point3d = get3DPoint((int)kp.pt.x, (int)kp.pt.y);
                
                if (point3d.norm() == 0) continue;

                
                first_image_mapping[key] = result_point_indices.size();
                if(verbose) cout << "first_image_mapping[key]" << first_image_mapping[key] << endl;

                result_point_indices.push_back(result_point_indices.size());
                point_3d_locs.push_back(get3DPoint(kp.pt.x, kp.pt.y));

                geometry_msgs::Point32 point;
                point.x = point3d[0];
                point.y = point3d[1];
                point.z = point3d[2];
                p2.points.push_back(point);

                // printf("%f %f\n", kp.pt.x, kp.pt.y);
                // printf("%f %f %f\n", point.x, point.y, point.z);
            }
            image_mappings.push_back(first_image_mapping);
            if(verbose) {
                for(auto const& observation : image_mappings[0]) {
                    int observationKey = observation.first;
                    int observationPointIndex = observation.second;
                    Point2f observationLocation = extractKey(observationKey);
                    cout << 0 << " " << observationPointIndex << " " << observationLocation.x - (IMG_WIDTH / 2) << " " << -(observationLocation.y - (IMG_HEIGHT / 2)) << endl;
                }
            }
            return;
        }

        else {
            // read images
            Mat img1 = last_image; // readFromDataset(prevIndex);
            Mat img2 = curr_image; // readFromDataset(currIndex);
            
            int prevIndex = image_mappings.size() - 1;

            // Eigen::AngleAxisf aa = Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f{1.0, 0.0, 0.0}); // rotate around x-axis from base_link
            // aa = aa * Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f{0.0, 1.0, 0.0}); // rotate around y-axis (new z-axis)
            
            // now we transform from base link to odom
            Eigen::AngleAxisf aa = Eigen::AngleAxisf(-camera_angle, Eigen::Vector3f{0.0, 1.0, 0.0}); // rotate odom around some axis

            // Mat inp = Mat::zeros(3, 3, CV_64F);
            // cv::eigen2cv(aa.toRotationMatrix(), inp);
            // Mat res = Mat::zeros(1, 3, CV_64F);
            // cv::Rodrigues(inp, res);

            // camera_angles.push_back(Eigen::Vector3f {res.at<float>(0, 0), res.at<float>(0, 1), res.at<float>(0, 2)});

            camera_angles.push_back(aa.axis().normalized() * aa.angle());
            camera_translations.push_back(-camera_pose); // TODO: Fix Z

            if ( img1.empty() || img2.empty() )
            {
                cout << "Could not open or find the image!\n" << endl;
            }

            // get feature points and compute descriptors
            std::vector<KeyPoint> keypoints1, keypoints2;
            Mat descriptors1, descriptors2;
            twoFrameDetectAndCompute(img1, img2, detector, keypoints1, keypoints2, descriptors1, descriptors2);
            if(descriptors1.empty()|| descriptors2.empty()) {
                cout << "empty descriptor" << endl;
                return;
            }
            
            if(prevIndex == 50) {
                cv::imwrite("../corner_img50.png", curr_image);
                for(int i = 0; i < 3; i++) printf("%f\n", camera_angles.back()[i]);
                for(int i = 0; i < 3; i++) printf("%f\n", camera_translations.back()[i]);
                for(auto kp : keypoints2) {
                    Eigen::Vector3f point = get3DPoint((int)kp.pt.x, (int)kp.pt.y);
                    if(point.norm() == 0) continue;
                    printf("%f %f\n", kp.pt.x, kp.pt.y);
                    printf("%f %f %f\n", point[0], point[1], point[2]);
                }
            }
            
            // match
            std::vector<DMatch> good_matches = match(matcher, descriptors1, descriptors2);

            if(verbose) {
                cout << "Number of Good Matches: " << good_matches.size() << endl;
                cout << "Number of Keypoints img1: " << keypoints1.size() << endl;
                cout << "Number of Keypoints img2: " << keypoints2.size() << endl;
                // x is col y is row
            }

            // cout << "good_matches.size() " << good_matches.size() << endl;

            unordered_map<int, int> image_mapping;
            // matched points
            for(size_t i = 0; i < good_matches.size(); i++) {
                KeyPoint prev_keypoint = keypoints1[good_matches[i].queryIdx];
                KeyPoint curr_keypoint = keypoints2[good_matches[i].trainIdx];

                if( sqrt((prev_keypoint.pt.x - curr_keypoint.pt.x) * (prev_keypoint.pt.x - curr_keypoint.pt.x)
                        + (prev_keypoint.pt.y - curr_keypoint.pt.y) * (prev_keypoint.pt.y - curr_keypoint.pt.y)) > 15) {
                            // printf("skipping match\n");
                            continue;
                        }

                int prevKey = computeKey((int) prev_keypoint.pt.x, (int) prev_keypoint.pt.y);
                int currKey = computeKey((int) curr_keypoint.pt.x, (int) curr_keypoint.pt.y);

                if(verbose) cout << (int) prev_keypoint.pt.y << " " << (int) prev_keypoint.pt.x << endl;

                if(image_mappings[prevIndex].find(prevKey) == image_mappings[prevIndex].end())
                    continue;
                int prevPointIndex = image_mappings[prevIndex][prevKey];
                if(verbose) cout << "prevKey " << prevKey << endl;
                if(verbose) cout << "image_mappings[prevIndex][prevKey] " << image_mappings[prevIndex][prevKey] << endl;
                if(verbose) cout << (image_mappings[prevIndex].find(prevKey) == image_mappings[prevIndex].end()) << endl;
                if(verbose) cout << "prevPointIndex " << prevPointIndex << endl;

                image_mapping[currKey] = prevPointIndex;
            }

            // return 0;


            // unmatched points
            for(size_t i = 0; i < keypoints2.size(); i++) {
                KeyPoint curr_keypoint = keypoints2[i];
                int currKey = computeKey(curr_keypoint.pt.x, curr_keypoint.pt.y);
                
                if(image_mapping.find(currKey) == image_mapping.end()) {
                    
                    Eigen::Vector3f point3d = get3DPoint((int)curr_keypoint.pt.x, (int)curr_keypoint.pt.y);
                    if (point3d.norm() == 0) {
                        continue;
                    }
                    
                    image_mapping[currKey] = result_point_indices.size();
                    result_point_indices.push_back(result_point_indices.size());
                    point_3d_locs.push_back(get3DPoint(curr_keypoint.pt.x, curr_keypoint.pt.y));
                    if(verbose) cout << "writing new index to " << currKey << endl;
                    geometry_msgs::Point32 point;
                    point.x = point3d[0];
                    point.y = point3d[1];
                    point.z = point3d[2];
                    p2.points.push_back(point);
                }
            }

            // return 0;

            image_mappings.push_back(image_mapping);
            

            // draw some lines
            // int lineCount = 0;
            for (size_t i = 0; i < good_matches.size(); i++) {
                KeyPoint prev_keypoint = keypoints1[good_matches[i].queryIdx];
                KeyPoint curr_keypoint = keypoints2[good_matches[i].trainIdx];
                if( sqrt((prev_keypoint.pt.x - curr_keypoint.pt.x) * (prev_keypoint.pt.x - curr_keypoint.pt.x)
                        + (prev_keypoint.pt.y - curr_keypoint.pt.y) * (prev_keypoint.pt.y - curr_keypoint.pt.y)) > 15)
                        continue;
                line(img2, keypoints1[good_matches[i].queryIdx].pt, keypoints2[good_matches[i].trainIdx].pt, Scalar(0, 255, 0), 1);
                if (verbose) {
                    // cout << keypoints1[good_matches[i].queryIdx].pt << " " << keypoints2[good_matches[i].trainIdx].pt << endl;
                }
                // auto prevKey = computeKey((int) keypoints1[good_matches[i].queryIdx].pt.x, (int) keypoints1[good_matches[i].queryIdx].pt.y);
                // auto currKey = computeKey((int) keypoints2[good_matches[i].trainIdx].pt.x, (int) keypoints2[good_matches[i].trainIdx].pt.y);
            }
            imshow("lines", img2);
            waitKey(1);
        }
        featurePub.publish(p2);
    } catch(cv_bridge::Exception& e) {
        printf("err\n");
        return;
    }
}
ros::Publisher posePub;
uint32_t poseSeq = 0;
void odomCallback(const nav_msgs::Odometry& msg)
{
    if (!odomInit) {
        camera_pose = {0, 0, 0};
        camera_angle = 0;

        camera_ref_pose = {-(float)msg.pose.pose.position.y, 0, -(float)msg.pose.pose.position.x};
        camera_ref_angle = 2.0 * atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);
        odomInit = true;
    } else {
        camera_pose = Eigen::Vector3f{-(float)msg.pose.pose.position.y, 0, -(float)msg.pose.pose.position.x} - camera_ref_pose;
        camera_angle = 2.0 * atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w) - camera_ref_angle; 
        auto p = geometry_msgs::PoseStamped();
        p.pose.position.x = camera_pose[0];
        p.pose.position.y = camera_pose[1];
        p.pose.position.z = camera_pose[2];
        Eigen::AngleAxisf r = Eigen::AngleAxisf((M_PI / 2 + camera_angle), Eigen::Vector3f{0.0, 1.0, 0.0});
        Eigen::Quaternionf rQuat = Eigen::Quaternionf(r);
        rQuat.normalize();
        p.pose.orientation.x = rQuat.x();
        p.pose.orientation.y = rQuat.y();
        p.pose.orientation.z = rQuat.z();
        p.pose.orientation.w = rQuat.w();
        p.header.seq = poseSeq++;
        p.header.frame_id = "odom";
        posePub.publish(p);
    }
}

cv::Mat depth(720, 1280, CV_16UC1);
ros::Publisher cloudPub;
sensor_msgs::PointCloud p;
uint32_t seq = 0;
void depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        p.points.clear();
        p.header.seq = seq++;
        p.header.frame_id = "odom";
        memcpy(depth.data, msg->data.data(), msg->data.size());
        for(int i = 0; i < depth.rows; i++){
            for(int j = 0; j < depth.cols; j++){
                Eigen::Vector3f point3d = get3DPoint(j, i);
                geometry_msgs::Point32 point;
                point.x = point3d[0];
                point.y = point3d[1];
                point.z = point3d[2];
                p.points.push_back(point);
            }
        }
        cloudPub.publish(p);
        cv::imshow("depth.jpg", depth);
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // cv::Mat mono8_img = cv::Mat(cv_ptr->image.size(), CV_8UC1);
    // cv::convertScaleAbs(cv_ptr->image, mono8_img, 100, 0.0);
    
}

// Eigen::Vector3f rotateVector(Eigen::Vector3f v, Eigen::Vector3f rotAxis, float theta){
//     return (v * cos(odomAngle)) + (rotAxis.cross(v) * sin(theta)) + (rotAxis * (rotAxis.dot(v) * 1.0 - cos(theta)));
// }

// bonk
Eigen::Vector3f get3DPoint(int rgbX, int rgbY){
    assert(rgbX >= 0 && rgbX < IMG_WIDTH);
    assert(rgbY >= 0 && rgbY <= IMG_HEIGHT);
    uint16_t pixelDepthMM = depth.at<uint16_t>(rgbY, rgbX);
    if(pixelDepthMM == 0) return Eigen::Vector3f{0, 0, 0};
    float depthMeters = (float) pixelDepthMM / 1000;
    float normalizedX = ((float) rgbX / 1280) * 2 - 1;
    float normalizedY = ((float) rgbY / 1280) * 2 - (720.0/1280.0);
    Eigen::Vector3f v{-normalizedX, -normalizedY, -1};
    v *= (-depthMeters/v[2]); // ensures depth is equal to depthMeters

    // right now v is in the camera frame
    // printf("camera frame v: %f %f %f\n", v[0], v[1], v[2]);

    // now we transform to base link
    // Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f {1.0, 2.0, -3.0}, Eigen::Vector3f {3.0, -1.0, 2.0});
    
    // Eigen::AngleAxisf aa = Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f{1.0, 0.0, 0.0}); // rotate around x-axis from base_link
    // aa = aa * Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f{0.0, 1.0, 0.0}); // rotate around y-axis (new z-axis)
    
    // now we transform from base link to odom
    // aa = aa * Eigen::AngleAxisf(odomAngle, Eigen::Vector3f{0.0, -1.0, 0.0}); // rotate odom around some axis
    
    Eigen::AngleAxisf r = Eigen::AngleAxisf(camera_angle, Eigen::Vector3f{0.0, 1.0, 0.0});
    v = r.toRotationMatrix() * v;
    v += camera_pose;

    // printf("odomX, odomY, odomAngle: %f %f %f\n", odomX, odomY, odomAngle);
    // printf("pointLoc: %f %f %f\n", v[0], v[1], v[2]);

    // now we transform from base link to odom
    

    // local camera frame to odom frame
    // rotate
    // Eigen::Vector3f rotAxis{0.0, 1.0, 0.0};
    // // Rodrigues Rotation Formula: 
    // v = rotateVector(v, rotAxis, odomAngle);

    // translate
    // v += Eigen::Vector3f{(float) odomY, 0, (float) -odomX};

    return v;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_feature_matcher");
    ros::NodeHandle n;
    ros::Subscriber odomSub = n.subscribe("odom", 1000, odomCallback);
    cloudPub = n.advertise<sensor_msgs::PointCloud>("chatter", 1000);
    featurePub = n.advertise<sensor_msgs::PointCloud>("feature_points_3d", 1000);
    posePub = n.advertise<geometry_msgs::PoseStamped>("cam_pose", 10);

    image_transport::ImageTransport it(n);
    image_transport::Subscriber depthSub = it.subscribe("camera/depth/image_raw", 1000, depthCallback);

    int minHessian = 8000;
    
    Ptr<ORB> detector = ORB::create(minHessian);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // ROS stuff
    ros::init(argc, argv, "bonk", ros::init_options::NoSigintHandler);

    ros::Subscriber camera_sub = n.subscribe(IMAGE_TOPIC, 100, &ImageCallback);


    // Fill result_points and image_mapping for first image
    printf("spinning\n");
    ros::spin();

    printf("\n");
    printf("writing BAL file...\n");
    ofstream bal("bal.bal");

    size_t numImages = image_mappings.size();
    size_t numPoints = result_point_indices.size();
    size_t numObservations = 0;
    for(size_t imageIndex = 0; imageIndex < numImages; imageIndex++) {
        numObservations += image_mappings[imageIndex].size();
    }
    assert(numImages == camera_angles.size());
    assert(numImages == camera_translations.size());
    assert(numPoints == point_3d_locs.size());
    bal << numImages << " " << numPoints << " " << numObservations << endl;

    for(size_t imageIndex = 0; imageIndex < numImages; imageIndex++) {
        for(auto const& observation : image_mappings[imageIndex]) {
            int observationKey = observation.first;
            int observationPointIndex = observation.second;
            Point2f observationLocation = extractKey(observationKey);
            bal << imageIndex << " " << observationPointIndex << " " << observationLocation.x - (IMG_WIDTH / 2) << " " << -(observationLocation.y - (IMG_HEIGHT / 2)) << endl;
        }
    }

    for(size_t imageIndex = 0; imageIndex < numImages; imageIndex++){
        // Eigen::Vector3f rotation = rotateVector(Eigen::Vector3f{1.0, 0.0, 0.0}, Eigen::Vector3f{0.0, 0.0, 1.0}, camera_angles[imageIndex]);
        // Eigen::Vector3f rotation = Eigen::Vector3f{0.0, 0.0, camera_angles[imageIndex]};
        bal << camera_angles[imageIndex][0] << endl << camera_angles[imageIndex][1] << endl << camera_angles[imageIndex][2] << endl;
        bal << camera_translations[imageIndex][0] << endl << camera_translations[imageIndex][1] << endl << camera_translations[imageIndex][2] << endl;
        bal << 225 << endl << .411396 << endl <<  -2.710792 << endl; // camera params
    }
    
    for(size_t pointIndex = 0; pointIndex < point_3d_locs.size(); pointIndex++) {
        bal << point_3d_locs[pointIndex][0] << endl << point_3d_locs[pointIndex][1] << endl << point_3d_locs[pointIndex][2] << endl;
    }

    // cout << camera_translations.back()[0] << " " << camera_translations.back()[1] << " " << camera_translations.back()[2] << endl;

    printf("done writing bal file.\n");

    return 0;
}