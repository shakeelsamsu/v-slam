#include <iostream>
#include <dirent.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <unordered_map>
#include <unordered_set>

using namespace cv;
// using namespace cv::xfeatures2d;
using namespace std;

const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }"
    "{ v verbose |               | Use of verbose flag. }";

const char* DATASET_PATH = "dataset/sequences/00/image_0/";

const int IMG_WIDTH = 1241;
const int IMG_HEIGHT = 376;

const int NUM_IMAGES = 10;

bool verbose = false;

vector<int> result_point_indices {};
vector<unordered_map<int, int>> image_mappings;

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

int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );
    verbose = parser.get<bool>("verbose");

    int minHessian = 400;
    
    Ptr<ORB> detector = ORB::create(minHessian);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");


    // Fill result_points and image_mapping for first image
    {
        Mat img = readFromDataset(0);
        std::vector<KeyPoint> keypoints;
        detector->detect(img, keypoints, noArray());

        unordered_map<int, int> first_image_mapping {};
        for(KeyPoint kp : keypoints) {
            if(verbose) cout << (int) kp.pt.y << " " << (int) kp.pt.x << endl;
            int key = computeKey((int) kp.pt.x, (int) kp.pt.y);
            if(verbose) cout << "corresponding key: " << key << endl;
            first_image_mapping[key] = result_point_indices.size();
            if(verbose) cout << "first_image_mapping[key]" << first_image_mapping[key] << endl;
            result_point_indices.push_back(result_point_indices.size());
        }
        image_mappings.push_back(first_image_mapping);
    }

    unordered_map<int, unordered_set<int>> correspondent_indices {};

    for(int image_index = 1; image_index < NUM_IMAGES; image_index++) {
        int prevIndex = image_index - 1;
        int currIndex = image_index;

        // read images
        Mat img1 = readFromDataset(prevIndex);
        Mat img2 = readFromDataset(currIndex);

        if ( img1.empty() || img2.empty() )
        {
            cout << "Could not open or find the image!\n" << endl;
            parser.printMessage();
            return -1;
        }

        // get feature points and compute descriptors
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        twoFrameDetectAndCompute(img1, img2, detector, keypoints1, keypoints2, descriptors1, descriptors2);

        // match
        std::vector<DMatch> good_matches = match(matcher, descriptors1, descriptors2);

        if(verbose) {
            cout << "Number of Good Matches: " << good_matches.size() << endl;
            cout << "Number of Keypoints img1: " << keypoints1.size() << endl;
            cout << "Number of Keypoints img2: " << keypoints2.size() << endl;
        }


        for(auto const& e : image_mappings[prevIndex]) {
            auto prevKey = e.first;
            auto prevPointIndex = e.second;

            if(verbose) cout << prevKey << " : " << prevPointIndex << endl;
        }
        

        unordered_map<int, int> image_mapping;
        // matched points
        for(int i = 0; i < good_matches.size(); i++) {
            KeyPoint prev_keypoint = keypoints1[good_matches[i].queryIdx];
            KeyPoint curr_keypoint = keypoints2[good_matches[i].trainIdx];

            int prevKey = computeKey((int) prev_keypoint.pt.x, (int) prev_keypoint.pt.y);
            int currKey = computeKey((int) curr_keypoint.pt.x, (int) curr_keypoint.pt.y);

            if(verbose) cout << (int) prev_keypoint.pt.y << " " << (int) prev_keypoint.pt.x << endl;


            int prevPointIndex = image_mappings[prevIndex][prevKey];
            if(verbose) cout << "prevKey " << prevKey << endl;
            if(verbose) cout << "image_mappings[prevIndex][prevKey] " << image_mappings[prevIndex][prevKey] << endl;
            if(verbose) cout << (image_mappings[prevIndex].find(prevKey) == image_mappings[prevIndex].end()) << endl;
            if(verbose) cout << "prevPointIndex " << prevPointIndex << endl;

            image_mapping[currKey] = prevPointIndex;
        }

        // return 0;


        // unmatched points
        for(int i = 0; i < keypoints2.size(); i++) {
            KeyPoint curr_keypoint = keypoints2[i];
            int currKey = computeKey(curr_keypoint.pt.x, curr_keypoint.pt.y);
            
            if(image_mapping.find(currKey) == image_mapping.end()) {
                image_mapping[currKey] = result_point_indices.size();
                result_point_indices.push_back(result_point_indices.size());
                cout << "writing new index to " << currKey << endl;
            }
        }

        // return 0;

        image_mappings.push_back(image_mapping);


        // draw some lines
        int lineCount = 0;
        for (int i = 0; i < good_matches.size(); i++) {
            if (verbose) {
                // cout << keypoints1[good_matches[i].queryIdx].pt << " " << keypoints2[good_matches[i].trainIdx].pt << endl;
            }
            auto prevKey = computeKey((int) keypoints1[good_matches[i].queryIdx].pt.x, (int) keypoints1[good_matches[i].queryIdx].pt.y);
            auto currKey = computeKey((int) keypoints2[good_matches[i].trainIdx].pt.x, (int) keypoints2[good_matches[i].trainIdx].pt.y);
            
            if(image_mappings[prevIndex].find(prevKey) != image_mappings[prevIndex].end() && image_mapping.find(currKey) != image_mapping.end()) {
                correspondent_indices[prevIndex].insert(prevKey);
                correspondent_indices[currIndex].insert(currKey);
                line(img2, extractKey(prevKey), extractKey(currKey), Scalar(0, 255, 0),
                    2, LINE_AA);
                lineCount++;
            }
        //     // circle()
        }
        cout << "lineCount " << lineCount << endl; 
        imshow("lines", img2);
        waitKey();

        // for(int i = 0; i < )

        // circle()
        // imshow("Lines", img2);
        // waitKey();
        // imwrite("result.png", img2);

    }

    vector<Scalar> colors {};
    for(auto pi : result_point_indices) {
        colors.push_back(Scalar(rand() % 256, rand() % 256, rand() % 256));
        cout << colors[colors.size() - 1] << endl;
    }

    for(int image_index = 1; image_index < NUM_IMAGES; image_index++) {
        int prevIndex = image_index - 1;
        int currIndex = image_index;

        // read images
        Mat img1 = readFromDataset(prevIndex);
        Mat img2 = readFromDataset(currIndex);

        for(auto const& e : image_mappings[prevIndex]) {
            auto prevKey = e.first;
            auto prevPointIndex = e.second;

            Point2f prevPos = extractKey(prevKey);
            // if(prevPointIndex == 0)
            if(correspondent_indices[prevIndex].find(prevKey) != correspondent_indices[prevIndex].end()) {
                circle(img1, prevPos, 3, colors[prevPointIndex], 3);
                putText(img1, 
                        to_string(prevPointIndex), 
                        Point2f(std::min(IMG_WIDTH, (int) prevPos.x + 3), std::max(0, (int) prevPos.y - 3)),
                        FONT_HERSHEY_SIMPLEX,
                        0.3,
                        colors[prevPointIndex],
                        2);
            }
            
            if(verbose) cout << "e.second " << prevPointIndex << endl;
            // if(verbose) cout << "prevPointIndex " << prevPointIndex << endl;
        }

        for(auto const& e : image_mappings[currIndex]) {
            auto currKey = e.first;
            auto currPointIndex = e.second;

            Point2f currPos = extractKey(currKey);
            // if(currPointIndex == 0)

                
            if(correspondent_indices[currIndex].find(currKey) != correspondent_indices[currIndex].end()) {
                circle(img2, currPos, 3, colors[currPointIndex], 3);
            
                putText(img2, 
                        to_string(currPointIndex), 
                        Point2f(std::min(IMG_WIDTH, (int) currPos.x + 3), std::max(0, (int) currPos.y - 3)),
                        FONT_HERSHEY_SIMPLEX,
                        0.3,
                        colors[currPointIndex],
                        2);
            }
            // if(verbose) {
            //     cout << "currKey " << currKey << endl;
            //     cout << "currPointIndex " << currPointIndex << endl;
            // }
        }

        Mat res;
        vconcat(img1, img2, res);
        imshow("res", res);
        imwrite("tracking_" + to_string(image_index) + ".png", res);
        waitKey();
    }

    return 0;
}