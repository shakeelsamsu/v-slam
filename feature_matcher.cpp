#include <iostream>
#include <dirent.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
// using namespace cv::xfeatures2d;
using namespace std;

const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }"
    "{ v verbose |               | Use of verbose flag. }";

const char* dataset_path = "dataset/sequences/00/image_0/";

bool verbose = false;

Mat readFromDataset(int num) {
    string filename = to_string(num);
    filename.insert(0, 6 - min(6, (int) filename.size()), '0');
    string path(dataset_path);
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

int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );
    verbose = parser.get<bool>("verbose");


    for(int i = 0; i < 1000; i++) {
        // read images
        Mat img1 = readFromDataset(i);
        Mat img2 = readFromDataset(i + 1);

        if ( img1.empty() || img2.empty() )
        {
            cout << "Could not open or find the image!\n" << endl;
            parser.printMessage();
            return -1;
        }

        // get feature points and compute descriptors
        int minHessian = 400;
        Ptr<ORB> detector = ORB::create( minHessian );
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        twoFrameDetectAndCompute(img1, img2, detector, keypoints1, keypoints2, descriptors1, descriptors2);

        // match
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        std::vector<DMatch> good_matches = match(matcher, descriptors1, descriptors2);

        // draw some lines
        for (int i = 0; i < good_matches.size(); i++) {
            if (verbose) 
                cout << keypoints1[good_matches[i].queryIdx].pt << " " << keypoints2[good_matches[i].trainIdx].pt << endl;
            line(img2, keypoints1[good_matches[i].queryIdx].pt, keypoints2[good_matches[i].trainIdx].pt, Scalar(0, 255, 0),
                2, LINE_AA);
        }
        imshow("Lines", img2);
        waitKey(10);
        // imwrite("result.png", img2);

    }

    return 0;
}