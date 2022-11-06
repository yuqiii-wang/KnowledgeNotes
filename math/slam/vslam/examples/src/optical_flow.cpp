#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>



int main(int argc, char **argv)
{
    auto image1 = cv::imread("street_view_left.png");
    auto image2 = cv::imread("street_view_right.png");    
    
    // Gray scale image conversion
    cv::Mat image1Gray, image2Gray;
    cv::cvtColor(image1, image1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, image2Gray, cv::COLOR_BGR2GRAY);

    // The function finds the most prominent corners in the image or in the specified image region
    // void cv::goodFeaturesToTrack	(	InputArray 	image,
    // OutputArray 	corners,
    // int 	maxCorners,
    // double 	qualityLevel,
    // double 	minDistance,
    // InputArray 	mask = noArray(),
    // int 	blockSize = 3,
    // bool 	useHarrisDetector = false,
    // double 	k = 0.04 
    // )
    std::vector<cv::Point2f> featurePts1, featurePts2;
    goodFeaturesToTrack(image1Gray, featurePts1, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

    // calculate optical flow
    // void cv::calcOpticalFlowPyrLK(	InputArray 	prevImg,
    // InputArray 	nextImg,
    // InputArray 	prevPts,
    // InputOutputArray 	nextPts,
    // OutputArray 	status,
    // OutputArray 	err,
    // Size 	winSize = Size(21, 21),
    // int 	maxLevel = 3,
    // TermCriteria 	criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
    // int 	flags = 0,
    // double 	minEigThreshold = 1e-4 
    // )	
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
    cv::calcOpticalFlowPyrLK(image2Gray, image2Gray, featurePts1, featurePts2, status, err, cv::Size(15,15), 2, criteria);

    return 0;
}