#include <iostream>
#include <filesystem>
#include <regex>
#include <type_traits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>


int main(int argc, char **argv)
{
    std::vector<cv::Mat> imageStream;
    std::vector<std::string> filenameList;
    
    // find, sort and load images
    std::regex regexPattern ("\\./traffic[0-9]*\\.png");
    for (auto const& dir_entry : std::filesystem::directory_iterator{"./"}) 
    {
        std::string filename = dir_entry.path();
        std::smatch sm;
        std::regex_match (filename,sm,regexPattern);
        if (sm.size() == 1)
        {
            filenameList.push_back(filename);
        }
    }
    sort(filenameList.begin(), filenameList.end());

    for (auto& filename : filenameList)
    {
        imageStream.push_back(cv::imread(filename));
        std::cout << "Loaded image: " << filename << std::endl;
    }

    // crop images to have only one consistent image size
    int minRow = std::min_element(
                imageStream.begin(), imageStream.end(),
                [](const cv::Mat &a, const cv::Mat &b) { return a.rows < b.rows; })->rows;
    int minCol = std::min_element(
                imageStream.begin(), imageStream.end(),
                [](const cv::Mat &a, const cv::Mat &b) { return a.cols < b.cols; })->cols;
    for (auto& image : imageStream)
    {
        image = image.rowRange(0, minRow);
        image = image.colRange(0, minCol);
    }

    // Gray scale image conversion
    std::vector<cv::Mat> imageStreamGray = imageStream;
    for (int i = 0; i < imageStream.size(); i++)
    {
        cv::cvtColor(imageStream[i], imageStreamGray[i], cv::COLOR_BGR2GRAY);
    }
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
    std::vector<std::vector<cv::Point2f>> featurePts;
    featurePts.resize(imageStream.size());
    for (int i = 0; i < imageStream.size(); i++)
    {
        goodFeaturesToTrack(imageStreamGray[i], featurePts[i], 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    }

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
    std::vector<std::vector<uchar>> status;
    status.resize(imageStream.size()-1);
    std::vector<std::vector<float>> err;
    err.resize(imageStream.size()-1);
    for (int i = 0; i < imageStream.size()-1; i++)
    {   
        cv::calcOpticalFlowPyrLK(imageStreamGray[i], imageStreamGray[i+1], featurePts[i], featurePts[i+1], status[i], err[i]);
    }

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(imageStream[imageStream.size()-1].size(), imageStream[imageStream.size()-1].type());
    // next image feature points
    std::vector<cv::Point2f> nextFeaturePts;
    // Use blueViolet as the feature marking color for display
    auto markColor = cv::Scalar(220,20,60);

    for (uint i = 1; i < imageStream.size(); i++)
    for (uint j = 0; j < featurePts[i].size(); j++)
    {
        // Select good points
        if(status[i-1][j] == 1) {
            nextFeaturePts.push_back(featurePts[i][j]);
            // draw the tracks
            cv::line(mask,featurePts[i][j], featurePts[i-1][j], markColor, 2);
            cv::circle(imageStream[i], featurePts[i][j], 5, markColor, -1);
        }
    }

    // plot
    cv::Mat img;
    cv::add(imageStream[imageStream.size()-1], mask, img);
    cv::imshow("Lucas-Kanade Optical Flow", img);
    int keyboard = cv::waitKey();

    return 0;
}