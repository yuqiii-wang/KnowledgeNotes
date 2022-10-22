#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

int main() 
{
	cv::Mat imageL = cv::imread("lenna.png");
	cv::Mat imageR = cv::imread("lenna.png");

	cv::flip(imageR, imageR, 1);

	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	std::vector<cv::KeyPoint> keyPointL, keyPointR;

	sift->detect(imageL, keyPointL);
	sift->detect(imageR, keyPointR);

	// Drawing keypoints
	cv::Mat keyPointImageL;
	cv::Mat keyPointImageR;
	drawKeypoints(imageL, keyPointL, keyPointImageL, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(imageR, keyPointR, keyPointImageR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::namedWindow("KeyPoints of imageL");
	cv::namedWindow("KeyPoints of imageR");

	cv::imshow("KeyPoints of imageL", keyPointImageL);
	cv::imshow("KeyPoints of imageR", keyPointImageR);

	cv::Mat despL, despR;
	// Compute keypoint descriptors
	sift->detectAndCompute(imageL, cv::Mat(), keyPointL, despL);
	sift->detectAndCompute(imageR, cv::Mat(), keyPointR, despR);

	//Struct for DMatch: query descriptor index, train descriptor index, train image index and distance between descriptors.
	//int queryIdx : test image descriptor
	//int trainIdx : sample image descriptor
	//int imgIdx : used when having multiple sample images
	//float distance Euclidean distance of two descriptors
	std::vector<cv::DMatch> matches;

	// data type conversion before using FlannBased method
	if (despL.type() != CV_32F || despR.type() != CV_32F)
	{
		despL.convertTo(despL, CV_32F);
		despR.convertTo(despR, CV_32F);
	}

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	matcher->match(despL, despR, matches);

	// find max dis 
	double maxDist = 0; 
	for (int i = 0; i < despL.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist > maxDist) 
			maxDist = dist;
	}

	// select matching points
	std::vector< cv::DMatch > good_matches;
	for (int i = 0; i < despL.rows; i++)
	{
		if (matches[i].distance < 0.5*maxDist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	cv::Mat imageOutput;
	cv::drawMatches(imageL, keyPointL, imageR, keyPointR, good_matches, imageOutput);

	cv::namedWindow("picture of matching");
	cv::imshow("picture of matching  by SIFT", imageOutput);

	cv::waitKey(0);

	return 0;
}

