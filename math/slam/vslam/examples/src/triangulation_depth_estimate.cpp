#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

const int chessboardWidth = 4;
const int chessboardHeight = 6;

const int keypointDetectionGridHeight = chessboardHeight;
const int keypointDetectionGridWidth = chessboardWidth;

const int maxFeaturePerImage = chessboardWidth*chessboardHeight; 

const float goodMatchPercentage = 0.9;

const bool isUseRadiusMatch = false;

int main(){


	std::vector<cv::Mat> srcImgs ;
    srcImgs.push_back(cv::imread("chessboard1.png", 1));
    srcImgs.push_back(cv::imread("chessboard2.png", 1));
    assert(srcImgs.size() > 1);

    std::vector<cv::Mat> srcImgBaks(srcImgs.size());
    for (int i = 0; i < srcImgs.size(); i++) {
        srcImgs[i]=srcImgs[i](cv::Range(0,210), cv::Range(0,220));
        srcImgBaks[i] = srcImgs[i].clone();
    }

    std::cout << "============== Started Calibration and Undistortion ==============" << std::endl;

    cv::Size boardSize(chessboardHeight, chessboardWidth);

    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;

    std::vector<cv::Point3f> objp;
    for(int i{0}; i<chessboardWidth; i++)
    {
        for(int j{0}; j<chessboardHeight; j++)
            objp.push_back(cv::Point3f(j,i,0));
    }
    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;

    std::vector<cv::Mat> grayImgs(srcImgs.size());
    for (int i = 0; i < srcImgs.size(); i++)
    {
        cv::cvtColor(srcImgs[i],grayImgs[i],cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;

        bool found = cv::findChessboardCorners(grayImgs[i], boardSize, corners);

        if (!found){
            std::cout << "Chessboard corner not found" << std::endl;
            return -1;
        }
        else {
            cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);

            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(grayImgs[i],corners,cv::Size(11,11), cv::Size(-1,-1),criteria);
            
            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(srcImgs[i], cv::Size(chessboardHeight, chessboardWidth), 
                                        corners, found);
            
            objpoints.push_back(objp);
            imgpoints.push_back(corners);
        }
    }


    // calibrates camera and returns reprojection error
    cv::Mat cameraMatrix; // instrinsic parameters
    cv::Mat distCoeffs; // distortion coefficients
    cv::Mat R,T; // extrinsic parameters: rotation and translation
    double reprojError = cv::calibrateCamera(objpoints, imgpoints, 
                                        cv::Size(grayImgs[0].rows,grayImgs[0].cols), 
                                        cameraMatrix, distCoeffs, R, T);

    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;

    // undistort images
    cv::Mat hImage;
    std::vector<cv::Mat> undistortedImgs(srcImgs.size());
    for (int i = 0; i < srcImgs.size(); i++) {
        cv::undistort(srcImgs[i], undistortedImgs[i], cameraMatrix, distCoeffs);
        if (i > 0) {
            cv::hconcat(hImage, srcImgs[i], hImage);
        }
        else {
            hImage = srcImgs[0];
        }
    }

    // draw the image with annotated corners
    // cv::imshow("calibrated_undistorted_imgs", hImage);
    // cv::waitKey(0);

    std::cout << "============== Finished Calibration and Undistortion ==============" << std::endl;

    std::cout << "============== Started ORB Feature Matching ==============" << std::endl;

    // Variables to store keypoints and descriptors
    std::vector<std::vector<cv::KeyPoint>> keypoints(srcImgs.size());
    std::vector<cv::Mat> descriptors(srcImgs.size());

    // Detect ORB features and compute descriptors.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
    // Match features.
    std::vector<std::vector<cv::DMatch>> matches(srcImgs.size()-1);
    std::vector<std::vector<cv::DMatch>> bestMatches(srcImgs.size()-1);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    for (int i = 0; i < srcImgs.size(); i++) {
        // for local area detection only
        int localAreaHeight = srcImgs[i].rows / keypointDetectionGridHeight;
        int localAreaWidth = srcImgs[i].cols / keypointDetectionGridWidth;
        for (int row = 0; row < keypointDetectionGridHeight; row++ ) {
            for (int col = 0; col < keypointDetectionGridWidth; col++) {
                if (i > 0) {
                    cv::Mat mask1 = cv::Mat::zeros(grayImgs[i-1].size(), grayImgs[i-1].type());  
                    cv::Mat mask2 = cv::Mat::zeros(grayImgs[i].size(), grayImgs[i].type());      
                    cv::Mat roi1(mask1, cv::Rect(col*localAreaWidth,row*localAreaHeight,
                                localAreaWidth, localAreaHeight));
                    cv::Mat roi2(mask2, cv::Rect(col*localAreaWidth,row*localAreaHeight,
                                localAreaWidth, localAreaHeight));
                    roi1 = cv::Scalar(255); 
                    roi2 = cv::Scalar(255);
                    std::vector<cv::KeyPoint> localKeypoints1, localKeypoints2;
                    cv::Mat localDescriptors1, localDescriptors2;
                    orb->detectAndCompute(grayImgs[i-1], mask1, localKeypoints1, localDescriptors1);
                    orb->detectAndCompute(grayImgs[i], mask2, localKeypoints2, localDescriptors2);

                    std::vector<cv::DMatch> matches_tmp;

                    if ( !localKeypoints1.empty() && !localKeypoints2.empty() ) {
                        matcher->match(localDescriptors1, localDescriptors2, matches_tmp);
                        for (auto& match : matches_tmp) {
                            match.queryIdx += matches[i-1].size();
                            match.trainIdx += matches[i-1].size();
                        }
                        std::copy(matches_tmp.begin(), matches_tmp.end(), std::back_inserter(matches[i-1]));
                        
                        if (i == 1)
                            std::copy(localKeypoints1.begin(), localKeypoints1.end(), std::back_inserter(keypoints[0]));

                        std::copy(localKeypoints2.begin(), localKeypoints2.end(), std::back_inserter(keypoints[i]));
                    }
                }
            }
        }
        if ( i > 0 )
            bestMatches[i-1] = matches[i-1];
    }

    for (int i = 0; i < srcImgs.size()-1; i++) {
        std::cout << "The transform between the " << i << "-th and the " << i+1
                << "-th images has " << bestMatches[i].size() << " matches" << std::endl;
    }

    // Draw top matches
    std::vector<cv::Mat> imMatches(srcImgs.size()-1);
    for (int i = 0; i < srcImgs.size(); i++) {
        if (i>0) {
           cv::drawMatches(srcImgBaks[i-1], keypoints[i-1], srcImgBaks[i], keypoints[i], bestMatches[i-1], imMatches[i-1]);
        }
    }
    // cv::imshow("picture of matching by ORB", imMatches[0]);
    // cv::waitKey(0);

    std::cout << "============== Finished ORB Feature Matching ==============" << std::endl;

    std::cout << "============== Started Fundamental and Homography Matrix Computation ==============" << std::endl;

    for (int i = 0; i < srcImgs.size()-1; i++) {

        // cv::Mat keypointMat1(bestMatches[i].size(), 2);
        // cv::Mat keypointMat2(bestMatches[i].size(), 2);
        // for (int j = 0; j < bestMatches[i].size(); j++) {
        //     bestMatches[i][j].
        // }

        // cv::Mat H = cv::findHomography(keypoints[i], keypoints[i+1]);
        // cv::Mat F = cv::findFundamentalMat(keypoints[i], keypoints[i+1]);

        // std::cout << "F:\n" << F << std::endl;
        // std::cout << "H:\n" << H << std::endl;
    }

    std::cout << "============== Finishes Fundamental and Homography Matrix Computation ==============" << std::endl;

    std::cout << "============== Started Triangulation Computation ==============" << std::endl;

    std::cout << "============== Finished Triangulation Computation ==============" << std::endl;


    return 0;
}

