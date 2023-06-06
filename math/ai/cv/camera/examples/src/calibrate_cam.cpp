#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

const int chessboardWidth = 4;
const int chessboardHeight = 6;

const int keypointDetectionGridHeight = chessboardHeight;
const int keypointDetectionGridWidth = chessboardWidth;

const int maxFeaturePerImage = chessboardWidth*chessboardHeight; 

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
    double reprojError = 0; // Root Mean Square Error
    reprojError = cv::calibrateCamera(objpoints, imgpoints, 
                                        cv::Size(grayImgs[0].rows,grayImgs[0].cols), 
                                        cameraMatrix, distCoeffs, R, T);

    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;
    std::cout << "Reprojection Error : " << reprojError << std::endl;

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
    cv::imshow("calibrated_undistorted_imgs", hImage);
    cv::waitKey(0);

    std::cout << "============== Re-project the 3d points to 2d by Calibrated Camera Intrinsics ==============" << std::endl;

    std::vector<std::vector<cv::Point2f>> objrps(srcImgs.size());
    cv::Mat hImageReproj;

    for (int i = 0; i < srcImgs.size(); i++) {
        cv::projectPoints(objpoints[i], R.row(i), T.row(i), cameraMatrix, distCoeffs, objrps[i]);
        cv::undistort(srcImgBaks[i], undistortedImgs[i], cameraMatrix, distCoeffs);

        for (auto& pt : objrps[i]) {
            cv::circle(undistortedImgs[i], pt, 3, cv::Scalar(0,255,0), -1);
        }
    
        if (i > 0) {
            cv::hconcat(hImageReproj, undistortedImgs[i], hImageReproj);
        }
        else {
            hImageReproj = undistortedImgs[0];
        }
    }
    

    cv::imshow("reprojected_img", hImageReproj);
    cv::waitKey(0);

    return 0;
}