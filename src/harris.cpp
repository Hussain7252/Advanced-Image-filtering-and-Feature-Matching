/*
Project :- Image Analysis:- Implementing Filters with Feature Detection and Matching
@ Author:- Hussain Kanchwala
@ Date  :- Start: - 01/12/24 End:- 01/23/24
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../include/feature_s.h"
using namespace cv;
using namespace std;

//Harris Feature corner detection
vector<Point> harris(Mat& inputImage, Mat& outputImage,double k ,double threshold,int minDistance) {
    inputImage.copyTo(outputImage);
    Mat grayscaleImage;
    cvtColor(inputImage, grayscaleImage, COLOR_BGR2GRAY);

    Mat dx, dy;
    Sobel(grayscaleImage, dx,CV_32F, 1, 0, 3);
    Sobel(grayscaleImage, dy, CV_32F, 0, 1, 3);

    Mat dx2, dy2, dxy;
    multiply(dx, dx, dx2);
    multiply(dy, dy, dy2);
    multiply(dx, dy, dxy);

    blur(dx2, dx2, Size(2,2), Point(-1, -1));
    blur(dy2, dy2, Size(2,2), Point(-1, -1));
    blur(dxy, dxy, Size(2,2), Point(-1, -1));

    Mat harrisResponse = Mat::zeros(grayscaleImage.size(), CV_32F);
    vector<Point> detectedCorners;
    for (int y = 0; y < grayscaleImage.rows; ++y) {
        for (int x = 0; x < grayscaleImage.cols; ++x) {
            Mat M = (Mat_<float>(2, 2) << dx2.at<float>(y, x), dxy.at<float>(y, x),
                                           dxy.at<float>(y, x), dy2.at<float>(y, x));

            float det = determinant(M);
            float trace = M.at<float>(0, 0) + M.at<float>(1, 1);

            float response = det - k * trace * trace;

            if (response > threshold) {
                Point currentCorner(x, y);
                bool isFarEnough = true;

                for (const Point& corner : detectedCorners) {
                    double distance = std::sqrt(std::pow(currentCorner.x - corner.x, 2) +
                                                std::pow(currentCorner.y - corner.y, 2));

                    if (distance < minDistance) {
                        isFarEnough = false;
                        break;
                    }
                }
                if (isFarEnough){
                    circle(outputImage, Point(x, y), 2, Scalar(0, 0, 255), 2);
                    detectedCorners.push_back(currentCorner);
                }
            }
        }
    }
    return detectedCorners;
}

void scanMatching(Mat& referenceImage, Mat& targetImage, vector<Point>& referenceCorners, vector<Point>& matchedCorners,int maxMatches) {
    // Define a region around each corner for intensity comparison
    const int windowSize = 5;

    // Create separate images for visualization
    Mat resultReference = referenceImage.clone();
    Mat resultTarget = targetImage.clone();

    // Iterate through Harris corners in the reference image
    for (size_t i = 0; i<min(referenceCorners.size(), static_cast<size_t>(maxMatches)); ++i) {
        Point refCorner = referenceCorners[i];
        int refX = refCorner.x;
        int refY = refCorner.y;

        // Extract intensity window around the reference corner
        Mat refWindow;
        try {
            refWindow = referenceImage(Rect(refX - windowSize / 2, refY - windowSize / 2, windowSize, windowSize)).clone();
        } catch (const cv::Exception& e) {
            cerr << "Error: Unable to extract intensity window around the reference corner." << endl;
            cerr << e.what() << endl;
            continue; // Skip to the next corner in case of an error
        }

        double bestMatchScore = numeric_limits<double>::max();
        Point bestMatch;

        // Search for the best matching window in the target image
        for (int y = windowSize / 2; y < targetImage.rows - windowSize / 2; ++y) {
            for (int x = windowSize / 2; x < targetImage.cols - windowSize / 2; ++x) {
                // Extract intensity window around the current point in the target image
                Mat targetWindow = targetImage(Rect(x - windowSize / 2, y - windowSize / 2, windowSize, windowSize));

                // Compute sum of absolute intensity differences between windows
                double score = norm(refWindow, targetWindow, NORM_L1);

                // Update the best match if the current score is lower
                if (score < bestMatchScore) {
                    bestMatchScore = score;
                    bestMatch = Point(x, y);
                }
            }
        }

        // Add the best match to the list of matched corners
        matchedCorners.push_back(bestMatch);
                // Visualize the matched corners by drawing circles
        circle(resultReference, refCorner, 3, Scalar(0, 255, 0), 2);
        circle(resultTarget, bestMatch, 3, Scalar(0, 0, 255), 2);

        // Annotate with text
        string text = "Match " + to_string(i);
        putText(resultReference, text, refCorner, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        putText(resultTarget, text, bestMatch, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    }

    // Combine the visualized images horizontally for display
    Mat combinedResult;
    hconcat(resultReference, resultTarget, combinedResult);

    // Display the result
    imshow("Scan Matching Result", combinedResult);
    waitKey(0);
    string path_save;
    cout<<"ENTER PATH TO SAVE THE IMAGE"<<endl;
    getline(cin,path_save);
    imwrite(path_save,combinedResult);
    destroyAllWindows();
}

/*
int main(){
        string path;
        cout<<"Enter Path to Image if you don't want to then a default image will openup"<<endl;
        getline(cin,path);
        cout<<"The path entered is: "<<path<<endl;
        Mat src;
        if ((path.empty())){
            return -1;
        }else{
            src  = imread(path);
        }
        Mat org;
        harris(src,org,0.06,150,10);
        imshow("Corner_detection",org);
        waitKey(0);
        string outpath;
        cout<<"Enter path to save the corner filter image"<<endl;
        getline(cin,outpath);
        cout<<"The path entered is: "<<path<<endl;
        imwrite(outpath, org);
        destroyAllWindows();
        return 0;       

}
*/