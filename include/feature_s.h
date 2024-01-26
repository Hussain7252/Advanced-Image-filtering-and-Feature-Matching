#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// Harris feature Description
vector<Point> harris(Mat& inputImage, Mat& outputImage,double k ,double threshold,int minDistance);
// Feature description and Matching
void scanMatching(Mat& referenceImage, Mat& targetImage, vector<Point>& referenceCorners, vector<Point>& matchedCorners,int maxMatches = 100); 
