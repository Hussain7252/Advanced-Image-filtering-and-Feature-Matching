/*
Project :- Video Special Effects
@ Author:- Hussain Kanchwala
@ Date  :- Start: - 01/12/24 End:- 01/23/24
*/

//This is a header file containing different filter functions.
#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// Grey Scale
int greyscale(Mat &src, Mat &dst);
// Sepina Tone Filter
int sepiatonefilter(Mat &src, Mat &dat);
// Blur Filter
int blur5x5_2(Mat &src,Mat &dat);
//Sobel X Filter
int sobelX3x3(Mat &src, Mat &dst);
//Sobel Y Filter
int sobelY3x3(Mat &src, Mat &dst);
//Blur Quantize Filter
int blurQuantize(Mat &src,Mat &dst,const int &level);
//Face Detection 
int facedetect(Mat &src,Mat &dat);
//Face Detection with background out of focus
int backblur(Mat &src,Mat &dst);
//Grey Scale background with rgb in face ROI
int face_color(Mat &src, Mat &dst);
//GRadient Magnitude filter
int magnitude(Mat &sx, Mat &sy,Mat &dst);
//Brightness Filter
int adjustBrightness(Mat &src, Mat &dst, const int &brightness);
//Cartoon Filter
int cartoonize(Mat &src,Mat &dst);
//Harris Corner Filter
//Make sure this requires a lot of computational power hence for high resolution images the laptop crashes
int harrisCornerDetection(Mat& inputImage,Mat& outputImage,double k = 0.04,double threshold = 100,const int minDistance=10);