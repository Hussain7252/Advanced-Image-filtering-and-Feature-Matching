/*
Project :- Image Analysis:- Implementing Filters with Feature Detection and Matching
@ Author:- Hussain Kanchwala
@ Date  :- Start: - 01/12/24 End:- 01/23/24
*/

//This file contains implementation of filter functions declared in filter.h



#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../include/filter.h"
#include "../include/faceDetect.h"
using namespace cv;
using namespace std;

//Display the grey scale frame using a custom greyscale function
int greyscale(Mat &src, Mat &dst){

    if (src.empty()){
        cout<<"The source frame is empty";
        return -1;
    }
    dst.create(src.size(),src.type());
    for (int i=0;i<src.rows;i++){
        for (int j=0;j<src.cols;j++){
            //Substract 255 from the Blue layer of Origninal Image.
            int intensity = 255 -src.at<Vec3b>(i,j)[0];
            //Save the changed pixels in dst image
            dst.at<Vec3b>(i,j) = Vec3b(intensity,intensity,intensity);
        }
    }
    return 0;

}

//This function implements the sepina tone filter for the frame
int sepiatonefilter(Mat &src, Mat &dat){
    if (src.empty()){
        cout<<"Frame is empty";
        return -1;
    }
    dat.create(src.size(), src.type());
    CV_Assert(src.channels()==3);
    for (int i=0;i<src.rows;i++){
        for (int j=0;j<src.cols;j++){
            Vec3b pixel=src.at<Vec3b>(i,j);
            
            uchar newRed = saturate_cast<uchar>(0.272 * pixel[2] + 0.534 * pixel[1] + 0.131 * pixel[0]);
            uchar newGreen = saturate_cast<uchar>(0.349 * pixel[2] + 0.686 * pixel[1] + 0.168 * pixel[0]);
            uchar newBlue = saturate_cast<uchar>(0.393 * pixel[2] + 0.769 * pixel[1] + 0.189 * pixel[0]);
            // Update pixel values with sepia tones
            dat.at<Vec3b>(i, j) = Vec3b(newRed, newGreen, newBlue);
        }
    }
    return 0;
}

//This functions implements the Blur Filter for the frame
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    // Check if the input image is empty
    if (src.empty()) {
        return -1; // Error: Empty source image
    }

    // Create a temporary image for intermediate results
    cv::Mat temp(src.size(), src.type());
    
    // Define 1x5 filter kernel
    float kernel[] = {1, 2, 4, 2, 1};

    // Convert kernel to 1x5 matrix
    cv::Mat kernelMatrix(1, 5, CV_32F, kernel);

    // Normalize the kernel
    kernelMatrix /= cv::sum(kernelMatrix)[0];

    // Apply the horizontal filter
    cv::filter2D(src, temp, -1, kernelMatrix, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    // Apply the vertical filter
    cv::filter2D(temp, dst, -1, kernelMatrix.t(), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    return 0; 
}

// This Function implements the Sobel X filter for the frame
int sobelX3x3(Mat &src, Mat &dst){
    if (src.empty()){
        cout<<"The frame is empty"<<endl;
        return -1;
    }
    dst.create(src.size(),CV_16SC3);
    for (int y=0;y<src.rows-1;y++){
        for(int x=0;x<src.cols-1;x++){
            for (int c=0;c<3;c++){
                //Apply [1 0 -1;2 0 -2;1 0 -1] sobel filter across BGR
                short value = src.at<cv::Vec3b>(y-1,x-1)[c] - src.at<cv::Vec3b>(y-1,x+1)[c]+ 
                                2*(src.at<cv::Vec3b>(y,x-1)[c]-src.at<cv::Vec3b>(y,x+1)[c])+
                                src.at<cv::Vec3b>(y+1,x-1)[c]-src.at<cv::Vec3b>(y+1,x+1)[c];
                dst.at<cv::Vec3s>(y,x)[c]=value;
            }
        }
    }
    return 0;
}

//This Function implements the Sobel Y filter for the frame
int sobelY3x3(Mat &src, Mat &dst){
        if (src.empty()){
        cout<<"The frame is empty"<<endl;
        return -1;
    }
    dst.create(src.size(),CV_16SC3);
    for (int y=0;y<src.rows-1;y++){
        for(int x=0;x<src.cols-1;x++){
            for (int c=0;c<3;c++){
                //Apply [1 2 1;0 0;-1 -2 -1] sobel filter 
                short value = src.at<cv::Vec3b>(y-1,x-1)[c] - src.at<cv::Vec3b>(y+1,x-1)[c]+ 
                                2*(src.at<cv::Vec3b>(y-1,x)[c]-src.at<cv::Vec3b>(y+1,x)[c])+
                                src.at<cv::Vec3b>(y-1,x+1)[c]-src.at<cv::Vec3b>(y+1,x+1)[c];
                dst.at<cv::Vec3s>(y,x)[c]=value;
            }
        }
    }
    return 0;
}

//Blur QUantize Filter
int blurQuantize(Mat &src,Mat &dst,const int &level){
    if (src.empty()){
        cout<<"The frame is empty"<<endl;
        return -1;
    }
    blur5x5_2(src,src);
    for (int y=0;y<src.rows;y++){
        for(int x=0;x<src.cols;x++){
            for(int c=0;c<3;c++){
                //Divide color levels into bins
                float b = 255/level;
                //Extract Pixel color values of Image and add it to the respective bin
                float xt = src.at<cv::Vec3b>(y,x)[c]/b;
                //Take the bin value of that pixel and multiply with b (which is number of units each bin is divided into)
                float xf = round(xt)*b;
                dst.at<Vec3b>(y,x)[c]=static_cast<uchar>(xf); 
            }
        }
    }
    return 0;
}

//Face Detection 
int facedetect(Mat &src, Mat &dst){
    if (src.empty()){
        cout<<"The frame is empty";
        return -1;
    }
    Mat grey;
    vector<cv::Rect> faces;
    src.copyTo(dst);
    cv::cvtColor(src,grey,cv::COLOR_BGR2GRAY,0);
    //Call Face detection function
    detectFaces(grey,faces);
    //Draw a box around the face
    drawBoxes(dst,faces);

    return 0;
}

// Blur the background except the face
int backblur(Mat &src,Mat &dst){
    if (src.empty()){
        cout<<"The frame is empty";
        return -1;
    }
    Mat grey;
    dst.create(src.size(),src.type());
    cv::cvtColor(src,grey,cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> facer;
    //Detect face
    detectFaces(grey,facer);
    //Store the blurred version of original frame in dst
    cv::blur(src,dst,cv::Size(40,40),cv::Point(-1, -1));
    
    for (cv::Rect& face : facer) {
        // Create a region of interest (ROI) around the face and sharpening it with colors.
        for (int x=face.x;x<face.x+face.width;x++){
            for(int y = face.y;y<face.y+face.height;y++){
                Vec3b srcvec = src.at<Vec3b>(y,x);
                dst.at<Vec3b>(y,x)=srcvec;
            }
        }
        cv::rectangle(dst,face,cv::Scalar(0,255,0),3);
    }
    return 0;
}

// Color the Face and make the background Grey Scale
int face_color(Mat &src, Mat &dst){
    if (src.empty()){
        return -1;
    }
    vector<cv::Rect> faces;
    Mat grey;
    cv::cvtColor(src,grey,COLOR_BGR2GRAY);
    //DEtect faces
    detectFaces(grey,faces);
    greyscale(src,dst);
    for(cv::Rect& face:faces){
        //Creating ROI around the face and coloring the face
        for(int x=face.x;x<face.x+face.width;x++){
            for(int y=face.y;y<face.y+face.height;y++){
                Vec3b srcvec = src.at<Vec3b>(y,x);
                dst.at<Vec3b>(y,x)=srcvec;
            }
        }
        cv::rectangle(dst,face,cv::Scalar(0,255,0),3);
    }
    return 0;
}

// This function implements magnitude gradient on the frame.
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // Check if the input images have the same size and type
    if (sx.size() != sy.size() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        return -1; // Error: Invalid input images
    }

    // Create a single-channel 16-bit signed short image for magnitude
    cv::Mat magnitudeImage(sx.size(), CV_16SC3);
    cv::Mat magnitudeFloat;
    // Compute the squared values for sx and sy
    cv::Mat sxSquared, sySquared;
    cv::multiply(sx, sx, sxSquared);
    cv::multiply(sy, sy, sySquared);
    
    // Compute the gradient magnitude using Euclidean distance
    cv::add(sxSquared, sySquared, magnitudeImage);
    magnitudeImage.convertTo(magnitudeFloat, CV_32F);
    cv::sqrt(magnitudeFloat, magnitudeFloat);

    // Normalize the magnitude values 
    cv::normalize(magnitudeFloat, magnitudeFloat, 0, 255, cv::NORM_MINMAX);

    // Convert the magnitude image to 8-bit unsigned char color image for display
    cv::Mat coloredMagnitude;
    magnitudeFloat.convertTo(coloredMagnitude, CV_8U);
    cv::applyColorMap(coloredMagnitude, dst, cv::COLORMAP_JET);
    return 0; 
}

// This function brightens the face
int adjustBrightness(cv::Mat &src, Mat &dst, const int &brightness) {
    // Check if the input image is not empty
    if (src.empty()) {
        cout << "Error: Empty input image." << std::endl;
        return -1;
    }
    src.copyTo(dst);
    // Adjust brightness by adding the specified value to each pixel
    dst += brightness;

    // Clip pixel values to the valid range [0, 255]
    cv::threshold(dst, dst, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(dst, dst, 0, 0, cv::THRESH_TOZERO);

    return 0;
}

// This function implements a cartoon filter on the frame
int cartoonize(Mat &src,Mat &dst){
    if (src.empty()){
        return -1;
    }
    Mat grey;
    Mat smoothImage;
    Mat edges;
    //Convert bgr to grey
    cvtColor(src,grey,COLOR_BGR2GRAY);
    // Implement edge preserving smoothening filter called Bilateral Filter
    bilateralFilter(src,smoothImage,10,150,150);
    //Thresholding the grey scale frame
    adaptiveThreshold(grey,edges,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,9,2);
    //Convert grey scale 
    cvtColor(edges,edges,COLOR_GRAY2BGR);
    bitwise_and(smoothImage,edges,dst);


    return 0;
}

//Harris Feature corner detection
int harrisCornerDetection(Mat& inputImage, Mat& outputImage,double k ,double threshold,const int minDistance) {
    if (inputImage.empty()){
        return -1;
    }
    inputImage.copyTo(outputImage);
    Mat grayscaleImage;
    cvtColor(inputImage, grayscaleImage, COLOR_BGR2GRAY);

    Mat dx, dy;
    //Find vertical and horizontal edges using gradient.
    Sobel(grayscaleImage, dx,CV_32F, 1, 0, 3);
    Sobel(grayscaleImage, dy, CV_32F, 0, 1, 3);

    Mat dx2, dy2, dxy;
    multiply(dx, dx, dx2);
    multiply(dy, dy, dy2);
    multiply(dx, dy, dxy);
     
    //Blur so as to smoothen the sharp image and only detect the edges.
    blur(dx2, dx2, Size(50,50), Point(-1, -1));
    blur(dy2, dy2, Size(50,50), Point(-1, -1));
    blur(dxy, dxy, Size(50,50), Point(-1, -1));

    Mat harrisResponse = Mat::zeros(grayscaleImage.size(), CV_32F);
    //Vector to store detected corners.
    vector<Point> detectedCorners;
    for (int y = 0; y < grayscaleImage.rows; ++y) {
        for (int x = 0; x < grayscaleImage.cols; ++x) {
            Mat M = (Mat_<float>(2, 2) << dx2.at<float>(y, x), dxy.at<float>(y, x),
                                           dxy.at<float>(y, x), dy2.at<float>(y, x));

            float det = determinant(M);
            float trace = M.at<float>(0, 0) + M.at<float>(1, 1);
            // Calculate the response
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
    return 0;
}