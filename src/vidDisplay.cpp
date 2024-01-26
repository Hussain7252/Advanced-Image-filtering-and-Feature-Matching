/*
Project :- Image Analysis:- Implementing Filters with Feature Detection and Matching
@ Author:- Hussain Kanchwala
@ Date  :- Start: - 01/12/24 End:- 01/23/24
*/

// This contains the code to run the respective filters on user command.

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../include/filter.h"
#include "../include/faceDetect.h"
using namespace std;
using namespace cv;

// Turns on your device default camera
void video_turnon() {
    cout<<"The key codes for different filters are: -"<<endl;
    cout<<"Press 'g':- Grey Scale"<<endl;
    cout<<"Press 'h':- Custom Grey Scale"<<endl;
    cout<<"Press 'v':- Sepina tone Filter"<<endl;
    cout<<"Press 'b':- Blur Filter"<<endl;
    cout<<"Press 'x':- Sobel X Filter"<<endl;
    cout<<"Press 'y':- Sobel Y Filter"<<endl;
    cout<<"Press 'm':- Gradient Magnitude Filter"<<endl;
    cout<<"Press 'i':- Blur and Quantize Filter"<<endl;
    cout<<"Press 'f':- Face Detection"<<endl;
    cout<<"Press 'a':- Blur the background except the face"<<endl;
    cout<<"Press 'c':- Grey the background except the face"<<endl;
    cout<<"Press 'n':- Increase the Brightness of the frame"<<endl;
    cout<<"Press 'o':- Cartoonize the frame"<<endl;
    cout<<"Press 'e':- Harris feature Corner detection"<<endl;
    VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        cout<<"Unable to open video device"<<"\n";
    }

    int frameWidth = capdev.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = capdev.get(CAP_PROP_FRAME_HEIGHT);
    double fps = capdev.get(CAP_PROP_FPS);
    int totalFrames = capdev.get(CAP_PROP_FRAME_COUNT);
    std::cout << "Video Details:" << std::endl;
    std::cout << "Frame Width: " << frameWidth << std::endl;
    std::cout << "Frame Height: " << frameHeight << std::endl;
    std::cout << "Frames Per Second (fps): " << fps << std::endl;
    std::cout << "Total Number of Frames: " << totalFrames << std::endl;
    namedWindow("Video", 1); // identifies a window
    Mat frame;
    Mat currentframe;
    int brightness;
    char currentkey = 'p';
    while (true) {
        capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout<<"frame is empty"<<"\n";
            break;
        }
        if (currentkey=='g'){
            cvtColor(frame,currentframe,COLOR_BGR2GRAY);
        }
        else if(currentkey =='h'){
            greyscale(frame,currentframe);
        }
        else if (currentkey == 'p'){
            currentframe = frame;
        }else if (currentkey == 'v'){
            sepiatonefilter(frame,currentframe);
        }else if (currentkey == 'b'){
            blur5x5_2(frame,currentframe);
        }else if (currentkey =='x'){
            sobelX3x3(frame,currentframe);
            cv::convertScaleAbs(currentframe,currentframe);
        }else if (currentkey =='y'){
            sobelY3x3(frame,currentframe);
            cv::convertScaleAbs(currentframe,currentframe);
        }else if (currentkey=='i'){
            blurQuantize(frame,currentframe,8);
        }else if (currentkey == 'f'){
            facedetect(frame,currentframe);
        }else if (currentkey == 'a'){
            backblur(frame,currentframe);
        }else if (currentkey == 'c'){
            face_color(frame,currentframe);
        }else if (currentkey == 'm'){
            Mat sobelx; Mat sobely;
            sobelX3x3(frame,sobelx);
            sobelY3x3(frame,sobely);
            magnitude(sobelx,sobely,currentframe);
        }else if (currentkey == 'n'){
            adjustBrightness(frame, currentframe,brightness);
        }else if (currentkey == 'o'){
            cartoonize(frame,currentframe);
        }else if (currentkey == 'e'){
            harrisCornerDetection(frame,currentframe,0.06,150,10);
        }

        imshow("Video",currentframe);
        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q'){
            break;
        }else if (key == 'G' || key =='g'){ // Grey Scale by OpenCV
            currentkey = 'g';
        }else if (key == 'p' || key == 'P'){ // Normal Camera
            currentkey ='p';
        }else if (key == 'S' || key=='s'){ // Saving the Image
            string path;
            cout<<"Enter the path to save the image";
            getline(cin,path);
            imwrite(path, currentframe);
            cout << "Frame saved" << endl;
        }else if (key=='h' || key=='H'){ // Special Type of Grey Scale
            currentkey = 'h';
        }else if (key == 'v' || key == 'V'){ // Applying Sepia Tone FIlter
            currentkey = 'v';
        }else if(key == 'b' || key == 'B'){ // Applying blur Filter
            currentkey = 'b';
        }else if (key=='x'||key=='X'){ // Edge Filter X giving Vetical Edges
            currentkey = 'x';
        }else if (key=='y'||key=='Y'){ // Edge Filter Y giving Horizontal Edges
            currentkey = 'y';
        }else if(key=='I' || key =='i'){ //BlurQuantize 
            currentkey = 'i';
        }else if (key == 'f'||key == 'F'){ // Face Detection
            currentkey = 'f';
        }else if (key == 'a' || key == 'A'){ //Blur Outside found face
            currentkey = 'a';
        }else if (key == 'c' || key == 'C'){ // Grey Outside found face
            currentkey = 'c';
        }else if (key == 'm' || key == 'M'){ // Gradient Magnitude of the frame
            currentkey = 'm';
        }else if (key == 'n' ||key =='N'){ // Increase the brightness Value
            cout<<"Enter Brightness value";
            cin>>brightness;
            currentkey = 'n';
        }else if(key =='o' || key =='O'){ //Cartoonize the frame
            currentkey = 'o';
        }else if (key =='e'|| key=='E'){ //Harris feature detection on frame
            currentkey = 'e';
        }
    }
    capdev.release();
    destroyAllWindows();
}

int main() {
    video_turnon();
    return 0;
}