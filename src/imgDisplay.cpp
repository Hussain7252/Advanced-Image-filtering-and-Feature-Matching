/*
Project :- Image Analysis:- Implementing Filters with Feature Detection and Matching
@ Author:- Hussain Kanchwala
@ Date  :- 01/12/24
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Read and Display Image

    void read_image(const string& path = "/home/hussain/computer_vision/CourseWork/Project_1/media/building.jpg"){
        Mat image;
        image = imread(path);
        if (image.empty()){
            cout<<"Check the path";
        }
        string windowname="Display";
        namedWindow(windowname, WINDOW_NORMAL);
        while (true){
            int key = waitKey(1);
            imshow(windowname,image);
            if(key == 'q' || key == 27){
                break;
            }

        }
        destroyAllWindows();
    }


//  Main Function is calling the read_image function
    int main() 
    { 
        string path;
        cout<<"Enter Path to Image if you don't want to then a default image will openup";
        getline(cin,path);
        cout<<"The path entered is: "<<path;
        if ((path.empty())){
            return -1;
        }else{
            read_image();
        }
        return 0; 
    }