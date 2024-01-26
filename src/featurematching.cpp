
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

int main() {
    // Load reference and target images
    string pathr;
    string patht;
    cout<<"Enter reference Image path"<<endl;
    getline(cin,pathr);
    cout<<"Enter target Image path"<<endl;
    getline(cin,patht);
    Mat referenceImage = imread(pathr);
    Mat targetImage = imread(patht);

    // Check if images are loaded successfully
    if (referenceImage.empty() || targetImage.empty()|| referenceImage.size() != targetImage.size()) {
        cerr << "Error: Unable to load images." << endl;
        return -1;
    }


    // Perform Harris corner detection on the reference image
    //30 Distance for matching.jpeg and 10 for checker board
    Mat outputImage;
    vector<Point> referenceCorners = harris(referenceImage, outputImage,0.06,150,30);
    imshow("COrners",outputImage);
    waitKey(0);
    string path_to_save;
    cout<<"Enter path to save Image with corners"<<endl;
    getline(cin,path_to_save);
    imwrite(path_to_save, outputImage);
    destroyAllWindows();

    // Perform scan matching
    
    vector<Point> matchedCorners;
    scanMatching(referenceImage, targetImage, referenceCorners, matchedCorners,100);
    
    return 0;
}
