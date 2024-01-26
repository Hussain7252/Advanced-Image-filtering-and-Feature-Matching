/*
  Author:- Brue A. Maxwell
  Modified By: - Hussain Kanchwala

  Example of how to time an image processing task.

  Program takes a path to an image on the command line
*/

#include <cmath>
#include <sys/time.h>
#include "opencv2/opencv.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

// prototypes for the functions to test
int blur5x5_1( cv::Mat &src, cv::Mat &dst ){
  if (src.empty()){
    cout<<"The frame is empty"<<endl;
    return -1;
  }
  int kernel[5][5]= {{1,2,4,2,1},{2,4,8,4,2},{4,8,16,8,4},{2,4,8,4,2},{1,2,4,2,1}};
  int kernelsum = 0;
  for (int n = 0;n<5;n++){
    for (int m=0;m<5;m++){
        kernelsum += kernel[n][m];
    }
  }
  dst.create(src.size(),src.type());
  for(int i=2;i<src.rows-2;i++){
    for(int j=2;j<src.cols-2;j++){
      int blue =0,green =0,red=0;
      
      for (int k=-2;k<=2;k++){
        for(int l=-2;l<=2;l++){
          Vec3b pixel = src.at<Vec3b>(i+k,j+l);

          blue += pixel[0] * kernel[k+2][l+2];
          green += pixel[1] * kernel[k+2][l+2];
          red += pixel[2] * kernel[k+2][l+2];          
        }
      }
    dst.at<Vec3b>(i,j) = Vec3b(blue/kernelsum, green/kernelsum, red/kernelsum);
    }
  }
  return 0;
}


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

    return 0; // Success
}


// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;
  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}
  

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

int main(int argc, char *argv[]) {  // main function, execution starts here
  cv::Mat src; // define a Mat data type (matrix/image), allocates a header, image data is null
  cv::Mat dst; // cv::Mat to hold the output of the process

  string filename;
  cout<<"Enter Path to Image if you don't want to then a default image will openup";
  getline(cin,filename);
  cout<<"The path entered is: "<<filename;

  // read the image
  src = cv::imread(filename); // allocating the image data
  // test if the read was successful
  if(src.data == NULL) {  // src.data is the reference to the image data
    cout<<"Unable to read image";
    return -1;
  }

  const int Ntimes = 10;
	
  //////////////////////////////
  // set up the timing for version 1
  double startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_1( src, dst );
  }

  // end the timing
  double endTime = getTime();

  // compute the time per image
  double difference = (endTime - startTime) / Ntimes;

  // print the results
  cout << "Time per image (1): " << difference << " seconds" << endl;

  //////////////////////////////
  // set up the timing for version 2
  startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_2( src, dst );
  }

  // end the timing
  endTime = getTime();

  // compute the time per image
  difference = (endTime - startTime) / Ntimes;
  
  // print the results
  cout << "Time per image (2): " << difference << " seconds" << endl;
  
  cv::imshow("Blur",dst);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return(0);
}
