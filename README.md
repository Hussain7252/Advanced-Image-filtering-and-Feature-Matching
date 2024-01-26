
# Implementing Filters with Feature Detection and Matching

This project covers image processing techniques from filtering to face detection, and features a custom Harris Corner Detector for feature detection and matching, using neighborhood pixel intensit.

## Authors

- [Hussain Kanchwala](https://hussainkanchwala.netlify.app/)


## Requirements

    1. OpenCV
    2. A C++ compiler such as GCC (GNU Compiler Collection) for Linux,
    3. Visual Studio Code or any other IDE (I used Visual Studio Code)


## Installation

```bash
  git clone https://github.com/Hussain7252 Advanced-Image-filtering-and-Feature-Matching
  
  cd Project_1/build

  cmake ..

  make
```

## Features
### Video Streaming Features (Run the ./vidDisplay after building the package)
- Grey Scaling and Custom grey scale (Click key 'g' and 'h' respectively)
- Sepina Tone Filter (Click key 'v')
- Gaussian Blur (Click key 'b')
- SobelX and SobelY Filter (Click key 'x' and 'Y' respectively)
- Gradient Magnitude (Click key 'm')
- Face Detection (Click key 'f')
- Selective background bluring (Click key 'a')
- Selective Color Desaturation (Click key 'c')
- Brightening the frame (Click key 'n')
- Cartooning Filter (Click key 'o')
- Save a frame from Video (Click key 's')
- Exit (Click key 'q' or ESC)

Run the ./vidDisplay to to run the executable of vidDisplay.cpp
### Feature Detection and Matching (Run ./matching after building the package)
Implemented harris corner feature detector from scratch and pixel neighbouring hood intensity based feature Matching.

Run the ./matching to to run the executable of featurematching.cpp





## Support

Feel free to reach out to me on my email kanchwala.h@northeastern.edu in case of any issues.

