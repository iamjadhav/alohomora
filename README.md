## Alohomora - Shake My Boundary
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
---
[Aditya Jadhav](https://github.com/iamjadhav)
Graduate Student of M.Eng Robotics at University of Maryland.


## Overview

One of the fundamental problems in Computer Vision is Boundary Detection. Considerable efforts have been made towards the solution of this problem with focuses on detection and localization of 
color/intensity discontinuities and dividing the image into homogeneous regions. One such approach with Contour Detection and Image Segmentation methods significantly outperforms established 
algorithms like Canny Edge and Sobel Descriptors. Making use of the Pb (Probability of Boundary) Detection algorithm I present my solution of Phase 1 - Shake My Boundary.

## Technology Used

* Python Programming Language
* Ubuntu 20.04 LTS


## License 

```
MIT License

Copyright (c) 2022 Aditya Jadhav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```

## Dependencies

* Python 3.7.9 or up
* Glob
* Cv2
* Math
* Matplotlib.pyplot
* Imutils
* Sklearn.cluster
* Numpy

## How to Run

```
git clone --recursive https://github.com/iamjadhav/alohomora.git
- Activate the environment which has all required dependencies
- cd to the Code directory inside the submission folder
- execute "python3 Wrapper.py" command
```

## Results

- Results Directory
	- Filters
	- Gradients
	- Maps
	- PB-Lite Output
