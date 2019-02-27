# Image Stitching
## Installation
`Scikit-Image` and `Annoy` are required for functions corner detection and feature matching. Install them by running the following lines in the terminal.
```
pip3 install scikit-image
pip3 install annoy
```
## Running the code
`demo_script.py` is the wrapper for the codebase. Put the test images in test_img folder and change the filename in demo_script.py. The code to perform stitching for the files can be run by running the following command in the terminal.
```
python3 demo_script.py
```

## Parameters that can be changed
There are many parameters that can be tuned to get optimal output.
* **Thresholding corner-metric matrix**
This threshold can be modified in `anms.py` at the top. Just modify the variable named thresh.
* **Maximum number of points from ANMS**
The maximum number of points that anms has to return must be passed to the function anms.py. This can be modified in `helper.py` line numbers 42, 44 and 46.

## Blending Images
The blending of images after warping the image using homography can be done in two ways - Gradient Domain Blending or Alpha Blending.
This mode can be set in `mymosaic.py` in line number 30. The variable mode is to be modified to 0 (or ALPHA_BLENDING) or 1 (or GRADIENT_BLENDING).
