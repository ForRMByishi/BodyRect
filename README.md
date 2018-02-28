# BodyRect
Python Skin color detection and image preparation for scoliosis quantification from photo images or live camera input

## Purpose
Python class for body part detection using OpenCV

## Steps
1. Initialize BodyRect with OpenCV-Image
2. BodyRect-Object stores various variables that are explained in the code header. Quality gives you the proportion of skin-masked pixels to zeros
3. Optimization of Object Rotation and Shift can be done. Make sure, there is enough space to left and right on the original image, so the algorithm can work. Otherwise it will produce a numpy error, saying that array don't fit. Image variabels are set to the new orientation. However, a backup image is stored in the Object.
4. retrieveRectDraw() returns the image, cropped to the detected part with a midline showing the presumed mirror axis
5. the retrieveSub functions yield different kinds mirror-flipped images
6. returnNormalizedXY(img) will return the mirror-flipped-subtracted histogrammoid normalized to 1.0 for X-axis and Y-axis. It depends on user input to show upper and lower border, however. I propose using C7 spinous process as upper and skin indentations at the sacroiliac joint for lower border. If anyone knows a better solution....
7. Prior to normalization it is quite useful to divide the subtracted histogrammoid into quarts at the longside. Those quarts are cumulative, starting from the mirror line. Meaning q[0] is the inner quart, q[1] the inner half, q[2] the 3/4 and q[3] the whole masked, subtracted, mirror-flipped histogrammoid. Partitioning into quarts offers different perspectives for further machine learning algorithms.

## Supplied Files
 - pic_n_save: Picks the images found in pic-origs folder, does BodyRect recognition and dumps normalized subtracted mirror-flipped scoliosis quantification data as a pickle file in a directory. User has to supply upper and lower boundaries as well as information, if image is either non-scoliotic, scoliotic or postoperative
 - load_n_plot: plots the pickle files contents with the quartile-information and does matplot
 - cam_test: test BodyRect recognition with live images
