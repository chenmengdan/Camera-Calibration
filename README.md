# Camera-Calibration
Camera Calibration implementation using OpenCV in python

Unzip the .rar first to get the example images and formulas in the text.
Mengdan Chen

Based on the following tutorial:  
  [docs.opencv.org](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html)

# Goal
learn about distortions in camera, intrinsic and extrinsic parameters of camera etc.
<br>learn to find these parameters, undistort images etc.
# Introduction
Some pinhole cameras introduces a lot of distortion to images. Two major distortions are radial distortion and tangential distortion.

#### Tangential Distortion
Tangential distortion occurs because image taking lense is not aligned perfectly parallel to the imaging plane. So some areas in image may look nearer than expected. It is represented as below:

![image](formulas_image/tangential_factor.png)

#### Radial Distortion
Similarly, another distortion is the radial distortion.Due to radial distortion, straight lines will appear curved. Its effect is more as we move away from the center of image.

![Image](formulas_image/radial_factor.png)

we need to find five parameters, known as distortion coefficients given by:

![Image](formulas_image/distortion_coefficents.png)

In addition to this, we need to find a few more information, like intrinsic and extrinsic parameters of a camera. Intrinsic parameters are specific to a camera. It includes information like focal length ( fx,fy), optical centers ( cx,cy) etc. It is also called camera matrix. It depends on the camera only, so once calculated, it can be stored for future purposes. It is expressed as a 3x3 matrix:

![Image](formulas_image/camera_matrix.png)



Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system. Here the presence of w is explained by the use of homography coordinate system (and w=Z).We find some specific points in it ( square corners in chess board). We know its coordinates in real world space and we know its coordinates in image. With these data, the distortion coefficients could be solved. we need atleast 10 test patterns.

20 sample images of chess board are given. Consider just one image of a chess board. Important input datas needed for camera calibration is a set of 3D real world points and its corresponding 2D image points. 2D image points are OK which we can easily find from the image. These image points are locations where two black squares touch each other in chess boards
<br>What about the 3D points from real world space? Those images are taken from a static camera and chess boards are placed at different locations and orientations. So we need to know (X,Y,Z) values. But for simplicity, we can say chess board was kept stationary at XY plane, (so Z=0 always) and camera was moved accordingly. This consideration helps us to find only X,Y values. In this case, the results we get will be in the scale of size of chess board square.
<br>3D points are called object points and 2D image points are called image points.

## Setup
To find pattern in chess board, we use the function, cv2.findChessboardCorners(). We also need to pass what kind of pattern we are looking, like 8x8 grid, 5x5 grid etc. In this example, we use 11x12 grid.It returns the corner points and retval which will be True if pattern is obtained. These corners will be placed in an order (from left-to-right, top-to-bottom)


```python
#Import numpy, openCV environment
import numpy as np
import cv2
import glob
%matplotlib inline

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

# Define the chess board rows and columns
rows = 12
cols = 12

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
objectPoints.shape

```




    (144, 3)




```python
# Arrays to store object points and image points from all the images.
objectPointsArray = [] # 3d point in real world space
imgPointsArray = [] # 2d points in image plane.
```


```python
glob.glob('calib_example\*.tif')
```




    ['calib_example\\Image1.tif',
     'calib_example\\Image10.tif',
     'calib_example\\Image11.tif',
     'calib_example\\Image12.tif',
     'calib_example\\Image13.tif',
     'calib_example\\Image14.tif',
     'calib_example\\Image15.tif',
     'calib_example\\Image16.tif',
     'calib_example\\Image17.tif',
     'calib_example\\Image18.tif',
     'calib_example\\Image19.tif',
     'calib_example\\Image2.tif',
     'calib_example\\Image20.tif',
     'calib_example\\Image3.tif',
     'calib_example\\Image4.tif',
     'calib_example\\Image5.tif',
     'calib_example\\Image6.tif',
     'calib_example\\Image7.tif',
     'calib_example\\Image8.tif',
     'calib_example\\Image9.tif']



Runing the foloowing code returns the corner points and retval if pattern is obtained.
<br>Here is an example of the result:
![Image](example_points.png)


```python
# Loop over the image files
for path in glob.glob('calib_example\*.tif'):
    # Load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # Make sure the chess board pattern was found in the image
    if ret:
        # Refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Add the object points and the image points to the arrays
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        # Draw the corners on the image
        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
    
    # Display the image
    cv2.imshow('chess board', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### Calibration
Now we have our object points and image points we are ready to go for calibration. For that we use the function, ***cv2.calibrateCamera()***. It returns the camera matrix, distortion coefficients, rotation and translation vectors etc.


```python
# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
```


```python
#dist Coef.
dist
```




    array([[-2.47543410e-01,  7.14418145e-02, -1.83688275e-04,
            -2.74411144e-04,  1.05769974e-01]])




```python
#show the camera matrix
print(mtx)
```

    [[658.98431343   0.         302.50047925]
     [  0.         659.73448089 243.58638325]
     [  0.           0.           1.        ]]
    

### Re-projection Error
Re-projection error gives a good estimation of just how exact is the found parameters. This should be as close to zero as possible. Given the intrinsic, distortion, rotation and translation matrices, we first transform the object point to image point using ***cv2.projectPoints()***. Then we calculate the absolute norm between what we got with our transformation and the corner finding algorithm. To find the average error we calculate the arithmetical mean of the errors calculate for all the calibration images.


```python
# Print the camera calibration error
error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error: ", error / len(objectPointsArray))

```

    Total error:  0.01804429700829216
    

### Undistortion
We have got what we were trying. Now we can take an image and undistort it. OpenCV comes with two methods, we will see both. But before that, we can refine the camera matrix based on a free scaling parameter using ***cv2.getOptimalNewCameraMatrix()***. If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels. So it may even remove some pixels at image corners. If alpha=1, all pixels are retained with some extra black images. It also returns an image ROI which can be used to crop the result.

**Take a new image (Image2.tif in this case.)**


```python
# Load one of the test images
img = cv2.imread('calib_example\Image1.tif')
h, w = img.shape[:2]
img.shape
```




    (480, 640, 3)




```python
# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
```


```python
print(newCameraMtx)
```

    [[601.7623291    0.         300.82953281]
     [  0.         599.89715576 243.63463233]
     [  0.           0.           1.        ]]
    


```python
roi
```




    (9, 14, 621, 452)



#### Two methods to undistort image.
#### 1. Using cv2.undistort()

This is the shortest path. Just call the function and use ROI obtained above to crop the result.
<br>use **cv2.undistort()** to undistort the image. and compare it with the original image.


```python
# undistort
dst1 = cv2.undistort(src=img, cameraMatrix=mtx, distCoeffs=dist,newCameraMatrix=newCameraMtx)
#cv2.imwrite('calibresult.png', dst1)
cv2.imshow('chess board', np.hstack((img, dst1)))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Crop the undistorted image, and show the result


```python
# Crop the undistorted image
x, y, w, h = roi
dst1_croped = dst1 [y:y + h, x:x + w]
#dst2 = dst1 [y:y+h, x:x+w]
cv2.imshow('chess board', dst1_croped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 2. Using remapping
This is curved path. First find a mapping function from distorted image to undistorted image. Then use the remap function.


```python
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMtx, (w,h), 5)
dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow('chess board', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Crop the undistorted image, and show the result


```python
# crop the image
x, y, w, h = roi
dst2_croped = dst2[y:y+h, x:x+w]
cv2.imshow('chess board', dst2_croped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
