# Image Processing Algorithms
![image](https://user-images.githubusercontent.com/57361655/139246010-72c61f67-f782-459a-9647-3c87abee8fdc.png)

###### *Implemented by: Lilach Mor, Omer Rugi.*

In the growing world of computer vision, there is still a lot of usage in classic *Image Processing* algorithms.
So to understand them better and how they work, we have implemented them in `Python` - We focused on the well-known and commonly used algorithms.There are three files each one with different algorithms and simple code to show an example of the results.


[Features](#features)
<a name="features"></a>

## Table of Contents:
* [How to run](#HowTo)
* [Folders:](#Folders)
  * [Ex1:](#Ex1)
    * [Loading Grayscale and RGB image](#ImageLoading)
    * [Displaying figures and images](#ImageDisplay)
    * [Transforming color space - RGB to/formYIQ](#RGB_YIQ)
    * [Intensity transformations - Histogram Equalization](#HistEq)
    * [Optimal quantization](#Quant)
    * [Gamma Correction](#Gamma)
  * [Ex2:](#Ex2)
    * [Convolution on 1D and 2D arrays](#Conv)
    * [Image derivative blurring](#Derev)
    * [Image blurring](#Blur)
    * [Edge Detection](#EdgeDetection)
    * [Hough Circles](#Hough)
  * [Ex3:](#Ex3)
    * [Lucas Kanade - Optic Flow](#LK)
    * [Gaussian Pyramid](#GausPyr)
    * [Laplacian Pyramid](#LapPyr)



<a name = "HowTo"></a>
## How to run:
Each one of the folders contains several implementations of different algorithms - file `ex{num}_util`.
To run an example to see the results of our implantation - run file `ex{num}_main`.

<a name = "Folders"></a>
## Folders:
<a name = "Ex1"></a>
* ### Ex1:
  *  ##### Loading Grayscale and RGB image: <a name = "ImageLoading"></a>
      Implemented a function to load images from the dir using `cv2` - the function loads the images in `Grayscale` or `RGB`, based on the user, convert from `BGR` to `RGB` (if needed) and convert to NpArray.
<a name = "ImageDisplay"></a>
  * #### Displaying figures and images:
    Implemented a function to display images in `Grayscale` or `RGB` using `plt`.
<a name = "RGB_YIQ"></a>
  * #### Transforming color space - RGB to/formYIQ:
    Implemented a functions to convert color spaces - `RGB -> YIQ` & `YIQ -> RGB`.
    To do so we used a transformation matrix and dot produt to convert the original image to the disired color spacce.
<a name = "HistEq"></a>
  * #### Intensity transformations - Histogram Equalization:
    Histogram equalization is a method for contrast adjustment using the image's histogram.
    The method uses only a single chanel, so in our implamantation Grayscal images would be equalized, and `RGB` images would take extra steps - First convert to `YIQ` then equalized only the `Y` channle than convert back to `RGB`.
<a name = "Quant"></a>
  * #### Optimal quantization:
    Quantization is a lossy compression technique achieved by compressing a range of values to a single quantum value. In other words - given a numbers `nQuant`,`nIter` and the original image: first, algorithm will split the image's histogram into `nQuant` evenly with `nQuant-1` "barriers"(`z's`) , second, in each part will find the weighted average and move the barriers to the middle of the two weighted average - will repeat this step `nIter` times. In our implamantation Grayscal images would be quantized, and `RGB` images would take extra steps - First convert to `YIQ` then quantized only the `Y` channle than convert back to `RGB`.
<a name = "Gamma"></a>
  * #### Gamma Correction:
    Gamma correction or gamma is a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems. We did so but displaying a window with a sliding bar for the user to pick the gamma to use, the algorithm will do the img to the power of gamma divied by 100.


  <a name = "Ex2"></a>
* ### Ex2:
  * #### Convolution on 1D and 2D arrays: <a name = "Conv"></a>
    We implemented 1D and a 2D convolotion - Given an array and a kernal create the convolotion opperation between them. Also used zero padding to keep the size of the array.
<a name = "Derev"></a>
  * #### Image derivation:
    Calculate gradient of an image using it's `x-axis` and `y-axis` derivation images.
    The function uses convolotion with the derivation kernel on the original image to get `y-axis`, and with the kernal transpose to get  `x-axis`.
    With those two we can get the magnitude and the direction = the gradient of an image.
<a name = "Blur"></a>
  * #### Image blurring:
    Implamented Gaussian blur using our 1D and 2D convolotion functions.
<a name = "EdgeDetection"></a>
  * #### Edge Detection:
    We have implamented the well known edge detection methods - Soble, ZeroCrossing, LogZeroCrossing, and Canny.
<a name = "Hough"></a>
  * #### Hough Circles:
    The Hough transform is when usging the lines in space and thier cos angle to find lines in a image, in this case we have implamented the same idea only using circles in space and thier radius to find circles in a image.


<a name = "Ex3"></a>
* ### Ex3:
   * #### Lucas Kanade - Optic Flow: <a name = "LK"></a>
      The Lucas-Kanade optical flow algorithm is a simple technique which can provide an estimate of the movement of interesting features in successive
   images of a scene. 
      Our implamentation is reciving two following images one from time `t` and the second from time `t+1` and calculat the translation - the movment of the objects - from the first image to the second image.
<a name = "GausPyr"></a>
   * #### Gaussian Pyramid:
      Implamantation of the Gaussian pyramid - a technique in image processing that breaks down an image into successively smaller groups of pixels to blur it.
<a name = "LapPyr"></a>
   * #### Laplacian Pyramid:
     Implamantation of the Laplacian pyramid - Very similar to a Gaussian pyramid but saves the difference image of the blurred versions between each levels. Only the smallest level is not a difference image to enable reconstruction of the high resolution image using the difference images on higher levels.

## Libaris used:
* Numpy.
* OpenCV.
* Matplotlib.
