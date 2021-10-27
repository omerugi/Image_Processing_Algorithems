# Image Processing Algorithms
###### *Implemented by: Lilach Mor, Omer Rugi.*

In the growing world of computer vision, there is still a lot of usage in classic *Image Processing* algorithms.
So to understand them better and how they work, we have implemented them in `Python` - We focused on the well-known and commonly used algorithms.There are three files each one with different algorithms and simple code to show an example of the results.


[Features](#features)
<a name="features"></a>

## Table of Contents:
* [How to run](#HowTo)
* [Folders:](#Folders)
 * [Ex1-](#Ex1)
   * [Loading Grayscale and RGB image](#ImageLoading)
   * [Displaying figures and images](#ImageDisplay)
   * [Transforming color space - RGB to/formYIQ](#RGB_YIQ)
   * [Intensity transformations - Histogram Equalization](#HistEq)
   * [Optimal quantization](#Quant)
   * [Gamma Correction](#Gamma)
 * [Ex2 -](#Ex2)
   * [Convolution on 1D and 2D arrays](#Conv)
   * [Image derivative and blurring](#DerevBlur)
   * [Edge Detection](#EdgeDetection)
 * [Ex3](#Ex3)
   * [Lucas Kanade - Optic Flow](#LK)
   * [Pyramids](#Pyr)



<a name = "HowTo"></a>
## How to run:
Each one of the folders contains several implementations of different algorithms - file `ex{num}_util`.
To run an example to see the results of our implantation - run file `ex{num}_main`.

<a name = "Folders"></a>
## Folders:
<a name = "Ex1"></a>
* ### Ex1:
<a name = "#ImageLoading"></a>
  * ####Loading Grayscale and RGB image:
  Implemented a function to load images from the dir using `cv2` - the function loads the images in `Grayscale` or `RGB`, based on the user, convert from `BGR` to `RGB` (if needed) and convert to NpArray.
<a name = "#ImageDisplay"></a>
  * ####Displaying figures and images:
  Implemented a function to display images in `Grayscale` or `RGB` using `plt`.
<a name = "#RGB_YIQ"></a>
  * ####Transforming color space - RGB to/formYIQ:
  Implemented a functions to convert color spaces - `RGB -> YIQ` & `YIQ -> RGB`.
  To do so we used a transformation matrix and dot produt to convert the original image to the disired color spacce.
<a name = "#HistEq"></a>
  * ####Intensity transformations - Histogram Equalization:
  Histogram equalization is a method for contrast adjustment using the image's histogram.
  The method uses only a single chanel, so in out implamantation Grayscal images would be equalized, and `RGB` images would take extra steps - First convert to `YIQ` then equalized only the `Y` channle than convert back to `RGB`.
  
  
   * [Optimal quantization](#Quant)
   * [Gamma Correction](#Gamma)
<a name = "Ex2"></a>
* ### Ex2:


<a name = "Ex3"></a>
* ### Ex3:

## Libaris used:
* Numpy.
* OpenCV.
* Matplotlib.
