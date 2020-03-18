# import the necessary packages
import numpy as np
import argparse
import cv2
from skimage.util import img_as_ubyte
from skimage import io
from glob import glob
import os
import time

title_window = 'Linear Blend'

global trackbar_name1
global trackbar_name2

global thresh1_double_binary
global thresh2_double_binary
global thresh1_binary
global thresh2_binary
global i




def on_trackbar_canny_1(val):
    global thresh1_double_binary
    thresh1_double_binary=val
    redraw_edges()

def on_trackbar_canny_2(val):
    global thresh2_double_binary
    thresh2_double_binary=val
    redraw_edges()

def on_trackbar_binary_1(val):
    global thresh1_binary
    thresh1_binary=val
    redraw_edges()

def on_trackbar_binary_2(val):
    global thresh2_binary
    thresh2_binary=val
    redraw_edges()


def redraw_edges(found_circles=0):
    circles=[]
    zstack = io.imread(overlayed_image[i])
    output = img_as_ubyte(zstack[:, :, 0]).copy()
    edged = image.copy()
    # ret, thresh = cv2.threshold(image, thresh1_binary, thresh2_binary, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(image, thresh1_binary, thresh2_binary, cv2.THRESH_TOZERO_INV)
    ret, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)
    print("bw:",thresh1_binary, thresh2_binary)
    edged = cv2.Canny(thresh, 101, 112)
    print("edged:", thresh1_double_binary, thresh2_double_binary)
    circles = cv2.HoughCircles(image=edged, method=cv2.HOUGH_GRADIENT, dp=3, minDist=400, minRadius=475, maxRadius=495)
    if circles is not None:
        found_circles += 1
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (255, 0, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # else:
    #     print(image_path)

    cv2.imshow(title_window, np.hstack([output, edged, thresh]))
        # show the output image
    # global contours
    # global hierarchy
    # contours, hierarchy = cv2.findContours(edged,
    #                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Path to the dir of images overlay")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
found_circles = 0

thresh1_binary = 255
thresh1_double_binary = 255
thresh2_binary = 255
thresh2_double_binary = 255

# overlay_files = list(sorted(glob(args["dir"])))

dropimages = list(sorted(glob(os.path.join(args["dir"],"organizedWells","wellNum_*")), key=lambda s: int(s.split('wellNum_')[1])))
overlayed_image = list(sorted(glob(os.path.join(args["dir"],"overlayed","*.jpg"))))


for wellNum, drop in enumerate(dropimages):
    image_list = sorted(glob(os.path.join(drop,"*")))
    if len(image_list) == 3:
        image_path = image_list[2]
    else:
        image_path = image_list[0]

    zstack = io.imread(image_path)
    image = img_as_ubyte(zstack[:, :, 0])  # finds the top x-y pixels in the z-stack
    print(image_path)
    output = image.copy()
    cv2.namedWindow(title_window)

    cv2.createTrackbar('thresh1 canny %d' % thresh1_double_binary, title_window, 0, thresh1_double_binary, on_trackbar_canny_1)
    cv2.createTrackbar('thresh1 threshold %d' % thresh1_binary, title_window , 0, thresh1_binary, on_trackbar_binary_1)
    cv2.createTrackbar('thresh2 canny %d' % thresh2_double_binary, title_window, 0, thresh2_double_binary, on_trackbar_canny_2)
    cv2.createTrackbar('thresh2 threshold %d' % thresh2_binary, title_window , 0, thresh2_binary, on_trackbar_binary_2)

    i = wellNum
    on_trackbar_canny_1(thresh1_double_binary)
    on_trackbar_canny_2(thresh2_double_binary)
    on_trackbar_binary_1(thresh1_binary)
    on_trackbar_binary_2(thresh2_binary)

    cv2.waitKey(0)
    exit()