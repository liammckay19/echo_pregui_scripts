# python3.7.0
# Project/dropk#/well/profileID/name_ef.jpg
# run this command above the echo project directory

import glob
import string
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.util import pad
from skimage import io
import os
from classes_only import Well_well_well as well
from classes_only import Plate

from datetime import datetime, date, time
import time as ti
import json
from tqdm import tqdm
import cv2
import argparse


def argparse_reader():
    parser = argparse.ArgumentParser()
    parser.add_argument('plateID', type=int,
                        help='RockMaker Plate ID (4 or 5 digit code on barcode or 2nd number on RockMaker screen experiment file')
    parser.add_argument('output_plate_folder', type=str, help='Output folder for images and json')
    parser.add_argument('plate_temp', type=int, help='Temperature of plate stored at')
    parser.add_argument('-json', '--generateJson', action='store_true',
                        help="JSON Output re-run well pixel finding algorithm")
    return parser


# Function Definitions
# Params
# r1 lower radius bound
# r2 upper radius bound
# search step size between radii
# edge: a binary image which is the output of canny edge detection
# peak_num the # of circles searched for in hough space

# Output:
# accums
# cx : circle center x
# cy : circle center y
# radii circle radii
def circular_hough_transform(r1, r2, step, edge,
                             peak_num):  # do we need to do this? difference in a radium in circle matters to the conversion rate
    # be able to distinguish up to one pixel of the circle 

    hough_radii = np.arange(r1, r2, step)
    hough_res = hough_circle(edge, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=peak_num)
    return accums, cx, cy, radii


def single_radii_circular_hough_transform(r1, edge):
    hough_res = hough_circle(edge, r1)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, r1, total_num_peaks=1)
    return accums, cx, cy, radii


# Params
# This functions draws a circle of radius r centered on x,y on an image. It draws the center of the circle
# image: input greyscale numpy 2darray
# cx: int center x
# cy: int center y
# color: single 8-bit channel int. i.e 0-255
def draw_circles_on_image(image, cx, cy, radii, colour, dotsize):
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius)
        image[circy, circx] = colour
        image[cy[0] - dotsize:cy[0] + dotsize, cx[0] - dotsize:cx[0] + dotsize] = colour


def draw_circles_on_image_center(image, cx, cy, radii, colour, dotsize):
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius)
        image[circy, circx] = colour
        image[cy[0] - dotsize:cy[0] + dotsize, cx[0] - dotsize:cx[0] + dotsize] = colour


def save_canny_save_fit(path, temp):  # sig=3,low = 0, high = 30
    # circles=[]
    # zstack = io.imread(overlayed_image[i])
    # output = img_as_ubyte(zstack[:, :, 0]).copy()
    # edged = image.copy()
    # # ret, thresh = cv2.threshold(image, thresh1_binary, thresh2_binary, cv2.THRESH_BINARY)
    # print("bw:",thresh1_binary, thresh2_binary)
    # edged = cv2.Canny(thresh, thresh1_canny, thresh2_canny)
    # print("edged:",thresh1_canny, thresh2_canny)
    # circles = cv2.HoughCircles(image=edged, method=cv2.HOUGH_GRADIENT, dp=3, minDist=400, minRadius=475, maxRadius=495)
    # if circles is not None:
    #     found_circles += 1
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(output, (x, y), r, (255, 0, 0), 4)
    #         cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    accum_d, cx_d, cy_d, radii_d, cx_w, cy_w, radii_w = [0, 0, 0, 0, 0, 0, 0]  # initialize variables

    zstack = io.imread(path)
    image = img_as_ubyte(zstack[:, :, 0])  # finds the top x-y pixels in the z-stack
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)  # tested march 17 2020 on plate 10818

    # _, thresh = cv2.threshold(image, 0, 50, cv2.THRESH_BINARY)  # tested march 17 2020 on plate 10818

    edged = cv2.Canny(thresh, 101, 112)
    # drop_circles = cv2.HoughCircles(image=thresh, method=cv2.HOUGH_GRADIENT, dp=3, minDist=100, minRadius=135,
    #                                 maxRadius=145)

    print(temp)

    if int(temp) == 20:  ### This works well for echo RT plate type for rockmaker
        circles = cv2.HoughCircles(image=edged, method=cv2.HOUGH_GRADIENT, dp=1, minDist=0, minRadius=475,
                                   maxRadius=495)
    else:  ### This works well for echo 4C plate type for rockmaker
        circles = cv2.HoughCircles(image=edged, method=cv2.HOUGH_GRADIENT, dp=1, minDist=0, minRadius=459,
                                   maxRadius=475)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cx_w, cy_w, radii_w = circles[0]  # always take first circle found
        print(circles)
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(thresh, (x, y), r, (255, 0, 0), 4)
            cv2.rectangle(thresh, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    cv2.imshow("output", thresh)
    cv2.waitKey(0)

    # # edges = canny(image,sigma=sig,low_threshold=low,high_threshold=high)   # edge detection
    # if temp == '20C': ### This works well for echo RT plate type for rockmaker
    #     accum_w, cx_w, cy_w, radii_w = circular_hough_transform(479,495,1,edges,1) #edge detection on well. Units for both are in pixels
    # else: ### This works well for echo 4C plate type for rockmaker
    #     accum_w, cx_w, cy_w, radii_w = circular_hough_transform(459,475,1,edges,1) #edge detection on well. Units for both are in pixels

    return cx_d, cy_d, radii_d, cx_w, cy_w, radii_w


def get_dict_image_to_well(plate_dir):
    current_directory = os.getcwd()

    image_list = glob.glob(os.path.join(current_directory, plate_dir, "overlayed", "*"))
    overview_list = list(sorted(glob.glob(os.path.join(current_directory,plate_dir, "overview", "*", "*")),key=lambda x: x))
    print(os.path.join(current_directory, plate_dir, "overlayed", "*"))
    well_overlay_subwell_format = True
    try:
        image_list = sorted(image_list,key=lambda x: (int(x.split('well_')[1].split('_overlay')[0].split("_subwell")[0])))
    except IndexError:
        if not image_list:
            well_overlay_subwell_format = False
            image_list = sorted(image_list,key=lambda x: x)

    dict_image_path_subwells = {}
    for p,o in zip(image_list,overview_list):
        if well_overlay_subwell_format:
            well_subwell = p.split('well_')[1].split('_overlay')[0].replace("subwell", "")
        else:
            well_subwell = p.split("overlayed")[1].replace(".jpg", "")
        well, subwell = well_subwell.split("_")

        if well_overlay_subwell_format:
            well = "{:02d}".format(int(well))
        well_subwell = well + "_" + subwell
        dict_image_path_subwells[well_subwell.replace(os.path.sep,"")] = p,o
    return dict_image_path_subwells


def createJson(plate_dir: str, plate_id: int, plate_temperature: int, dict_image_path_subwells: dict) -> None:
    current_directory = os.getcwd()

    plateKeys = ["date_time", "temperature"]
    wellKeys = ["image_path", "well_id", "well_radius", "well_x", "well_y", "drop_radius", "drop_x", "drop_y",
                "offset_x", "offset_y"]

    ### Create json output dictionary
    a = {}
    a[plate_id] = {}
    a[plate_id] = {key: 0 for key in plateKeys}
    a[plate_id]["plate_id"] = plate_id
    a[plate_id]["date_time"] = datetime.now().isoformat(" ")
    a[plate_id]["temperature"] = plate_temperature
    a[plate_id]["subwells"] = {}
    if plate_temperature == "UNKNOWN":
        try:
            raise FileNotFoundError(
                "Since the plate temperature could not be found, circles will be fit for 20C room temp. continuing...")
        except FileNotFoundError:
            pass
    print("Finding pixel location of wells.")
    for im_idx, im_paths in tqdm(sorted(dict_image_path_subwells.items())):
        im_path, im_overview = im_paths
        if im_path:
            cx_d, cy_d, radii_d, cx_w, cy_w, radii_w = save_canny_save_fit(im_overview,
                                                                           plate_temperature)  ### calling this function for 4c or 20c temp
        else:
            try:
                raise FileNotFoundError("Well x,y,r will be zeros "+im_idx)
            except FileNotFoundError:
                cx_d, cy_d, radii_d, cx_w, cy_w, radii_w = [0, 0, 0, 0, 0, 0]  # time saving code (will output zeros)

        # radii radius of the drop circle
        # everything _w is for the well
        # everything _d is for the drop
        # plan on keeping the drop information
        offset_x = cx_d - cx_w
        offset_y = cy_w - cy_d
        well, subwell = im_idx.split("_")

        str_well_id = Plate.well_names[int(well) - 1]

        letter_number_new_image_path = im_path
        if 'well' in im_path:
            letter_number_new_image_path = im_path.split('well')[0] + str_well_id + "_" + subwell + ".jpg"
            os.rename(im_path, letter_number_new_image_path)

        # print(cx_w,cy_w,radii_w,cx_d,cy_d,radii_d,cx_w,cy_w,radii_d,name,im_path,0,0,0)

        str_currentWell = "{0}_{1}".format(str_well_id, subwell)
        a[plate_id]["subwells"][str_currentWell] = {key: 0 for key in wellKeys}
        a[plate_id]["subwells"][str_currentWell]["image_path"] = letter_number_new_image_path
        a[plate_id]["subwells"][str_currentWell]["well_id"] = str_well_id
        a[plate_id]["subwells"][str_currentWell]["well_radius"] = int(radii_w)
        a[plate_id]["subwells"][str_currentWell]["well_x"] = int(cx_w)
        a[plate_id]["subwells"][str_currentWell]["well_y"] = int(cy_w)
        a[plate_id]["subwells"][str_currentWell]["drop_radius"] = int(radii_d)
        a[plate_id]["subwells"][str_currentWell]["drop_x"] = int(cx_d)
        a[plate_id]["subwells"][str_currentWell]["drop_y"] = int(cy_d)
        a[plate_id]["subwells"][str_currentWell]["offset_y"] = int(offset_y)
        a[plate_id]["subwells"][str_currentWell]["offset_x"] = int(offset_x)
        a[plate_id]["subwells"][str_currentWell]["subwell"] = int(subwell)

    print("created:", os.path.join(current_directory, plate_dir,
                                   plate_dir.replace(os.path.join("a", "").replace("a", ""), '')) + '.json')
    with open(os.path.join(current_directory, plate_dir,
                           plate_dir.replace(os.path.join("a", "").replace("a", ""), '')) + '.json', 'w') as fp:
        json.dump(a, fp)
    print('wrote to json')


def main():
    current_directory = os.getcwd()

    args = argparse_reader().parse_args()

    t0 = ti.time()  ### save time to know how long this script takes (this one takes longer than step 2)

    plate_dir = args.output_plate_folder
    plate_id = args.plateID
    plate_temperature = args.plate_temp
    run_only_json = args.generateJson
    if run_only_json:
        try:
            with open(os.path.join(current_directory, plate_dir, "dict_image_path_subwells.json"),
                      'r') as images_to_subwell_json:
                d = dict(json.load(images_to_subwell_json))
            createJson(plate_dir=plate_dir, plate_id=plate_id, plate_temperature=plate_temperature,
                       dict_image_path_subwells=d)
            exit(1)
        except FileNotFoundError as e:
            print(e)
            exit(0)

    dict_image_path_subwells = get_dict_image_to_well(plate_dir=plate_dir)
    with open(os.path.join(current_directory, plate_dir, "dict_image_path_subwells.json"),
              'w') as images_to_subwell_json:
        json.dump(dict_image_path_subwells, images_to_subwell_json)

    createJson(plate_dir=plate_dir, plate_id=plate_id, plate_temperature=plate_temperature,
               dict_image_path_subwells=dict_image_path_subwells)

    print("time to run: %s minutes" % str(int(ti.time() - t0) / 60))


if __name__ == "__main__":
    main()
