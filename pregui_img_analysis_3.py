#python3.7.0
#Project/dropk#/well/profileID/name_ef.jpg
#run this command above the echo project directory

import glob
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

#Function Definitions
#Params
    #r1 lower radius bound
    #r2 upper radius bound
    #search step size between radii
    #edge: a binary image which is the output of canny edge detection
    #peak_num the # of circles searched for in hough space

#Output:
    #accums 
    #cx : circle center x
    #cy : circle center y
    #radii circle radii
def circular_hough_transform(r1,r2,step,edge, peak_num):  # do we need to do this? difference in a radium in circle matters to the conversion rate
    # be able to distinguish up to one pixel of the circle 

    hough_radii = np.arange(r1,r2,step)
    hough_res = hough_circle(edge,hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res,hough_radii,total_num_peaks=peak_num)
    return accums, cx, cy, radii

def single_radii_circular_hough_transform(r1,edge):
    hough_res = hough_circle(edge,r1)
    accums, cx, cy, radii = hough_circle_peaks(hough_res,r1,total_num_peaks=1)
    return accums, cx, cy, radii

#Params
    #This functions draws a circle of radius r centered on x,y on an image. It draws the center of the circle
    #image: input greyscale numpy 2darray
    #cx: int center x
    #cy: int center y
    #color: single 8-bit channel int. i.e 0-255
def draw_circles_on_image(image,cx,cy,radii,colour,dotsize):
    for center_y, center_x, radius in zip(cy,cx,radii):
        circy, circx = circle_perimeter(center_y,center_x,radius)
        image[circy,circx] = colour
        image[cy[0]-dotsize:cy[0]+dotsize,cx[0]-dotsize:cx[0]+dotsize] = colour

def draw_circles_on_image_center(image,cx,cy,radii,colour,dotsize):
    for center_y, center_x, radius in zip(cy,cx,radii):
        circy, circx = circle_perimeter(center_y,center_x,radius)
        image[circy,circx] = colour
        image[cy[0]-dotsize:cy[0]+dotsize,cx[0]-dotsize:cx[0]+dotsize] = colour

def save_canny_save_fit(path,sig,low,high,temp): #sig=3,low = 0, high = 30
    zstack = io.imread(path)  
    image = img_as_ubyte(zstack[:,:,0]) # finds the top x-y pixels in the z-stack
    edges = canny(image,sigma=sig,low_threshold=low,high_threshold=high)   # edge detection
    accum_d, cx_d, cy_d, radii_d = circular_hough_transform(135,145,2,edges,1) #edge detection on drop, params: r1,r2,stepsize,image,peaknum. Key params to change are r1&r2 for start & end radius
    
    if temp == '20C': ### This works well for echo RT plate type for rockmaker
        accum_w, cx_w, cy_w, radii_w = circular_hough_transform(479,495,1,edges,1) #edge detection on well. Units for both are in pixels
    else: ### This works well for echo 4C plate type for rockmaker
        accum_w, cx_w, cy_w, radii_w = circular_hough_transform(459,475,1,edges,1) #edge detection on well. Units for both are in pixels

    return cx_d,cy_d,radii_d, cx_w, cy_w, radii_w

    
def main():

    t0=ti.time()  ### save time to know how long this script takes (this one takes longer than step 2)

    if len(sys.argv) != 2:
        print('Usage: python pregui_analysis.py [plate_dir]')
        print('Aborting script')
        sys.exit()

    current_directory = os.getcwd()
    plate_dir = sys.argv[1]


    image_list=glob.glob(os.path.join(current_directory,plate_dir,"overlayed","*"))
    print(os.path.join(current_directory,plate_dir,"overlayed","*"))
    image_list.sort(key=lambda x: (int(x.split('well_')[1].split('_overlay')[0].split("_subwell")[0])))

    dict_image_path_subwells = {}
    for p in image_list:
        well_subwell=p.split('well_')[1].split('_overlay')[0].replace("subwell","")
        well,subwell = well_subwell.split("_")
        well = "{:02d}".format(int(well))
        well_subwell = well+"_"+subwell
        dict_image_path_subwells[well_subwell] = p

    # Try to find the plateid.txt file
    try:
        with open(os.path.join(current_directory,plate_dir,"plateid.txt"), 'r') as plate_id_file:
            plate_id = int(plate_id_file.read().rstrip())
    except:
        print("File Error: plateid.txt not found. JSON will not have plate_id key")
        plate_id = "UNKNOWN_ID_at_"+plate_dir
    

    try:
        with open(os.path.join(current_directory,plate_dir,"temperature.txt"), 'r') as plate_id_file:
            plate_temperature = plate_id_file.read().rstrip()
    except:
        print("File Error: temperature.txt not found. JSON will not have temperature defined")
        plate_temperature = "UNKNOWN"

    plateKeys = ["date_time","temperature"]
    wellKeys = ["image_path","well_id","well_radius","well_x","well_y","drop_radius","drop_x","drop_y","offset_x","offset_y"]

    ### Create json output dictionary
    a = {}
    a[plate_id] = {}
    a[plate_id] = {key:0 for key in plateKeys}
    a[plate_id]["plate_id"] = plate_id
    a[plate_id]["date_time"] = datetime.now().isoformat(" ")
    a[plate_id]["temperature"] = plate_temperature
    a[plate_id]["subwells"] = {}    
    if plate_temperature == "UNKNOWN":
        print("File Error: Since the plate temperature could not be found, circles will be fit for 20C room temp. continuing...")
    print("Finding pixel location of wells.")
    for im_idx, im_path in tqdm(sorted(dict_image_path_subwells.items())):
        if im_path:
            cx_d,cy_d,radii_d, cx_w, cy_w, radii_w = save_canny_save_fit(im_path,3,0,50,plate_temperature) ### calling this function for 4c or 20c temp
        # cx_d,cy_d,radii_d, cx_w, cy_w, radii_w = [0,0,0,0,0,0] # time saving code (will output zeros)
        ### radii radius of the drop circle 
        ### everything _w is for the well
        ### everything _d is for the drop
        ### plan on keeping the drop information
        offset_x = cx_d - cx_w
        offset_y = cy_w - cy_d
        print(im_idx)
        well,subwell = im_idx.split("_")

        str_well_id = Plate.well_names[int(well)-1]

        letter_number_new_image_path = im_path.split('well')[0]+str_well_id+"_"+subwell+".jpg"
        os.rename(im_path,letter_number_new_image_path)

        # print(cx_w,cy_w,radii_w,cx_d,cy_d,radii_d,cx_w,cy_w,radii_d,name,im_path,0,0,0)

        str_currentWell = "{0}_{1}".format(str_well_id, subwell)
        a[plate_id]["subwells"][str_currentWell] = {key:0 for key in wellKeys}
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

    print("created:", os.path.join(current_directory,plate_dir,plate_dir.replace(os.path.join("a","").replace("a",""),'')) + '.json')
    with open(os.path.join(current_directory,plate_dir,plate_dir.replace(os.path.join("a","").replace("a",""),'')) + '.json', 'w') as fp:
        json.dump(a, fp)
    print('wrote to json')

    print("time to run: %s minutes"%str(int(ti.time()-t0)/60))


if __name__ == "__main__":
    main()
    