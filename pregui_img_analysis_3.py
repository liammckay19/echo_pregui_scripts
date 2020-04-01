# python3.7.0
# Project/dropk#/well/profileID/name_ef.jpg
# run this command above the echo project directory

import glob
import numpy as np
import os

from datetime import datetime
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
    parser.add_argument('-debug', '--debug', action='store_true',
                        help='Shows images that well/drop were not found in analysis')
    return parser


# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m=2):
    """
    Trim data in numpy array
    @param data: numpy array
    @param m: max standard deviations
    @return: trimmed data
    """
    return data[abs(data - np.mean(data)) <= m * np.std(data)]


def process_found_circles(circles):
    """
    Average trimmed data to get one circle x,y,radius
    @param circles: numpy array of circles
    @return: x position, y position, radius of circle
    """
    x, y, r = np.average(reject_outliers(circles[:, 0])).astype(int), np.average(reject_outliers(circles[:, 1])).astype(
        int), np.average(reject_outliers(circles[:, 2])).astype(int)
    return x, y, r


def save_canny_save_fit(path, temp, debug=False):
    """
    Find location of well center, drop center on overview image. Averages and trims circles found to be more accurate
    @param path: overview file path
    @param temp: int Tempurature of storage
    @param debug: Show wells and drop found
    @return:
    """
    accum_d, cx_d, cy_d, radii_d, cx_w, cy_w, radii_w = [0, 0, 0, 0, 0, 0, 0]  # initialize variables

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(image, 30, 74, cv2.THRESH_TOZERO_INV)
    ret, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)  # brightens grey to white TO ZERO threshold image
    edged = cv2.Canny(thresh, 101, 112)

    drop_circles = cv2.HoughCircles(image=image, method=cv2.HOUGH_GRADIENT, dp=3, minDist=1, minRadius=135,
                                    maxRadius=145)

    if int(temp) == 20:  ### This works well for echo RT plate type for rockmaker
        circles = cv2.HoughCircles(image=edged, method=cv2.HOUGH_GRADIENT, dp=3, minDist=1, minRadius=475,
                                   maxRadius=495)
    else:  ### This works well for echo 4C plate type for rockmaker
        circles = cv2.HoughCircles(image=edged, method=cv2.HOUGH_GRADIENT, dp=3, minDist=1, minRadius=459,
                                   maxRadius=475)

    image = cv2.UMat(image)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cx_w, cy_w, radii_w = process_found_circles(circles)
        if debug:
            cv2.circle(image, (cx_w, cx_y), radii_w, color=(255,0,0)) # blue
            cv2.imshow("could not find drop, press key to continue", image)
            cv2.waitKey(0)
    else:
        if debug:
            cv2.imshow("could not find well, press key to continue", np.concatenate([image, edged], axis=1))
            cv2.waitKey(0)

    if drop_circles is not None:
        drop_circles = np.round(drop_circles[0, :]).astype("int")
        cx_d, cy_d, radii_d = process_found_circles(drop_circles)
        if debug:
            cv2.circle(image, (cx_d, cx_d), radii_d, color=(0,102,204)) # orange
            cv2.imshow("could not find drop, press key to continue", image)
            cv2.waitKey(0)
    else:
        if debug:
            cv2.imshow("could not find drop, press key to continue", image)
            cv2.waitKey(0)

    return cx_d, cy_d, radii_d, cx_w, cy_w, radii_w


def get_dict_image_to_well(plate_dir):
    """
    Create data file in json format for relating well id to image path
    @param plate_dir: str output directory
    @return:
    """
    current_directory = os.getcwd()

    image_list = glob.glob(os.path.join(current_directory, plate_dir, "overlayed", "*"))
    overview_list = list(
        sorted(glob.glob(os.path.join(current_directory, plate_dir, "overview", "*")), key=lambda x: x))
    print("overviewimgs = ", len(overview_list))
    image_list = sorted(image_list, key=lambda x: x)

    dict_image_path_subwells = {}
    for p, o in zip(image_list, overview_list):
        well_subwell = p.split("overlayed")[1].replace(".jpg", "").replace(os.sep, '')
        well, subwell = well_subwell.split("_")

        well_subwell = well + "_" + subwell
        dict_image_path_subwells[well_subwell.replace(os.path.sep, "")] = p, o
    return dict_image_path_subwells


def create_json(plate_dir: str, plate_id: int, plate_temperature: int, dict_image_path_subwells: dict,
                debug=False) -> None:
    """
    Create pregui script json file
    @param plate_dir: output dir
    @param plate_id: Rockimager plate id
    @param plate_temperature: storage temperatrue
    @param dict_image_path_subwells: well id to image path relationship (get_dict_image_to_well(plate_dir))
    @param debug: Show images that are being processed
    """
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
                                                                           plate_temperature,
                                                                           debug)  ### calling this function for 4c or 20c temp
        else:
            try:
                raise FileNotFoundError("Well x,y,r will be zeros " + im_idx)
            except FileNotFoundError:
                cx_d, cy_d, radii_d, cx_w, cy_w, radii_w = [0, 0, 0, 0, 0, 0]  # time saving code (will output zeros)

        # radii radius of the drop circle
        # everything _w is for the well
        # everything _d is for the drop
        # plan on keeping the drop information
        offset_x = cx_d - cx_w
        offset_y = cy_w - cy_d
        well, subwell = im_idx.split("_")

        letter_number_new_image_path = im_path
        if 'well' in im_path:
            letter_number_new_image_path = im_path.split('well')[0] + well + "_" + subwell + ".jpg"
            os.rename(im_path, letter_number_new_image_path)

        str_currentWell = "{0}_{1}".format(well, subwell)
        a[plate_id]["subwells"][str_currentWell] = {key: 0 for key in wellKeys}
        a[plate_id]["subwells"][str_currentWell]["image_path"] = letter_number_new_image_path
        a[plate_id]["subwells"][str_currentWell]["well_id"] = well
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
    """
    Run image analysis on a directory of images containing overlayed, overview, and batchID_#### folders
    """
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
            create_json(plate_dir=plate_dir, plate_id=plate_id, plate_temperature=plate_temperature,
                        dict_image_path_subwells=d, debug=args.debug)
            exit(1)
        except FileNotFoundError as e:
            print(e)
            exit(0)

    dict_image_path_subwells = get_dict_image_to_well(plate_dir=plate_dir)
    with open(os.path.join(current_directory, plate_dir, "dict_image_path_subwells.json"),
              'w') as images_to_subwell_json:
        json.dump(dict_image_path_subwells, images_to_subwell_json)

    create_json(plate_dir=plate_dir, plate_id=plate_id, plate_temperature=plate_temperature,
                dict_image_path_subwells=dict_image_path_subwells, debug=args.debug)

    print("time to run: %s minutes" % str(int(ti.time() - t0) / 60))


if __name__ == "__main__":
    main()
