import subprocess

import imutils
from PIL import Image as Im
import numpy as np
import time, sys
import glob
import os

from cv2 import cv2

import organizeImages as oI
from tqdm import tqdm

from Plate import Plate


def save_overview_img(original_fp, imageDirectory):
    plate = Plate(r=8, c=12, subwell_num=1)  # don't need to worry about subwell (that's specified in img path)

    original_fp = os.path.abspath(original_fp)
    fp = original_fp.split(os.path.sep)

    well_num = "".join([(fp[x] if c == 0 else '') for x, c in
                        enumerate([s.find('well') for s in fp])])  # just gets the wellNum_## folder name

    subwell_number = fp[-1][1]

    well_id = plate.get_number_to_well_id(int(well_num.split("_")[1])).split("_")[0]
    new_fp = os.path.join(imageDirectory, "overview", well_id + "_" + subwell_number + ".jpg")

    subprocess.run(["cp", original_fp, new_fp])
    return well_id + "_" + subwell_number


def overlay_images(overview_dl_fh, overview_ef_fh, zoom_fh, output_fh):
    ### This is the main function of the script

    ### Get image with signal from red box in drop location image
    # box_im_open = red_box_subt(overview_dl_fh, 4)
    overview_dl = cv2.imread(overview_dl_fh)
    zoom = cv2.imread(zoom_fh)
    overview_ef = cv2.imread(overview_ef_fh)

    mask = cv2.inRange(overview_dl, np.array([0, 2, 57]), np.array([69, 92, 255]))
    # all pixels in our image that have a R >= 100, B >= 15, and G >= 17 along with R <= 200, B <= 56, and G <= 50 will be considered red.
    # https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

    output = cv2.bitwise_and(overview_dl, overview_dl, mask=mask)
    _, output = cv2.threshold(output, 100, 255, cv2.THRESH_BINARY)

    grey = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 1)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    boundRect = [None] * len(cnts)

    contours_poly = [None] * len(cnts)
    # centers = [None] * len(cnts)
    # radius = [None] * len(cnts)
    biggest_area = 0
    box_with_biggest_area = 0
    for i, c in enumerate(cnts):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        area = boundRect[i][2]*boundRect[i][3]
        if area > biggest_area:
            box_with_biggest_area = i
            biggest_area = area

        # centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    # for cases with blue boxes on overview drop location, this messes up.
    # 1. try to take last contour
    # 2. take largest area of bounding box

    # draw the contours
    # c = c.astype("float")
    # c = c.astype("int")
    # cv2.drawContours(overview_dl, [c], -1, (0, 255, 0), 2)

    # cv2.rectangle(overview_dl, (int(boundRect[i][0]), int(boundRect[i][1])),
    #               (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), (255, 0, 0),
    #               thickness=-1)

    rows, cols, _ = zoom.shape
    max_b = box_with_biggest_area

    b_x, b_y, b_w, b_h = int(boundRect[max_b][0]), int(boundRect[max_b][1]), int(boundRect[max_b][2]), int(boundRect[max_b][3])

    # width of box found = new x width for zoom picture
    xscale = b_w
    # height of box found = new y height for zoom picture
    yscale = b_h

    xscale = max(xscale,yscale)
    yscale = max(xscale,yscale)

    if (xscale, yscale) > (0, 0):
        drop = cv2.resize(zoom, (xscale, yscale), interpolation=cv2.INTER_AREA)
    else:
        drop = cv2.resize(zoom, (cols//2, rows//2), interpolation=cv2.INTER_AREA)

    if (drop.shape[0]+b_y,drop.shape[1]+b_x) <= (overview_ef.shape[0], overview_ef.shape[1]):
        overview_ef[b_y:b_y + drop.shape[0], b_x:b_x + drop.shape[1]] = drop
    cv2.imwrite(output_fh, overview_ef)
    return overview_ef


def red_box_subt(w_box_fh, scaling_factor):
    ### Funciton to read in overview image with red drop location box and convert to grey scale image of red box signal
    ### Open images
    # ef_open = Im.open(wo_box_fh)
    dl_open = Im.open(w_box_fh)

    ### Check and get size
    # assert ef_open.size == dl_open.size
    dl_im_width, dl_im_height = dl_open.size
    search_w = int(dl_im_width / scaling_factor)
    search_h = int(dl_im_height / scaling_factor)
    # print(search_w, search_h)

    ### Resize image for speed
    dl_thumb = dl_open.resize((search_w, search_h), resample=Im.BICUBIC)
    # ef_thumb = ef_open.resize((search_w,search_h),resample=Im.BICUBIC)

    ### Create new image object
    new_dl_box_img = Im.new("L", (search_w, search_h))
    new_pixels = new_dl_box_img.load()

    ### Transform to custom greyscale and subtract
    threshold_val = 50
    # print("time_to_start_loop: %s"%(time.time()-t0))
    for i in range(0, search_w):
        for j in range(0, search_h):
            ### Get pixel are recalculate signal for red box as greyscale
            # pixel_ef = ef_thumb.getpixel((i, j))
            pixel_dl = dl_thumb.getpixel((i, j))

            ### This is an old way of calculating the signal
            # average_ef_bkgd = np.average([pixel_ef[1],pixel_ef[2]])
            # average_dl_bkgd = np.average([pixel_dl[1],pixel_dl[2]])
            # complex_r = np.round(max(pixel_dl[0]-average_dl_bkgd-(pixel_ef[0]-average_ef_bkgd),0))

            complex_r = pixel_dl[0] - (pixel_dl[1] + pixel_dl[2]) / 2

            ### This is an old way of calculating the signal
            # complex_r = max(int(np.round((pixel_dl[0]-pixel_ef[0]+pixel_ef[1]-pixel_dl[1]+pixel_ef[2]-pixel_dl[2])/4.)),0)
            # complex_r = min(255,np.round((pixel_dl[0]/255.)*(abs(pixel_ef[1]-pixel_dl[0])+abs(pixel_dl[1]-pixel_ef[1])+abs(pixel_dl[2]-pixel_ef[2])-50)))

            ### Threshold the new pixel value (complex_r)
            if complex_r < threshold_val:
                complex_r = 0
            ### Set pixel value in new image
            new_pixels[i, j] = (int(complex_r))
    # new_dl_box_img.show()
    dl_open.close()
    ### return new image with calculated pixel values
    return new_dl_box_img


def find_bounding_box(box_open, scaling_factor):
    ### Function finds the oringal size of the red box signal
    x0, y0, x1, y1 = box_open.getbbox()

    return (x0 * scaling_factor, y0 * scaling_factor, scaling_factor * (x1 - x0), scaling_factor * (y1 - y0))


def align_drop_to_box(over_ef, drop_fh, box):
    ### This funciton figures out the correct alignment of the drop to the overview and overlays the images

    ### open the image and compare aspect ratios of the drop location to the drop image
    drop_open = Im.open(drop_fh)
    # print("drop_f_size: {}".format(drop_open.size))
    drop_ratio = drop_open.size[0] / float(drop_open.size[1])
    ##=print("drop_ratio: {}".format(drop_ratio))
    # print("box: {}".format(box))
    box_ratio = box[2] / float(box[3])
    # print("box_ratio: {}".format(box_ratio))

    ### The calcualtion for the alignment of the images is different depending on the ratio of the aspect ratios
    if drop_ratio <= box_ratio:
        ### X-axis based scaling
        ### resize the drop image and calculate the alignemnt
        resize_ratio = box[2] / float(drop_open.size[0])
        new_w = int(np.round(drop_open.size[0] * resize_ratio))
        new_h = int(np.round(drop_open.size[1] * resize_ratio))
        drop_resized = drop_open.resize((new_w, new_h))
        new_x = box[0]
        new_y = int(np.round(((box[3] - new_h) / 2) + box[1]))
    else:
        ### Y-axis based scaling
        ### resize the drop image and calculate the alignemnt
        resize_ratio = box[3] / float(drop_open.size[1])
        new_w = int(np.round(drop_open.size[0] * resize_ratio))
        new_h = int(np.round(drop_open.size[1] * resize_ratio))
        drop_resized = drop_open.resize((new_w, new_h))
        new_x = int(np.round(((box[2] - new_w) / 2) + box[0]))
        new_y = box[1]

    ### open overview image and do the overlay
    overview_open = Im.open(over_ef)
    overview_open.paste(drop_resized, box=(new_x, new_y))
    # overview_open.show()
    return overview_open


def run(imageDirectory):
    if not os.path.exists(imageDirectory):  # case 2: directory doesn't exist
        print("Error: cannot find directory " + imageDirectory)
    else:
        if not os.path.exists(os.path.join(imageDirectory, "overlayed")):
            os.mkdir(os.path.join(imageDirectory, "overlayed"))
        print("overlaying images.\n")
        completedWells = 0

        for i in tqdm(range(1, 97)):
            filepaths = sorted(
                glob.glob(os.path.join(imageDirectory, 'organizedWells', 'wellNum_' + str(i), '*')))  # find all images
            if len(filepaths) % 3 == 0:
                for j in range(0, len(filepaths), 3):
                    zoom_ef_fh = filepaths[0 + j]
                    dl_fh = filepaths[1 + j]
                    ef_fh = filepaths[2 + j]

                    # save overview image (no drop location box) to overview folder
                    well_name = save_overview_img(ef_fh, imageDirectory)

                    output_fp = os.path.join(imageDirectory, "overlayed", well_name + ".jpg")
                    try:
                        overlayed_img = overlay_images(dl_fh, ef_fh, zoom_ef_fh, output_fp)
                        completedWells += 1
                    except TypeError:
                        try:
                            raise RuntimeWarning(
                                "wellNum_%d" % i +
                                'error with overlay: Could not get bounding box from box_open.getbbox(). Image wasn\'t loaded')
                        except RuntimeWarning as e:
                            print(e)
                    except OSError:
                        try:
                            raise RuntimeWarning("wellNum_%d Could not open image file" % i)
                        except RuntimeWarning as e:
                            print(e)

            else:
                try:
                    raise RuntimeWarning("\nwellNum_" + str(
                        i) + " does not have the 3 required images for bounding box overlay. Continuing...")
                except RuntimeWarning as e:
                    print(e)

        ### show how many wells have an overlay
        print("Completed images (should be 96 for 96 well):", completedWells)


def main():
    ### save the time to later see how long script took
    t0 = time.time()
    # save usage to a string to save space
    usage = "Usage: python bounding_box_overlay.py [parent image directory]"
    imageDirectory = ''
    try:  # case 1: catches if there is no argv 1
        # not the greatest, but works
        imageDirectory = sys.argv[1]
    except IndexError:  # if there is, leave the program
        print(usage)
        exit(1)

    run(imageDirectory)
    ### print the time it took to run the script
    print("time to run: %s" % (time.time() - t0))


if __name__ == "__main__":
    main()
