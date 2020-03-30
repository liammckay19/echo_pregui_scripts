import subprocess

import imutils
import numpy as np
import time, sys
import glob
import os

from cv2 import cv2
from tqdm import tqdm
from Plate import Plate

plate = Plate(r=8, c=12, subwell_num=1)  # don't need to worry about subwell (that's specified in img path)
white = (255, 255, 255)  # white - color


def save_overview_img(original_fp, imageDirectory):
    original_fp = os.path.abspath(original_fp)
    fp = original_fp.split(os.path.sep)

    well_num = "".join([(fp[x] if c == 0 else '') for x, c in
                        enumerate([s.find('well') for s in fp])])  # just gets the wellNum_## folder name

    subwell_number = fp[-1][1]

    well_id = plate.get_number_to_well_id(int(well_num.split("_")[1])).split("_")[0]
    new_fp = os.path.join(imageDirectory, "overview", well_id + "_" + subwell_number + ".jpg")

    subprocess.run(["cp", original_fp, new_fp])
    return well_id + "_" + subwell_number


def align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, black_white_mask_2=None):
    if black_white_mask_2 is None:
        black_white_mask_2 = cv2.bitwise_not(
            np.zeros((zoom.shape[0], zoom.shape[1]), np.uint8))  # make white box zoom size
    box = [b_x, b_y, b_w, b_h]

    drop_ratio = zoom.shape[0] / float(zoom.shape[1])
    box_ratio = b_w / float(b_h)

    ### The calcualtion for the alignment of the images is different depending on the ratio of the aspect ratios
    if drop_ratio <= box_ratio:
        ### X-axis based scaling
        ### resize the drop image and calculate the alignemnt
        resize_ratio = box[2] / float(zoom.shape[1])
        new_w = int(np.round(zoom.shape[1] * resize_ratio))
        new_h = int(np.round(zoom.shape[0] * resize_ratio))
        # drop_resized = drop_open.resize((new_w, new_h))
        new_x = box[0]
        new_y = int(np.round(((box[3] - new_h) / 2) + box[1]))
    else:
        ### Y-axis based scaling
        ### resize the drop image and calculate the alignemnt
        resize_ratio = box[3] / float(zoom.shape[0])
        new_w = int(np.round(zoom.shape[1] * resize_ratio))
        new_h = int(np.round(zoom.shape[0] * resize_ratio))
        # drop_resized = drop_open.resize((new_w, new_h))
        new_x = int(np.round(((box[2] - new_w) / 2) + box[0]))
        new_y = box[1]

    rows, cols, _ = zoom.shape

    xscale = new_w
    yscale = new_h

    # resize drop image and its mask (mask- for convex or circle overlay)
    if (xscale, yscale) > (0, 0):
        drop = cv2.resize(zoom, (xscale, yscale), interpolation=cv2.INTER_AREA)
        drop_mask = cv2.resize(black_white_mask_2, (xscale, yscale), interpolation=cv2.INTER_AREA)
    else:
        drop = cv2.resize(zoom, (cols // 2, rows // 2), interpolation=cv2.INTER_AREA)
        drop_mask = cv2.resize(black_white_mask_2, (cols // 2, rows // 2), interpolation=cv2.INTER_AREA)

    # if drop is a reasonable size
    if (drop.shape[0] + new_y, drop.shape[1] + new_x) <= (overview_ef.shape[0], overview_ef.shape[1]):
        drop_grey = cv2.cvtColor(drop, cv2.COLOR_BGR2GRAY)
        if len(overview_ef.shape) == 3:
            overview_ef_grey = cv2.cvtColor(overview_ef, cv2.COLOR_BGR2GRAY)
        else:
            overview_ef_grey = overview_ef

        overview_mask = np.zeros((overview_ef_grey.shape[0], overview_ef.shape[1]), np.uint8)
        zoom_overview_size = np.zeros((overview_ef_grey.shape[0], overview_ef.shape[1]), np.uint8)

        # overview_mask = large mask
        overview_mask[new_y:new_y + drop.shape[0], new_x:new_x + drop.shape[1]] = drop_mask
        mask_inv = cv2.bitwise_not(overview_mask)

        # zoom picture on black bg
        zoom_overview_size[new_y:new_y + drop.shape[0], new_x:new_x + drop.shape[1]] = drop_grey

        # Now black-out the area of drop in overview_img
        img1_bg = cv2.bitwise_and(overview_ef_grey, overview_ef_grey, mask=mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(zoom_overview_size, zoom_overview_size, mask=overview_mask)

        # Put logo in ROI and modify the main image
        overlay = cv2.add(img1_bg, img2_fg)

        return overlay
    else:
        print("not overlaying an image, drop location is the entire well (not accurate)")
        return overview_ef


def find_image_features(image, mask_color=True, mask_color_min=None, mask_color_max=None, percent_arc_length=0.1,
                        bilateral=False, CONTOUR_METHOD=cv2.CHAIN_APPROX_SIMPLE, RETREIVAL_METHOD=cv2.RETR_EXTERNAL,
                        blur_image=True, blur_iterations=1, box=False):
    if mask_color_max is None:
        mask_color_max = np.array([255, 255, 255])
    if mask_color_min is None:
        mask_color_min = np.array([0, 0, 0])
    if mask_color:
        # mask = low red [0, 2, 57], light red [69, 92, 255]
        mask = cv2.inRange(image, mask_color_min, mask_color_max)
        output = cv2.bitwise_and(image, image, mask=mask)
    else:
        output = image

    _, thresh = cv2.threshold(output, 100, 255, cv2.THRESH_BINARY)

    if len(image.shape) == 3:
        grey_thresh = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
    else:
        grey_thresh = thresh

    if blur_image:
        if bilateral:
            blurred = cv2.bilateralFilter(grey_thresh, 75, 1, 1)
        else:
            blurred = cv2.GaussianBlur(grey_thresh, (5, 5), 1)
    else:
        blurred = grey_thresh

    if blur_iterations > 1:
        for i in range(blur_iterations):
            blurred = cv2.GaussianBlur(blurred, (5, 5), 1)

    _, thresh_blur_grey_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    if RETREIVAL_METHOD == cv2.RETR_EXTERNAL:
        cnts = cv2.findContours(thresh_blur_grey_thresh, cv2.RETR_EXTERNAL,
                                CONTOUR_METHOD)
        cnts = imutils.grab_contours(cnts)
        hierarchy = []
    else:
        cnts, hierarchy = cv2.findContours(thresh_blur_grey_thresh, RETREIVAL_METHOD, CONTOUR_METHOD)

    boundRect = [None] * len(cnts)
    contours_poly = [None] * len(cnts)
    biggest_area = 0
    box_with_biggest_area = 0
    for i, c in enumerate(cnts):
        if box:
            epsilon = 3
        else:
            epsilon = percent_arc_length * cv2.arcLength(c, True)

        contours_poly[i] = cv2.approxPolyDP(c, epsilon, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        area = boundRect[i][2] * boundRect[i][3]
        if area > biggest_area:
            box_with_biggest_area = i
            biggest_area = area
    if RETREIVAL_METHOD == cv2.RETR_EXTERNAL:
        return cnts, boundRect, contours_poly, box_with_biggest_area
    elif len(hierarchy) is not 0:
        return cnts, hierarchy, boundRect, contours_poly, box_with_biggest_area
    else:
        return cnts, boundRect, contours_poly, box_with_biggest_area


def get_drop_location_box(overview_dl, mask_color_min, mask_color_max, debug=False):
    _, boundRect, _, box_with_biggest_area = find_image_features(overview_dl,
                                                                 mask_color_min=mask_color_min,
                                                                 mask_color_max=mask_color_max, blur_iterations=0,
                                                                 box=True,
                                                                 blur_image=True)
    max_b = box_with_biggest_area

    b_x, b_y, b_w, b_h = int(boundRect[max_b][0]), int(boundRect[max_b][1]), int(boundRect[max_b][2]), int(
        boundRect[max_b][3])

    xoffset = 0
    yoffset = 0
    woffset = 6
    hoffset = 6
    b_x, b_y, b_w, b_h = b_x + xoffset, b_y + yoffset, b_w - xoffset + woffset, b_h - yoffset + hoffset
    if b_w <= 500 and b_h <= 500:
        if debug:
            cv2.rectangle(overview_dl, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0), 3)
            cv2.imshow('dl', overview_dl)
            cv2.imshow('dl', overview_dl)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return b_x, b_y, b_w, b_h, True
    else:
        # make these half x or half y whichever is bigger (above 500)
        return b_x, b_y, b_w, b_h, False


def find_biggest_contour(image, contours, max_area=None, min_area=100 ** 2, max_border=(100, 100, 100, 100)):
    if max_area is None:
        max_area = max(image.shape) ** 2

    ''' Contour searching algorithm '''
    # find initial contour data
    M = cv2.moments(contours[-1])
    area = cv2.contourArea(contours[-1])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    image_x, image_y, _ = image.shape
    center_image_x, center_image_y = (image_x // 2, image_y // 2)
    best_area = area
    best_center = (cx, cy)
    best_contour = contours[-1]
    border_x_min = max_border[0]
    border_x_max = image_x + max_border[2]
    border_y_min = max_border[1]
    border_y_max = image_y + max_border[3]
    for i in range(len(contours)):
        # find area of contour
        area = cv2.contourArea(contours[i])

        # find moments of contour
        M = cv2.moments(contours[i])

        # find center of mass of contour
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy)

        if min_area < area < max_area:
            if border_x_min < cx < border_x_max and border_y_min < cy < border_y_max:
                best_area = area
                best_center = center
                best_contour = contours[i]

    return best_contour


def overlay_images(overview_dl_fh, overview_ef_fh, zoom_fh, output_fh, circle=False, box=True, convex=False,
                   debug=False):
    ### This is the main function of the script

    overview_dl = cv2.imread(overview_dl_fh)
    zoom = cv2.imread(zoom_fh)
    overview_ef = cv2.imread(overview_ef_fh)

    if debug:
        cv2.imshow("dl, ef, zoom Press Any Key to Close", np.concatenate([overview_dl, overview_ef, zoom]))
        cv2.imshow("dl, ef, zoom Press Any Key to Close", np.concatenate([overview_dl, overview_ef, zoom]))
        cv2.waitKey(0)

    dark_red = np.array([0, 2, 57])
    light_red = np.array([69, 92, 255])
    b_x, b_y, b_w, b_h, img_is_normal_sized = get_drop_location_box(overview_dl, dark_red, light_red, debug=debug)

    if img_is_normal_sized:
        if circle or convex:
            # convert drop image to grey image
            zoom_grey = cv2.cvtColor(zoom, cv2.COLOR_RGB2GRAY)

            # invert image make dark pixels brighter (edge of drop)
            _, zoom_grey = cv2.threshold(zoom_grey, 175, 255, cv2.THRESH_BINARY_INV)

            # blur the image (fill gaps, reduce noise)
            zoom_blur_grey = cv2.GaussianBlur(zoom_grey, (5, 5), 0)

            # (make mask) high contrast dark black and bright white color
            _, zoom_sharp_grey = cv2.threshold(zoom_blur_grey, 0, 255, cv2.THRESH_BINARY_INV)

            # find edges of mask
            edges = cv2.Canny(zoom_sharp_grey, 0, 255)

            # find contour points of mask
            cnts, hierarchy, _, _, _ = find_image_features(edges, mask_color=False, percent_arc_length=0.01,
                                                           bilateral=False,
                                                           CONTOUR_METHOD=cv2.CHAIN_APPROX_NONE,
                                                           RETREIVAL_METHOD=cv2.RETR_TREE)

            # create blank image for masking drop image
            black_white_mask = np.zeros((zoom_grey.shape[0], zoom_grey.shape[1]), np.uint8)

            if circle:
                cnt = find_biggest_contour(image=zoom, contours=cnts, max_area=zoom.shape[0] * zoom.shape[1])
                (circle_x, circle_y), radius = cv2.minEnclosingCircle(cnt)
                (circle_x, circle_y, radius) = (int(circle_x), int(circle_y), int(radius))
                circle_mask = np.zeros((zoom_grey.shape[0], zoom_grey.shape[1]), np.uint8)
                cv2.circle(circle_mask, (circle_x, circle_y), radius, white, -1)
                overview_ef = align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, circle_mask)

            elif convex:
                # make convex shapes that fit biggest contour point set
                hull = []
                for i in range(len(cnts)):
                    hull.append(cv2.convexHull(cnts[i], False))

                # draw them on a mask
                for i in range(len(hull)):
                    cv2.drawContours(black_white_mask, hull, i, white, -1, 8)

                # find contours of that mask (adds some smoothing)
                cnts_mask, hierarchy_mask, _, _, _ = find_image_features(black_white_mask, mask_color=False,
                                                                         percent_arc_length=3,
                                                                         RETREIVAL_METHOD=cv2.RETR_TREE,
                                                                         bilateral=False, blur_image=True,
                                                                         blur_iterations=30)

                # create final convex shape mask
                black_white_mask_2 = np.zeros((zoom_grey.shape[0], zoom_grey.shape[1]), np.uint8)

                # make convex shapes that fit biggest contour point set
                hull = []
                for i in range(len(cnts_mask)):
                    hull.append(cv2.convexHull(cnts_mask[i], False))

                # draw them on a mask
                for i in range(len(hull)):
                    cv2.drawContours(black_white_mask_2, hull, i, white, -1, 8)

                overview_ef = align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, black_white_mask_2)
    elif box:
        overview_ef = align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef)
    else:
        overview_ef = overview_ef

    cv2.imwrite(output_fh, overview_ef)
    return overview_ef


def run(imageDirectory, circle=False, box=True, convex=False, debug=False):
    if not os.path.exists(imageDirectory):  # case 2: directory doesn't exist
        print("Error: cannot find directory " + imageDirectory)
    else:
        if not os.path.exists(os.path.join(imageDirectory, "overlayed")):
            os.mkdir(os.path.join(imageDirectory, "overlayed"))
        print("overlaying images.\n")
        completedWells = 0
        well_folders = glob.glob(os.path.join(imageDirectory, 'organizedWells', 'wellNum_*'))
        for i in tqdm(range(1, len(well_folders) + 1)):
            filepaths = sorted(
                glob.glob(os.path.join(imageDirectory, 'organizedWells', 'wellNum_' + str(i),
                                       '*')))  # find all images in this well
            if len(filepaths) % 3 == 0:
                for j in range(0, len(filepaths), 3):
                    zoom_ef_fh = filepaths[0 + j]
                    dl_fh = filepaths[1 + j]
                    ef_fh = filepaths[2 + j]

                    # save overview image (no drop location box) to overview folder
                    well_name = save_overview_img(ef_fh, imageDirectory)

                    output_fp = os.path.join(imageDirectory, "overlayed", well_name + ".jpg")

                    try:
                        overlayed_img = overlay_images(dl_fh, ef_fh, zoom_ef_fh, output_fp, circle=circle, box=box,
                                                       convex=convex, debug=debug)
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
