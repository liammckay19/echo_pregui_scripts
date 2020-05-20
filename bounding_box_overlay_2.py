import argparse
import glob
import os
import subprocess
import time

import imutils
import numpy as np
from cv2 import cv2
from tqdm import tqdm

from Plate import Plate

PLATE = Plate(r=8, c=12, subwell_num=1)  # don't need to worry about subwell (that's specified in img path)
COLOR_WHITE = (255, 255, 255)  # white - color


def argparse_reader():
    """
    Parse arguments for the bounding box overlay script
    @return: parse object with arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('output_plate_folder', type=str, help='Parent folder of batchID folders', required=True)
    parser.add_argument('-debug', '--debug', action='store_true',
                        help='Shows images that well/drop were not found in analysis')
    return parser


def save_overview_img(original_fp, output_directory):
    """
    Saves original overview image to its own folder
    @param original_fp: Overview image downloaded location 
    @param output_directory: output directory
    @return: Well_subwell ID (A01_2)
    """
    original_fp = os.path.abspath(original_fp)
    fp = original_fp.split(os.path.sep)

    # noinspection PyTypeChecker
    well_num = "".join([(fp[x] if c == 0 else '') for x, c in
                        enumerate([s.find("well") for s in fp])])  # just gets the wellNum_## folder name

    subwell_number = fp[-1][1]

    well_id = PLATE.get_number_to_well_id(int(well_num.split("_")[1])).split("_")[0]
    # noinspection PyTypeChecker
    new_fp = os.path.join(output_directory, "overview", well_id + "_" + str(subwell_number) + ".jpg")

    subprocess.run(["cp", original_fp, new_fp])
    return well_id + "_" + subwell_number


def align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, black_white_mask_2=None, debug=False):
    """
    Overlay zoom image (with its mask) onto overview image at the location of the red bounding-box drop location
    @param b_x: top left X of box
    @param b_y: top left Y of box
    @param b_w: width of box
    @param b_h: height of box 
    @param zoom: zoom/drop image (in cv2 array format)
    @param overview_ef: overview image (in cv2 array format)
    @param black_white_mask_2: mask for zoom image(in cv2 array format)
    @return: overlayed image (in cv2 array format)
    """
    if black_white_mask_2 is None:
        black_white_mask_2 = cv2.bitwise_not(
            np.zeros((zoom.shape[0], zoom.shape[1]), np.uint8))  # make white box zoom size
    box = [b_x, b_y, b_w, b_h]

    drop_ratio = zoom.shape[0] / float(zoom.shape[1])
    box_ratio = b_w / float(b_h)

    # The calcualtion for the alignment of the images is different depending on the ratio of the aspect ratios
    if drop_ratio <= box_ratio:
        # X-axis based scaling
        # resize the drop image and calculate the alignemnt
        resize_ratio = box[2] / float(zoom.shape[1])
        new_w = int(np.round(zoom.shape[1] * resize_ratio))
        new_h = int(np.round(zoom.shape[0] * resize_ratio))
        # drop_resized = drop_open.resize((new_w, new_h))
        new_x = box[0]
        new_y = int(np.round(((box[3] - new_h) / 2) + box[1]))
    else:
        # Y-axis based scaling
        # resize the drop image and calculate the alignemnt
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
        if debug:
            cv2.imshow('overlay', np.concatenate([overlay, overview_mask, mask_inv]))
            cv2.imshow('overlay', np.concatenate([overlay, overview_mask, mask_inv]))
            cv2.waitKey(0)
        return overlay
    else:
        print("not overlaying an image, drop location is the entire well (not accurate)")
        return overview_ef


def find_image_features(image: np.ndarray, mask_color: bool = True, mask_color_min: np.array = None,
                        mask_color_max: np.array = None,
                        percent_arc_length: float = 0.1,
                        bilateral: bool = False, contour_method: int = cv2.CHAIN_APPROX_SIMPLE,
                        retreival_method: int = cv2.RETR_EXTERNAL,
                        blur_image: bool = True, blur_iterations: int = 1, box: bool = False):
    """
    Analyze image for shapes and colors
    @param image: cv2 numpy array image
    @param mask_color: bool look for a color
    @param mask_color_min: Min BGR values (blue, green, red)
    @param mask_color_max: Max BGR values (blue, green, red)
    @param percent_arc_length: Contour sensitivity to hard turns
    @param bilateral: bool Use bilateral image filtering (very slow)
    @param contour_method: Use simple chain approximation or return all contours found (slower)
    @param retreival_method: Return a hierarchy of contours
    @param blur_image: bool to blur image during contour finding (Gaussian)
    @param blur_iterations: Times to blur the image over on itself (Gaussian)
    @param box: bool return best fit box
    @return: (all arrays of points on image) contours, hierarchy, polygon shape contours, box with biggest area
    """
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

    if retreival_method == cv2.RETR_EXTERNAL:
        cnts = cv2.findContours(thresh_blur_grey_thresh, cv2.RETR_EXTERNAL,
                                contour_method)
        cnts = imutils.grab_contours(cnts)
        hierarchy = []
    else:
        cnts, hierarchy = cv2.findContours(thresh_blur_grey_thresh, retreival_method, contour_method)

    bound_rect = [None] * len(cnts)
    contours_poly = [None] * len(cnts)
    biggest_area = 0
    box_with_biggest_area = 0
    for i, c in enumerate(cnts):
        if box:
            epsilon = 3
        else:
            epsilon = percent_arc_length * cv2.arcLength(c, True)

        contours_poly[i] = cv2.approxPolyDP(c, epsilon, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])
        area = bound_rect[i][2] * bound_rect[i][3]
        if area > biggest_area:
            box_with_biggest_area = i
            biggest_area = area
    if retreival_method == cv2.RETR_EXTERNAL:
        return cnts, bound_rect, contours_poly, box_with_biggest_area
    elif len(hierarchy) is not 0:
        return cnts, hierarchy, bound_rect, contours_poly, box_with_biggest_area
    else:
        return cnts, bound_rect, contours_poly, box_with_biggest_area


def get_drop_location_box(overview_dl, mask_color_min, mask_color_max, debug=False):
    """
    Get bounding box coordinates for drop location
    @param overview_dl: Drop location image
    @param mask_color_min: BGR color minimum
    @param mask_color_max: BGR color maximum
    @param debug: shows box found outlined in blue color
    @return:
    """
    _, bound_rect, _, box_with_biggest_area = find_image_features(overview_dl,
                                                                  mask_color_min=mask_color_min,
                                                                  mask_color_max=mask_color_max, blur_iterations=0,
                                                                  box=True,
                                                                  blur_image=True)
    max_b = box_with_biggest_area

    b_x, b_y, b_w, b_h = int(bound_rect[max_b][0]), int(bound_rect[max_b][1]), int(bound_rect[max_b][2]), int(
        bound_rect[max_b][3])

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
    """
    Get biggest area contour in list of contours
    @param image: image contours processed on
    @param contours: list of contours (from cv2.findContours())
    @param max_area: Upper limit of contour area
    @param min_area: Lower limit of contour area
    @param max_border: Border around image where contour can exist
    @return: contour
    """
    if max_area is None:
        max_area = max(image.shape) ** 2

    ''' Contour searching algorithm '''
    # find initial contour data
    # noinspection PyPep8Naming
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
        # noinspection PyPep8Naming
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


def roverlay_images(overview_dl_fh, overview_ef_fh, zoom_fh, output_fh, circle=False, box=True, convex=False,
                   debug=False):
    """
    Overlay drop image in convex, circle or box shape on overview image from Rockimager
    @param overview_dl_fh: Overview drop location file path
    @param overview_ef_fh: Overview file path
    @param zoom_fh: Drop file path
    @param output_fh: Overlay output file path
    @param circle: bool, Shape circle cut out
    @param box: bool, Shape box overlay
    @param convex: bool, Shape convex cut out
    @param debug: Show images during processing
    @return: overlayed image
    """
    # This is the main function of the script

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

    if debug:
        print("b_x, b_y, b_w, b_h, img_is_normal_sized", b_x, b_y, b_w, b_h, img_is_normal_sized)

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
                                                           contour_method=cv2.CHAIN_APPROX_NONE,
                                                           retreival_method=cv2.RETR_TREE)

            # create blank image for masking drop image
            black_white_mask = np.zeros((zoom_grey.shape[0], zoom_grey.shape[1]), np.uint8)

            if circle:
                cnt = find_biggest_contour(image=zoom, contours=cnts, max_area=zoom.shape[0] * zoom.shape[1])
                (circle_x, circle_y), radius = cv2.minEnclosingCircle(cnt)
                (circle_x, circle_y, radius) = (int(circle_x), int(circle_y), int(radius))
                circle_mask = np.zeros((zoom_grey.shape[0], zoom_grey.shape[1]), np.uint8)
                cv2.circle(circle_mask, (circle_x, circle_y), radius, COLOR_WHITE, -1)
                overview_ef = align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, circle_mask, debug=debug)

            elif convex:
                # make convex shapes that fit biggest contour point set
                hull = []
                for i in range(len(cnts)):
                    hull.append(cv2.convexHull(cnts[i], False))

                # draw them on a mask
                for i in range(len(hull)):
                    cv2.drawContours(black_white_mask, hull, i, COLOR_WHITE, -1, 8)

                # find contours of that mask (adds some smoothing)
                cnts_mask, hierarchy_mask, _, _, _ = find_image_features(black_white_mask, mask_color=False,
                                                                         percent_arc_length=3,
                                                                         retreival_method=cv2.RETR_TREE,
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
                    cv2.drawContours(black_white_mask_2, hull, i, COLOR_WHITE, -1, 8)

                overview_ef = align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, black_white_mask_2,
                                                     debug=debug)
        elif box:
            overview_ef = align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, debug=debug)
    elif box:
        overview_ef = align_drop_to_overview(b_x, b_y, b_w, b_h, zoom, overview_ef, debug=debug)
    else:
        overview_ef = overview_ef

    cv2.imwrite(output_fh, overview_ef)
    return overview_ef


def run(output_directory, circle=False, box=True, convex=False, debug=False):
    """
    Overlay a directory of images
    @param output_directory: output dir
    @param circle: bool Shape circle cut out
    @param box: bool Shape box on zoom image
    @param convex: bool Shape convex cut out
    @param debug: Shows images during processing
    """
    if not os.path.exists(output_directory):  # case 2: directory doesn't exist
        print("Error: cannot find directory " + output_directory)
    else:
        if not os.path.exists(os.path.join(output_directory, "overlayed")):
            os.mkdir(os.path.join(output_directory, "overlayed"))
        print("overlaying images.\n")
        completed_wells = 0
        well_folders = glob.glob(os.path.join(output_directory, 'organizedWells', 'wellNum_*'))
        for i in tqdm(range(1, len(well_folders) + 1)):
            filepaths = sorted(
                glob.glob(os.path.join(output_directory, 'organizedWells', 'wellNum_' + str(i),
                                       '*')))  # find all images in this well
            if len(filepaths) % 3 == 0:
                for j in range(0, len(filepaths), 3):
                    zoom_ef_fh = filepaths[0 + j]
                    dl_fh = filepaths[1 + j]
                    ef_fh = filepaths[2 + j]

                    # save overview image (no drop location box) to overview folder
                    well_name = save_overview_img(ef_fh, output_directory)

                    # noinspection PyTypeChecker
                    output_fp = os.path.join(output_directory, "overlayed", well_name + '.jpg')

                    try:
                        overlayed_img = overlay_images(dl_fh, ef_fh, zoom_ef_fh, output_fp, circle=circle, box=box,
                                                       convex=convex, debug=debug)
                        completed_wells += 1
                    except TypeError:
                        try:
                            raise RuntimeWarning(
                                "wellNum_%d" % i +
                                'error with overlay: Could not get bounding box from box_open.getbbox(). '
                                'Image wasn\'t loaded')
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

        # show how many wells have an overlay
        print("Completed images (should be 96 for 96 well):", completed_wells)


def main():
    """
    Run only overlay step. Overlay images in a directory with timing
    """
    # save the time to later see how long script took
    t0 = time.time()

    args = argparse_reader().parse_args()
    image_directory = args.output_plate_folder
    run(image_directory)
    # print the time it took to run the script
    print("time to run: %s" % (time.time() - t0))


if __name__ == "__main__":
    main()
