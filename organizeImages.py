import os
import glob
import shutil
import subprocess
import sys

from Plate import Plate


def rename_overview_images_well_id(imageDirectory):
    image_path_list = sorted(glob.glob(os.path.join(imageDirectory, "overview", "*", "*")), key=lambda x: x[-18:])

    # if (len(image_path_list)) % 96 == 0:  # if divisible by 96, make a 96 well plate
    # else:
    #     raise NotImplementedError("This script doesn't support other than 96 well plates")

    for path in image_path_list:
        subwell = path[path[-18:].find('d') + 1]
        wellNum = int(path.split("wellNum_")[1].split(os.path.sep)[0])
        well_id, _ = p.get_number_to_well_id(wellNum).split("_")
        subprocess.run(
            ["mv", path,
             os.path.join(imageDirectory, "overview", "wellNum_%d" % wellNum, well_id + "_" + subwell + ".jpg")])


def organizeImages(imageDirectory):
    print("organizing images.")
    try:
        if os.path.exists(os.path.join(".", imageDirectory)):
            newDirectory = os.path.join(imageDirectory, "organizedWells")
            try:
                os.mkdir(newDirectory)
            except:
                print(newDirectory, 'already exists. continuing')

            overview_img_dir = os.path.join(imageDirectory, "overview")
            try:
                os.mkdir(overview_img_dir)
            except:
                print(overview_img_dir, 'already exists. continuing')

            for path in sorted(glob.glob(os.path.join(imageDirectory, "batchID*", "*", "profileID_1", "*.jpg")),
                               key=lambda x: x[-18]):
                a = path.split(os.path.sep)
                well_num = "".join([(a[x] if c == 0 else '') for x, c in
                                    enumerate([s.find('well') for s in a])])  # just gets the wellNum_## folder name
                if not os.path.exists(os.path.join(newDirectory, well_num)):
                    os.mkdir(os.path.join(newDirectory, well_num))
                os.system("cp " + path + " " + os.path.join(newDirectory, well_num, a[-1]))
                #
                # if not os.path.exists(os.path.join(overview_img_dir, well_num)):
                #     os.mkdir(os.path.join(overview_img_dir, well_num))
                # if "ef.jpg" in path:
                #     well_id = p.get_number_to_well_id(int(well_num.split("_")[1])).split("_")[0]
                #     os.system("cp " + path + " " + os.path.join(overview_img_dir, well_num,
                #                                                 well_id + "_" + a[-1][1] + ".jpg"))
                #     # overwrites the *ef.jpg copied, all files saved as well id - last img is saved

        else:
            print("Error: cannot find image directory", imageDirectory)
            exit(1)

    except IndexError:
        print("Usage: python organizeImages.py [parent image directory]")
        exit(1)


p = Plate(r=8, c=12, subwell_num=1)  # don't need to worry about subwell (that's specified in img path)
