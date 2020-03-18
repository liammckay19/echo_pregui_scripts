### Testing for the corect number of input arguments
### 	expecting 1 arugment for PlateID

import argparse
import os
from glob import glob
from os.path import join, exists
import subprocess  # runs bash commands in python

from tqdm import tqdm


def argparse_reader():
    parser = argparse.ArgumentParser()
    parser.add_argument('plateID', type=int,
                        help='RockMaker Plate ID (4 or 5 digit code on barcode or 2nd number on RockMaker screen experiment file')
    parser.add_argument('output_plate_folder', type=str, help='Output folder for images and json')
    parser.add_argument('plate_temp', type=int, help='Temperature of plate stored at')
    parser.add_argument('rock_drive_IP_address', type=str, help="IP addess of rock_drive storage (images)")
    return parser


def main():
    args = argparse_reader().parse_args()

    plateID = args.plateID
    output_dir = args.output_plate_folder
    temperature = args.plate_temp
    rock_drive_ip = args.rock_drive_IP_address

    if not exists(join(output_dir)):
        os.mkdir(join(output_dir))

    rsync_log = ["rsync", "-nmav", "--include", "*/", "--exclude", "*_th.jpg", "--include", "*.jpg", "-e", "ssh",
                 "xray@" + rock_drive_ip + ":/volume1/RockMakerStorage/WellImages/" + str(plateID)[
                                                                                      2:] + '/plateID_' + str(
                     plateID) + '/', str(output_dir) + '/']

    print(*rsync_log)
    rsync = subprocess.run(rsync_log, capture_output=True)
    rsync_out = rsync.stdout.decode("utf-8")
    with open(join(output_dir, "log_rsync_init_file_list.txt"), 'w') as log_rsync_file_out:
        log_rsync_file_out.write(rsync.stdout.decode("utf-8"))

    # get all batches
    batches = set()
    with open(join(output_dir, "log_rsync_init_file_list.txt"), 'r', encoding='utf-8') as log_rsync_file:
        for file in sorted(log_rsync_file.readlines()):
            if 'batchID_' in file:
                batches.add(int(
                    file.split('batchID_')[1].split('wellNum')[0].replace("/", '').replace("\\", '').replace('n', '')))
    batches = sorted(list(batches))
    batchID_overview = batches[-1]  # last in list
    batchID_drop = batches[0]  # first in list
    print("batch IDs selected: ", *batches)

    # ### Create a list of files to transfer in a text file for rsync to transfer using the --files-from option

    # get unique image names
    # Tries to grab all the images in the same batch first, because doing two batches is two different batches of images
    """ 
    i think batch id are different times taking the pictures so if you try to match a extended focus to a drop 
    image from two different imaging times(aka batches) it will not match exactly and could be the problem with 
    the well not matching
    """
    image_names = set()
    path_names_only_necessary = list()
    for line in rsync_out.split('\n'):
        if ".jpg" in line:
            image_name = line.split('/')[-1]
            if image_name not in image_names:
                path_names_only_necessary.append(line)
            image_names.add(line.split('/')[-1])
    print()

    drop_images_paths = []
    overview_drop_location_paths = []
    overview_extended_focus_paths = []
    # sort by image name
    i = 1
    for path in list(sorted([line for line in path_names_only_necessary], key=lambda line: line.split('/')[-1])):
        if "ef.jpg" in path and i == 1:
            drop_images_paths.append(path)
            i += 1
        elif "dl.jpg" in path:
            overview_drop_location_paths.append(path)
        elif "ef.jpg" in path and i == 2:
            overview_extended_focus_paths.append(path)
            i = 1

    print("drop image:", len(drop_images_paths), " images \noverview ef:", len(overview_extended_focus_paths),
          " images \noverview drop location:", len(overview_drop_location_paths), "images")
    with open(join(output_dir, "files_to_transfer.txt"), 'w') as files_to_transfer:
        for path in tqdm([*drop_images_paths, *overview_drop_location_paths, *overview_extended_focus_paths]):
            files_to_transfer.write(path + "\n")

    rsync_download = [
        "rsync", "--info=progress2", "--files-from=" + output_dir + "/files_to_transfer.txt", "-e", "ssh",
                                     "xray@" + rock_drive_ip + ":" + join("/volume1",
                                                                          "RockMakerStorage",
                                                                          "WellImages",
                                                                          str(plateID)[2:],
                                                                          'plateID_' + str(plateID)),
        join(str(output_dir), "")
    ]
    print(*rsync_download)
    subprocess.run(rsync_download)

if __name__ == '__main__':
    main()
