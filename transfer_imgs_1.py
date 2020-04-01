import argparse
import os
from os.path import join, exists
import subprocess  # runs bash commands in python

from tqdm import tqdm


def argparse_reader():
    """
    Parse arguments for the image transfer script
    @return: parse object with arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('plateID', type=int,
                        help='RockMaker Plate ID (4 or 5 digit code on barcode or 2nd number on RockMaker screen experiment file')
    parser.add_argument('output_plate_folder', type=str, help='Output folder for images and json')
    parser.add_argument('rock_drive_IP_address', type=str, help="IP addess of rock_drive storage (images)")
    return parser


def get_path_names_necessary(rsync_out, selected_batches=None):
    """
    Get unique list of path names from rsync file output
    @param rsync_out: rsync files matched output
    @param selected_batches: Batches to use for downloading images (if not specified, find images starting at the last batch)
    @return:
    """
    image_names = set()
    unique_paths = []
    for line in sorted(rsync_out.split('\n'), reverse=True):
        if ".jpg" in line:
            batch = int(
                line.split('batchID_')[1].split('wellNum')[0].replace(os.sep, '').replace("\\", '').replace('n',
                                                                                                            ''))
            jpg = line.split(os.sep)[-1]
            if jpg not in image_names:
                unique_paths.append(line)
                image_names.add(jpg)
            if selected_batches is not None:
                if batch in selected_batches:
                    unique_paths.append(line)
    return unique_paths


def sort_image_path_names(paths):
    """
    Put paths into 3 bins: drop location, zoom, overview
    @param paths:  list of unique image paths from rsync
    @return: zoom, drop location, overview
    """
    drop_images_paths = []
    overview_drop_location_paths = []
    overview_extended_focus_paths = []
    i = 1
    # sort by image name
    for path in list(sorted([line for line in paths], key=lambda line: line.split(os.sep)[-1])):
        if "ef.jpg" in path and i == 1:
            drop_images_paths.append(path)
            i += 1
        elif "dl.jpg" in path:
            overview_drop_location_paths.append(path)
        elif "ef.jpg" in path and i == 2:
            overview_extended_focus_paths.append(path)
            i = 1
    return drop_images_paths, overview_drop_location_paths, overview_extended_focus_paths


def run(plateID, output_dir, rock_drive_ip):
    """
    Transfer Rockimager images from NAS server using rsync
    @param plateID: Rockimager plate ID
    @param output_dir: local output directory
    @param rock_drive_ip: IP address of NAS server
    """
    if not exists(join(output_dir)):
        os.mkdir(join(output_dir))

        rsync_log = ["rsync", "-nmav", "--include", "*/", "--exclude", "*_th.jpg", "--include", "*.jpg", "-e", "ssh",
                     "xray@" + rock_drive_ip + ":/volume1/RockMakerStorage/WellImages/" + str(plateID)[
                                                                                          2:] + '/plateID_' + str(
                         plateID) + '/', str(output_dir) + '/']

        print()
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
                        file.split('batchID_')[1].split('wellNum')[0].replace("/", '').replace("\\", '').replace('n',
                                                                                                                 '')))
        batches = sorted(list(batches))
        # batchID_overview = batches[-1]  # last in list
        # batchID_drop = batches[0]  # first in list
        print("batch IDs selected: droplocation, dropimg: ", batches[0], batches[-1])

        selected_batches = (batches[0], batches[-1])
        # ### Create a list of files to transfer in a text file for rsync to transfer using the --files-from option

        # get unique image names starting from the last image taken. Most recent images will be used.
        path_names_only_necessary = get_path_names_necessary(rsync_out)

        drop_images_paths, overview_drop_location_paths, overview_extended_focus_paths = sort_image_path_names(
            path_names_only_necessary)

        print("drop image:", len(drop_images_paths), " images \noverview ef:", len(overview_extended_focus_paths),
              " images \noverview drop location:", len(overview_drop_location_paths), "images")
        with open(join(output_dir, "files_to_transfer.txt"), 'w') as files_to_transfer:
            for path in tqdm([*drop_images_paths, *overview_drop_location_paths, *overview_extended_focus_paths]):
                files_to_transfer.write(path + "\n")

        rsync_download = [
            "rsync", "-mav", "-P", "--files-from=" + output_dir + "/files_to_transfer.txt", "-e", "ssh",
                                   "xray@" + rock_drive_ip + ":" + join("/volume1",
                                                                        "RockMakerStorage",
                                                                        "WellImages",
                                                                        str(plateID)[2:],
                                                                        'plateID_' + str(plateID)),
            join(str(output_dir), "")
        ]
        print()
        print(*rsync_download)
        rsync_stdout_download = subprocess.run(rsync_download, capture_output=True).stdout.decode("utf-8")
        downloaded_files = 0
        for line in rsync_stdout_download.split("\n"):
            if ".jpg" in line:
                downloaded_files += 1
        print("Downloaded Files = ", downloaded_files, "(should be 288 = 96*3)")
    else:
        try:
            raise RuntimeWarning("Using files from previous download in " + output_dir)
        except RuntimeWarning as e:
            print(e)
            pass


def main():
    args = argparse_reader().parse_args()

    plateID = args.plateID
    output_dir = args.output_plate_folder
    rock_drive_ip = args.rock_drive_IP_address

    run(plateID=plateID, output_dir=output_dir, rock_drive_ip=rock_drive_ip)


if __name__ == '__main__':
    main()
