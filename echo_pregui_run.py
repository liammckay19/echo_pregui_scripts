import argparse
import os

from bounding_box_overlay_2 import run as bounding_box_overlay
from organizeImages import organize_images, rename_overview_images_well_id
from pregui_img_analysis_3 import get_dict_image_to_well, create_json
from transfer_imgs_1 import run as transfer_imgs


def argparse_reader_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ids', '--plateID', nargs='+', type=int,
                        help='RockMaker Plate ID (4 or 5 digit code on barcode or 2nd number on RockMaker screen '
                             'experiment file',
                        required=True)
    parser.add_argument('-dir', '--output_plate_folder', type=str, help='Output folder for images and json',
                        required=True)
    parser.add_argument('-temp', '--plate_temp', type=int, help='Temperature of plate stored at', required=True)
    parser.add_argument('-box', '--box_overlay', action='store_true', default=True,
                        help='Fits original drop images to drop location')
    parser.add_argument('-convex', '--convex_overlay', action='store_true', default=False,
                        help='Fits cookie cutter of original drop images to drop location')
    parser.add_argument('-circle', '--circle_overlay', action='store_true', default=False,
                        help='Fits hole-punch circle of original drop images to drop location')
    parser.add_argument('-debug', '--debug', action='store_true', default=False,
                        help='Show images during process')
    return parser

def run(rockimager_id, temperature, box=True, circle=False, convex=False, debug=False):
    plateID_list = rockimager_id
    output_dir = os.path.join(os.path.curdir(), "rockimager_images")
    temperature = temperature
    rock_drive_ip = "169.230.29.134"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for plateID in plateID_list:
        output_dir = os.path.join(output_dir, str(plateID))
        transfer_imgs(plateID, output_dir, rock_drive_ip)

    for plateID in plateID_list:
        output_dir = os.path.join(output_dir, str(plateID))
        organize_images(output_dir)
        rename_overview_images_well_id(output_dir)
        bounding_box_overlay(output_dir, box=box, circle=circle, convex=convex,
                             debug=debug)
        img_well_dict = get_dict_image_to_well(output_dir)
        create_json(plate_dir=output_dir, plate_id=plateID, plate_temperature=temperature,
                    dict_image_path_subwells=img_well_dict)

def main():
    args = argparse_reader_main().parse_args()

    plateID_list = args.plateID
    output_dir = args.output_plate_folder
    temperature = args.plate_temp
    rock_drive_ip = "169.230.29.134"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for plateID in plateID_list:
        output_dir = os.path.join(args.output_plate_folder, str(plateID))
        transfer_imgs(plateID, output_dir, rock_drive_ip)

    for plateID in plateID_list:
        output_dir = os.path.join(args.output_plate_folder, str(plateID))
        organize_images(output_dir)
        rename_overview_images_well_id(output_dir)
        bounding_box_overlay(output_dir, box=args.box_overlay, circle=args.circle_overlay, convex=args.convex_overlay,
                             debug=args.debug)
        img_well_dict = get_dict_image_to_well(output_dir)
        create_json(plate_dir=output_dir, plate_id=plateID, plate_temperature=temperature,
                    dict_image_path_subwells=img_well_dict)


if __name__ == '__main__':
    main()
