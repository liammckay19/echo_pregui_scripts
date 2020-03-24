import argparse

from transfer_imgs_1 import run as transfer_imgs
from bounding_box_overlay_2 import run as bounding_box_overlay
from pregui_img_analysis_3 import get_dict_image_to_well, createJson
from organizeImages import organizeImages, rename_overview_images_well_id

def argparse_reader_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ids', '--plateID', nargs='+', type=int,
                        help='RockMaker Plate ID (4 or 5 digit code on barcode or 2nd number on RockMaker screen experiment file',
                        required=True)
    parser.add_argument('-dir', '--output_plate_folder', type=str, help='Output folder for images and json',
                        required=True)
    parser.add_argument('-temp', '--plate_temp', type=int, help='Temperature of plate stored at', required=True)
    return parser


def main():
    args = argparse_reader_main().parse_args()

    plateID_list = args.plateID
    output_dir = args.output_plate_folder
    temperature = args.plate_temp
    rock_drive_ip = "169.230.29.134"

    for plateID in plateID_list:
        transfer_imgs(plateID, output_dir, rock_drive_ip)
        organizeImages(output_dir)
        rename_overview_images_well_id(output_dir)
        bounding_box_overlay(output_dir)
        img_well_dict = get_dict_image_to_well(output_dir)
        createJson(plate_dir=output_dir, plate_id=plateID, plate_temperature=temperature,
                   dict_image_path_subwells=img_well_dict)


if __name__ == '__main__':
    main()
