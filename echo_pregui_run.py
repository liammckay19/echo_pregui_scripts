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
    parser.add_argument('-temp', '--plate_temp', type=int, help='Temperature of plate stored at', required=True)
    parser.add_argument('-box', '--box_overlay', action='store_true', default=True,
                        help='Fits original drop images to drop location')
    parser.add_argument('-convex', '--convex_overlay', action='store_true', default=False,
                        help='Fits cookie cutter of original drop images to drop location')
    parser.add_argument('-circle', '--circle_overlay', action='store_true', default=False,
                        help='Fits hole-punch circle of original drop images to drop location')
    parser.add_argument('-debug', '--debug', action='store_true', default=False,
                        help='Show images during process')
    parser.add_argument('-drop', "--drop_image_number", type=int, help="Specify the batch number for drop image",
                        required=False)
    return parser


def run(plateID_list, temperature, box=True, circle=False, convex=False, debug=False, drop=-1, minRadiusDrop=135,
        maxRadiusDrop=145,
        minRadiusWellRoomTemp=475,
        maxRadiusWellRoomTemp=580,
        minRadiusWellColdTemp=459,
        maxRadiusWellColdTemp=580):
    output_dir = os.path.join(os.path.curdir, "rockimager_images")
    temperature = temperature
    rock_drive_ip = "169.230.29.134"



    for plateID in plateID_list:
        output_dir = os.path.join(output_dir, str(plateID))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        transfer_imgs(plateID, output_dir, rock_drive_ip)

    for plateID in plateID_list:
        organize_images(output_dir)
        rename_overview_images_well_id(output_dir)
        bounding_box_overlay(output_dir, box=box, circle=circle, convex=convex,
                             debug=debug)
        img_well_dict = get_dict_image_to_well(output_dir)
        create_json(plate_dir=output_dir, plate_id=plateID, plate_temperature=temperature,
                                     dict_image_path_subwells=img_well_dict, minRadiusDrop=135, maxRadiusDrop=145,
                                     minRadiusWellRoomTemp=minRadiusWellRoomTemp,
                                     maxRadiusWellRoomTemp=maxRadiusWellRoomTemp,
                                     minRadiusWellColdTemp=minRadiusWellColdTemp,
                                     maxRadiusWellColdTemp=maxRadiusWellColdTemp)

def main():
    args = argparse_reader_main().parse_args()

    plateID_list = args.plateID
    temperature = args.plate_temp
    drop_image_number = args.drop_image_number
    rock_drive_ip = "169.230.29.134"
    convex = args.convex_overlay
    box = args.box_overlay
    circle = args.circle_overlay
    debug = args.debug

    run(plateID_list,temperature,box,circle,convex,debug,drop_image_number,135,145,475,580,459,580)

if __name__ == '__main__':
    main()
