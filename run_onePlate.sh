#!/bin/bash
### Testing for the corect number of input arguments
if [ ! $# -eq 3 ];then
	echo "Usage Error: incorrect number of arguments
sh run.sh [plateID] [output_plate_folder] [plate_temperature 4c/20c]"
	exit 1
fi

### Take command-line arguments and set them to variables
output_dir=$1
temperature=$2
plateID=$3

bash transfer_imgs_1.sh "${plateID} ${output_dir} ${temperature}" 169.230.29.134
python bounding_box_overlay_2.py "${output_dir}"
python pregui_img_analysis_3.py "${output_dir}"


