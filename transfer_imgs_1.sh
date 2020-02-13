### Testing for the corect number of input arguments
### 	expecting 1 arugment for PlateID
if [ ! $# -eq 4 ];then
	echo "Usage Error: incorrect number of arguments
sh transfer_imgs.sh [plateID] [output_plate_folder] [plate_temperature 4c/20c] [rock_drive LAN 1 ip address]"
	exit 1
fi

### Take command-line arguments and set them to variables
plateID=$1
output_dir=$2
temperature=$3
rock_drive_ip=$4

### Make output directory
mkdir -p "${output_dir}/overlay"

### Run rsync to grab all non-thumbnail image paths and store in file
rsync -nmav --rsync-path "/bin/rsync" --include "*/" --exclude "*_th.jpg" --include "*.jpg" xray@${rock_drive_ip}:"/volume1/RockMakerStorage/WellImages/${plateID: -3}/plateID_${plateID}/" ${output_dir}/ > ${output_dir}/log_rsync_init_file_list.txt

### grab first and last batch IDs from rsync path list
batchID_overview=`grep ".jpg" ${output_dir}/log_rsync_init_file_list.txt | awk -F "batchID_|/wellNum" '{print $2}' | sort | uniq | head -n 1`
batchID_drop=`grep ".jpg" ${output_dir}/log_rsync_init_file_list.txt | awk -F "batchID_|/wellNum" '{print $2}' | sort | uniq | tail -n 1`

echo "selected IDs: ${batchID_drop} ${batchID_overview}"

### Create a list of files to transfer in a text file for rsync to transfer using the --files-from option
###   first cat:  add drop images to file list
###   second cat: add overview drop location files to file list
###   third cat:  add overview extended focus images to file list
#cat log_rsync_init_file_list.txt | grep "${batchID_drop}\|${batchID_overview}" | grep ".jpg" > files_to_transfer.txt
cat ${output_dir}/log_rsync_init_file_list.txt | grep "${batchID_drop}" | grep ".jpg" > ${output_dir}/files_to_transfer.txt
cat ${output_dir}/log_rsync_init_file_list.txt | grep "${batchID_overview}" | grep "dl.jpg" >> ${output_dir}/files_to_transfer.txt
cat ${output_dir}/log_rsync_init_file_list.txt | grep "${batchID_overview}" | grep "dl.jpg" | sed 's/dl/ef/' >> ${output_dir}/files_to_transfer.txt
echo ${plateID} > ${output_dir}/plateid.txt
echo ${temperature} > ${output_dir}/temperature.txt

### transfer files using rsync
echo "After entering password, should only take ~30 sec to download all images on UCSFwpa"
rsync --progress -mav --rsync-path "/bin/rsync" --files-from=${output_dir}/files_to_transfer.txt xray@${rock_drive_ip}:"/volume1/RockMakerStorage/WellImages/${plateID: -3}/plateID_${plateID}" ${output_dir}/ > ${output_dir}/log_rsync_transferred.txt
