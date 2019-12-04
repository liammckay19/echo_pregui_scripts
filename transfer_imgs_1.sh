# THE ARGUMENT ORDER IS DIFFERENT BETWEEN run.sh and this code

### Testing for the corect number of input arguments
### 	expecting 1 arugment for PlateID
if [ ! $# -eq 3 ];then
	echo "Usage Error: incorrect number of arguments
sh transfer_imgs.sh [plateID] [output_plate_folder] [plate_temperature 4c/20c]"  # THE ARGUMENT ORDER IS DIFFERENT BETWEEN run.sh and this code
	exit 1
fi

### Take command-line arguments and set them to variables
plateID=$1
output_dir=$2
temperature=$3
NAS_IP_ADDRESS="169.230.29.115"  # works as of 12/3/2019 after getting a static IP address.

### Make output directory
mkdir -p "${output_dir}/overlay"

### Run rsync to grab all non-thumbnail image paths and store in file
rsync -e 'ssh -o StrictHostKeyChecking=no' -nmav --rsync-path "/bin/rsync" --include "*/" --exclude "*_th.jpg" --include "*.jpg" xray@${NAS_IP_ADDRESS}:"/volume1/RockMakerStorage/WellImages/${plateID: -3}/plateID_${plateID}/" ${output_dir}/ | tee ${output_dir}/log_rsync_init_file_list.txt

### grab first and last batch IDs from rsync path list
batchID_overview=`grep ".jpg" ${output_dir}/log_rsync_init_file_list.txt | awk -F "batchID_|/wellNum" '{print $2}' | sort | uniq | head -n 1`
batchID_drop=`grep ".jpg" ${output_dir}/log_rsync_init_file_list.txt | awk -F "batchID_|/wellNum" '{print $2}' | sort | uniq | tail -n 1`

echo "associated batchIDs on NAS Server: ${batchID_drop} ${batchID_overview}"

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
echo "Should print files being downloaded after entering password below."
rsync -e 'ssh -o StrictHostKeyChecking=no' -P -mav --rsync-path "/bin/rsync" --files-from=${output_dir}/files_to_transfer.txt xray@${NAS_IP_ADDRESS}:"/volume1/RockMakerStorage/WellImages/${plateID: -3}/plateID_${plateID}" ${output_dir}/ | tee ${output_dir}/log_rsync_transferred.txt
