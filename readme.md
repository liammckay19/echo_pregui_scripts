# Echo_pregui_script 
`usage: echo_pregui_run.py [-h] -ids PLATEID [PLATEID ...] -dir
                          OUTPUT_PLATE_FOLDER -temp PLATE_TEMP`
                          
                          
## Help:
  - `-h, --help`            
  
            show this help message and exit
  - `-ids PLATEID [PLATEID ...], --plateID PLATEID [PLATEID ...]`
            
            RockMaker Plate ID (4 or 5 digit code on barcode or
            2nd number on RockMaker screen experiment file
  - `-dir OUTPUT_PLATE_FOLDER, --output_plate_folder OUTPUT_PLATE_FOLDER`
                        
            Output folder for images and json
  - `-temp PLATE_TEMP, --plate_temp PLATE_TEMP`
                       
            Temperature of plate where it is stored at

#### SAMPLE OUTPUT

```▶ bash run.sh output_test 20 10962

xray@169.230.29.115's password: 

selected IDs: 108228 105546

After entering password, should only take ~30 sec to download all images on UCSFwpa

xray@169.230.29.115's password: 



organizing images.

('output_test/organizedWells', 'already exists. continuing')

overlaying images.

()

53%|█████▎    | 51/96 [00:09<00:08,  5.46it/s]^C




