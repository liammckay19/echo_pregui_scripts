
# Echo Pregui Script
For local use with hitsDB web application. 
Author - Liam McKay (adapted from scripts from Justin Biel) 

Stand-alone drop-image overlay and analysis tool for Formulatrix RockImager 1000 machine plate well images.

Uses OpenCV, Numpy, Pandas  

## Installation
1. Make a virtual environment 
2. `pip3 install -r requirements.txt`

## Note
When using `-convex` or `-circle` do not expect every image to be overlayed with a convex or circle drop image. 
If it is not possible/reasonable to overlay a circle or convex shape, it will overlay a rectangular image.


 
 ## Example
Example command: `python3 echo_pregui_run.py -ids 10818 -temp 20 -convex -dir echo_pregui_example`
 
 ## Usage
 Use -h flag for help on command-line arguments 

`[-h] ` help

`-ids PLATEID [PLATEID ...] ` Rockimager plate ID (looks like 10930 or 9580)

`-dir OUTPUT_PLATE_FOLDER` Output location of analysis and images from this script 

`-temp PLATE_TEMP 
` Temperature plate is stored at (must be '4' or "20" - not 4c or 20c)

`[-box] 
` (default) Overlay full rectangular drop image onto overview image

`[-convex] 
` Overlay a convex shaped cutout of the drop image onto the overview image

`[-circle] 
` Overlay a circular cutout of the drop image onto the overview image 

`[-debug]` Show images during finding the drop location bounding box 
                          

#### Sample Output

```
▶ python3 echo_pregui_run.py -ids 10818 -temp 20 -convex -dir images/march30_readme_outout

rsync -nmav --include */ --exclude *_th.jpg --include *.jpg -e ssh xray@169.230.29.134:/volume1/RockMakerStorage/WellImages/818/plateID_10818/ images/march30_readme_outout/10818/
xray@169.230.29.134's password: 
batch IDs selected: droplocation, dropimg:  103969 106812
drop image: 96  images 
overview ef: 96  images 
overview drop location: 96 images
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 288/288 [00:00<00:00, 396702.64it/s]

rsync -mav -P --files-from=images/march30_readme_outout/10818/files_to_transfer.txt -e ssh xray@169.230.29.134:/volume1/RockMakerStorage/WellImages/818/plateID_10818 images/march30_readme_outout/10818/
xray@169.230.29.134's password: 
Downloaded Files =  288 (should be 288 = 96*3)
organizing images.
overlaying images.

 75%|███████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 72/96 [00:10<00:03,  6.09it/s]not overlaying an image, drop location is the entire well (not accurate)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:12<00:00,  7.75it/s]
Completed images (should be 96 for 96 well): 96
overviewimgs =  96
Finding pixel location of wells.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:05<00:00, 18.82it/s]
created: /Users/liam_msg/Documents/echo_pregui_scripts/echo_pregui_scripts/images/march30_readme_outout/10818/imagesmarch30_readme_outout10818.json
wrote to json




