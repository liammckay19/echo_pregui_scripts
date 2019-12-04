import glob
import os
import sys
import json

# after JSONs are made individually for each plate, 
# create master large JSON in "original_output_directory/plate_analysis.json"

def main():
    if sys.argv[1]:
        parentDirectory = sys.argv[1]

    jsonFilePaths = glob.glob(parentDirectory+"/*/*.json")
    masterJSON={}
    for fp in jsonFilePaths:
        with open(fp, 'r') as fp:
            masterJSON.update(dict(json.load(fp)))

    with open(parentDirectory+"/plate_analysis.json", 'w') as outfile:
        outfile.write(json.dumps(masterJSON))

if __name__ == "__main__":
    main()