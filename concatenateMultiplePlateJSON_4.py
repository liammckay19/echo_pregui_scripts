import glob
import os
import sys
import json

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