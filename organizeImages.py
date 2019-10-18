import os
import glob
import shutil
import sys

def organizeImages(imageDirectory):
	print("organizing images.")
	try:
		if os.path.exists("./"+imageDirectory):
			newDirectory = imageDirectory+"/organizedWells"
			try:
				os.mkdir(newDirectory)
			except FileExistsError:
				print(newDirectory, 'already exists. continuing')
			for path in glob.glob(imageDirectory+"/batchID*/*/profileID_1/*.jpg"):
				parent_directory_paths = path.split("/")
				if not os.path.exists(""+newDirectory+"/"+parent_directory_paths[2]):
					os.mkdir(""+newDirectory+"/"+parent_directory_paths[2])
				os.system("cp " + path + " " + ""+newDirectory+"/"+parent_directory_paths[2]+"/"+parent_directory_paths[-1])
		else:
			print("Error: cannot find image directory", imageDirectory)
			exit(1)
	except IndexError:
		print("Usage: python organizeImages.py [parent image directory]")
		exit(1)

if __name__ == "__main__":
    main()