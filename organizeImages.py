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
			except:
				print(newDirectory, 'already exists. continuing')
			for path in glob.glob(imageDirectory+"/batchID*/*/profileID_1/*.jpg"):
				a = path.split("\\")
				well_num = "".join([(a[x] if c==0 else '') for x,c in enumerate([s.find('well') for s in a])]) # just gets the wellNum_## folder name
				if not os.path.exists(""+newDirectory+"/"+well_num):
					os.mkdir(""+newDirectory+"/"+well_num)
				os.system("cp " + path + " " + ""+newDirectory+"/"+well_num+"/"+a[-1])
		else:
			print("Error: cannot find image directory", imageDirectory)
			exit(1)
	except IndexError:
		print("Usage: python organizeImages.py [parent image directory]")
		exit(1)

if __name__ == "__main__":
	main()