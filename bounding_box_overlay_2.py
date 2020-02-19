from PIL import Image as Im
import numpy as np
import time, sys
import glob
import os
import organizeImages as oI
from tqdm import tqdm
def overlay_images(overview_dl_fh, overview_ef_fh, zoom_fh,output_fh):
  ### This is the main function of the script

  ### Get image with signal from red box in drop location image
  box_im_open = red_box_subt(overview_dl_fh,4)
  # box_im_open.show()

  ### Get the dimensions of the signal for the red box, by calculating the bounding box 
  box_dims=find_bounding_box(box_im_open,4)

  ### Overlay resized drop image onto the original overview image, using the bounding box dimensions
  overlay_open = align_drop_to_box(overview_ef_fh,zoom_fh,box_dims)

  ### Save the overlayed image
  overlay_open.save(output_fh,format="JPEG")
  #return overlay_open

def red_box_subt(w_box_fh,scaling_factor):
  ### Funciton to read in overview image with red drop location box and convert to grey scale image of red box signal
  ### Open images
  #ef_open = Im.open(wo_box_fh)
  dl_open = Im.open(w_box_fh)

  ### Check and get size
  #assert ef_open.size == dl_open.size
  dl_im_width, dl_im_height = dl_open.size
  search_w = int(dl_im_width/scaling_factor)
  search_h = int(dl_im_height/scaling_factor)
  #print(search_w, search_h)

  ### Resize image for speed
  dl_thumb = dl_open.resize((search_w,search_h),resample=Im.BICUBIC)
  #ef_thumb = ef_open.resize((search_w,search_h),resample=Im.BICUBIC)

  ### Create new image object
  new_dl_box_img = Im.new("L", (search_w,search_h))
  new_pixels = new_dl_box_img.load()

  ### Transform to custom greyscale and subtract
  threshold_val = 50
  #print("time_to_start_loop: %s"%(time.time()-t0))
  for i in range(0, search_w):
    for j in range(0, search_h):
      ### Get pixel are recalculate signal for red box as greyscale
      #pixel_ef = ef_thumb.getpixel((i, j))
      pixel_dl = dl_thumb.getpixel((i, j))

      ### This is an old way of calculating the signal
      #average_ef_bkgd = np.average([pixel_ef[1],pixel_ef[2]])
      #average_dl_bkgd = np.average([pixel_dl[1],pixel_dl[2]])
      #complex_r = np.round(max(pixel_dl[0]-average_dl_bkgd-(pixel_ef[0]-average_ef_bkgd),0))

      complex_r = pixel_dl[0]-(pixel_dl[1]+pixel_dl[2])/2
      
      ### This is an old way of calculating the signal
      #complex_r = max(int(np.round((pixel_dl[0]-pixel_ef[0]+pixel_ef[1]-pixel_dl[1]+pixel_ef[2]-pixel_dl[2])/4.)),0)  
      #complex_r = min(255,np.round((pixel_dl[0]/255.)*(abs(pixel_ef[1]-pixel_dl[0])+abs(pixel_dl[1]-pixel_ef[1])+abs(pixel_dl[2]-pixel_ef[2])-50)))

      ### Threshold the new pixel value (complex_r)
      if complex_r < threshold_val:
        complex_r=0
      ### Set pixel value in new image
      new_pixels[i,j] = (int(complex_r))
  #new_dl_box_img.show()
  dl_open.close()
  ### return new image with calculated pixel values
  return new_dl_box_img


def find_bounding_box(box_open,scaling_factor):
  ### Function finds the oringal size of the red box signal
  x0,y0,x1,y1 = box_open.getbbox()

  return (x0*scaling_factor,y0*scaling_factor,scaling_factor*(x1-x0),scaling_factor*(y1-y0))

def align_drop_to_box(over_ef,drop_fh,box):
  ### This funciton figures out the correct alignment of the drop to the overview and overlays the images

  ### open the image and compare aspect ratios of the drop location to the drop image
  drop_open = Im.open(drop_fh)
  #print("drop_f_size: {}".format(drop_open.size))
  drop_ratio = drop_open.size[0]/float(drop_open.size[1])
  ##=print("drop_ratio: {}".format(drop_ratio))
  #print("box: {}".format(box))
  box_ratio = box[2]/float(box[3])
  #print("box_ratio: {}".format(box_ratio))

  ### The calcualtion for the alignment of the images is different depending on the ratio of the aspect ratios
  if drop_ratio <= box_ratio:
    ### X-axis based scaling
    ### resize the drop image and calculate the alignemnt
    resize_ratio = box[2] / float(drop_open.size[0])
    new_w = int(np.round(drop_open.size[0] * resize_ratio))
    new_h = int(np.round(drop_open.size[1] * resize_ratio))
    drop_resized=drop_open.resize((new_w,new_h))
    new_x = box[0]
    new_y = int(np.round(((box[3] - new_h) / 2) + box[1]))    
  else:
    ### Y-axis based scaling
    ### resize the drop image and calculate the alignemnt
    resize_ratio = box[3] / float(drop_open.size[1])
    new_w = int(np.round(drop_open.size[0] * resize_ratio))
    new_h = int(np.round(drop_open.size[1] * resize_ratio))
    drop_resized=drop_open.resize((new_w,new_h))
    new_x = int(np.round(((box[2] - new_w) / 2) + box[0]))
    new_y = box[1]

  ### open overview image and do the overlay
  overview_open=Im.open(over_ef)
  overview_open.paste(drop_resized,box=(new_x,new_y))
  #overview_open.show()
  return overview_open

def main():
  ### save the time to later see how long script took
  t0=time.time()
  # save usage to a string to save space
  usage = "Usage: python bounding_box_overlay.py [parent image directory]"
  try: # case 1: catches if there is no argv 1
    # not the greatest, but works
    imageDirectory = sys.argv[1]
  except IndexError: # if there is, leave the program
    print(usage)
    exit(1)
  if not os.path.exists(imageDirectory): # case 2: directory doesn't exist
    print("Error: cannot find directory "+imageDirectory)
  else:
    oI.organizeImages(imageDirectory)
    if not os.path.exists(imageDirectory+"/overlayed"):
      os.mkdir(imageDirectory+"/overlayed")
    print("overlaying images.")
    print()
    completedWells = 0

    # generate wells a01-h12
    letters=list('abcdefgh'.upper())
    numbers = ["{:02d}".format(n) for n in range(1,13)]
    wells =[[c+n for n in numbers] for c in letters]
    wellflat=[]
    [[wellflat.append(wells[i][j]) for j in range(len(wells[i]))] for i in range(len(wells))]

    for i in tqdm(range(1,97)):
      filepaths = sorted(glob.glob(imageDirectory+'/organizedWells/wellNum_'+str(i)+'/*')) # find all images 
      subwell_list = [z.split("/d")[1].split("_")[0] for z in filepaths]
      if len(filepaths) % 3 == 0:
        for j in range(0,len(filepaths),3):
          output_fh = imageDirectory+"/overlayed/well_"+str(i)+"_subwell"+subwell_list[0+j]+"_overlay.jpg"
          zoom_ef_fh=filepaths[0+j]
          dl_fh=filepaths[1+j]
          ef_fh=filepaths[2+j]
          try:
            overlayed_img = overlay_images(dl_fh,ef_fh,zoom_ef_fh,output_fh)
            completedWells += 1
          except TypeError:
            print("\nwellNum_"+str(i)+' Overlay Error: Could not get bounding box from box_open.getbbox(). Image wasn\'t loaded')
          except OSError:
            print("\nwellNum_{0} File Error: Could not open image file".format(i))

      else:
        print("\nwellNum_"+str(i)+" does not have the 3 required images for bounding box overlay. Continuing...")

    ### show how many wells have an overlay
    print("Completed images:",completedWells)

    ### print the time it took to run the script
    print("time to run: %s"%(time.time()-t0))
  


if __name__ == "__main__":
  main()
