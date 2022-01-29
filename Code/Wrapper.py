"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author: 
Aditya Jadhav (amjadhav@umd.edu)
Master's Candidate in Robotics Engineering,
University of Maryland, College Park
"""

from PIL import Image
import glob
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
np.set_printoptions(suppress=True,precision=3)

def getImages():
    
    image_list = []
    for file in glob.glob(r'D:\UMD\Courses\Sem-3\733\HW0\YourDirectoryID_hw0\YourDirectoryID_hw0\Phase1\BSDS500\Images/*.jpg'): 
        im = cv2.imread(file)
        if im is not None:
            image_list.append(im)
            cv2.imshow("Image", im)
            cv2.waitKey(0)
        else:
            print("Couldn't load the image :( ", file)
    
    print(len(image_list))
    cv2.destroyAllWindows()
    
# getImages()


 







def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""



	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    
if __name__ == '__main__':
    main()
 


