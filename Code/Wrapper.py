"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author: 
Aditya Jadhav (amjadhav@umd.edu)
Master's Candidate in Robotics Engineering,
University of Maryland, College Park
"""

import glob
import cv2
import math
import matplotlib.pyplot as plt
import imutils
import sklearn.cluster
import numpy as np
np.set_printoptions(suppress=True,precision=3)


def getImages():
    
    image_list = []
    for i in range(10):
        for file in glob.glob('../BSDS500/Images/' + str(i + 1) + '.jpg'): 
            im = cv2.imread(file)
            if im is not None:
                image_list.append(im)
            else:
                print("Couldn't load the image :( ", file)
    return image_list
    
    
def gaussForDOG(sigma, filter_size):
    
    kernel_gaussian = np.zeros([filter_size,filter_size])
    limit = (filter_size - 1) / 2
    values = np.linspace(- limit, limit, filter_size)
    
    xx, yy = np.meshgrid(values,values)
    
    kernel_gaussian = ((0.5) * np.pi * sigma * sigma) * np.exp( -(np.square(xx) + np.square(yy)) / ( 2 * np.square(sigma)) )
    plt.imshow(kernel_gaussian)
    
    return kernel_gaussian


def DoG(scales, size, orient):
    
    dogFilters = []

    gradient_x = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_y = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    for eachScale in scales:
        gaussian = gaussForDOG(eachScale, size)
        
        # sobel_x = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
        # sobel_y = cv2.Scharr(gaussian, cv2.CV_64F, 0, 1)
        
        gauss_X = cv2.filter2D(gaussian, -1, gradient_x)
        gauss_Y = cv2.filter2D(gaussian, -1, gradient_y)
        
        for eachOrient in range(orient):
            gaussian_current = (gauss_X * np.cos((eachOrient * 2 * np.pi / orient)) + gauss_Y * np.cos((eachOrient * 2 * np.pi / orient)))
            dogFilters.append(gaussian_current)
            
    return dogFilters


def showDoG(dog):
    
    plt.subplots(int(len(dog) / 5), 5, figsize=(20, 20))
    for d in range(len(dog)):
        plt.subplot(len(dog) / 5, 5, d + 1)
        plt.axis('off')
        plt.imshow(dog[d], cmap='gray')
    plt.savefig('../Results/Filters/DoGBank.png')
    plt.close()


def gaussian1d(sigma, mean, x, order):
    
    mean_x = np.array(x) - mean
    var = sigma ** 2

    gaussian = (1 / np.sqrt(2 * np.pi * var)) * (np.exp((- 1 * mean_x * mean_x) / (2 * var)))

    if order == 0:
        g_1d = gaussian
        return g_1d
    elif order == 1:
        g_1d = - gaussian * ((mean_x) / (var))
        return g_1d
    else:
        g_1d = gaussian * (((mean_x * mean_x) - var) / (var ** 2))
        return g_1d


def gaussian2d(size, scales):
    
    var = scales * scales
    shape = (size, size)
    
    n, m = [(i - 1)/2 for i in shape]
    x, y = np.ogrid[- m : m + 1, - n : n + 1]
    g_2d = (1 / np.sqrt(2 * np.pi * var)) * np.exp( -(x * x + y * y) / (2 * var))
    
    return g_2d


def log2d(size, scales):
    
    var = scales ** 2
    shape = (size, size)
    
    n, m = [(i - 1)/2 for i in shape]
    x, y = np.ogrid[-m : m + 1, - n : n + 1]
    g_log = (1 / np.sqrt(2 * np.pi * var)) * np.exp( -(x * x + y * y) / (2 * var))
    h = g_log * ((x * x + y * y) - var) / (var ** 2)
    
    return h


def generateFilter(scale, phasex, phasey, pts, size):

    gauss_x = gaussian1d(3 * scale, 0, pts[0,...], phasex)
    gauss_y = gaussian1d(scale, 0, pts[1,...], phasey)
    image = np.reshape(gauss_x * gauss_y, (size, size))
    
    return image


def LM(size, scalex, orient, rotinv):
    
    nbar  = len(scalex) * orient
    nedge = len(scalex) * orient
    nf    = nbar + nedge + rotinv
    kernel    = np.zeros([size, size, nf])
    index  = (size - 1)/2
    
    [xx,yy] = np.meshgrid(np.arange(- index, index + 1), np.arange(- index, index + 1))
    pts = np.asarray([xx.flatten(), yy.flatten()])

    count = 0
    for scale in range(len(scalex)):
        for orientation in range(orient):
            angle = (np.pi * orientation)/orient
            cosine = np.cos(angle)
            sin = np.sin(angle)
            rotpts = np.asarray([[cosine + 0, - sin + 0], [sin + 0, cosine + 0]])
            rotpts = rotpts@pts
            kernel[:, :, count] = generateFilter(scalex[scale], 0, 1, rotpts, size)
            kernel[:, :, count + nedge] = generateFilter(scalex[scale], 0, 2, rotpts, size)
            count = count + 1

    count = nbar + nedge
    scales = np.asarray([np.sqrt(2), 2 * np.sqrt(2), 3 * np.sqrt(2), 4 * np.sqrt(2)])

    for i in range(len(scales)):
        kernel[:, :, count]   = gaussian2d(size, scales[i])
        count = count + 1

    for i in range(len(scales)):
        kernel[:, :, count] = log2d(size, scales[i])
        count = count + 1

    for i in range(len(scales)):
        kernel[:, :, count] = log2d(size, 3 * scales[i])
        count = count + 1

    return kernel


def showLM(lm):
    
    x, y, r = lm.shape
    plt.subplots(4, 12, figsize=(20, 20))
    for l in range(r):
        plt.subplot(4, 12, l + 1)
        plt.axis('off')
        plt.imshow(lm[:, :, l], cmap='binary')
    plt.savefig('../Results/Filters/LMBank.png')
    plt.close()


def gabor(sigma, theta, Lambda, psi, gamma):
    
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    nstds = 3
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(- 0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    
    return gb


def gaborBank(sigma, theta, _lambda, psi, gamma, kernels):
    
    g_bank = []
    for j in sigma:
        filters = gabor(j, theta, _lambda, psi, gamma)
        angle = np.linspace(0, 360, kernels)
        for i in range(kernels):
            rotated = imutils.rotate(filters, angle[i])
            g_bank.append(rotated)
            
    return g_bank


def showGabor(gabor):
    
    plt.subplots(int(len(gabor) / 5), 5, figsize=(20, 20))
    for g in range(len(gabor)):
        plt.subplot(len(gabor) / 5, 5, g + 1)
        plt.axis('off')
        plt.imshow(gabor[g], cmap='gray')
    plt.savefig('../Results/Filters/GaborBank.png')
    plt.close()


def textonGeneration(image, bank, flag):
    
    image_textron = np.array(image)
    
    if flag == 0 or flag == 2:
         filters = len(bank)
    else:
        x, y, filters = bank.shape
    for eachFilter in range(filters):
        if flag == 0 or flag == 2:
            filtered = cv2.filter2D(image, -1, bank[eachFilter])
        else:
            filtered = cv2.filter2D(image, -1, bank[:, :, eachFilter])
        image_textron = np.dstack((image_textron, filtered))
        
    return image_textron


def textonMap(k, image, dog, lm, gabor):
    
    r, g, b = image.shape
    
    for i in range(3):
        if i == 0:
            # print("DOG")
            dog_tMap = textonGeneration(image, dog, i)
        elif i == 1:
            # print("LM")
            lm_tMap = textonGeneration(image, lm, i)
        else:
            # print("GB")
            gabor_tMap = textonGeneration(image, gabor, i)
    
    final_texton = np.dstack((dog_tMap[:, :, 1:], lm_tMap[:, :, 1:], gabor_tMap[:, :, 1:]))
    x, y, z = final_texton.shape
    final_texton = final_texton.reshape((r * g), z)
    
    kmeans = sklearn.cluster.KMeans(n_clusters = k, random_state = 2)
    kmeans.fit(final_texton)
    labels = kmeans.predict(final_texton)
    labels = labels.reshape([x, y])
    plt.imshow(labels)
    
    return labels
    
 
def brightnessMap(k, image):
    
    r, g, b = image.shape
    
    final_brightness = image.reshape((r * g), b)
    
    kmeans = sklearn.cluster.KMeans(n_clusters = k, random_state = 2)
    kmeans.fit(final_brightness)
    labels = kmeans.predict(final_brightness)
    labels = labels.reshape([r, g])
    plt.imshow(labels)
    
    return labels 
    
    
def colorMap(k, image):
    
    r, g, b = image.shape
    
    final_color = image.reshape((r * g), b)
    
    kmeans = sklearn.cluster.KMeans(n_clusters = k, random_state = 2)
    kmeans.fit(final_color)
    labels = kmeans.predict(final_color)
    labels = labels.reshape([r, g])
    plt.imshow(labels)
    
    return labels


def hdGen(radius, orient):
    
    size = 2 * radius + 1
    hd = np.zeros([size, size])
    for i in range(radius):
        for j in range(size):
            r = np.square(i - radius) + np.square(j - radius)
            if r <= np.square(radius):
                hd[i,j] = 1
    hd = imutils.rotate(hd, orient)
    hd[hd<=0.5] = 0
    hd[hd>0.5] = 1
    
    return hd   


def halfDiskMasks(scales, orient):
    
    disk_filters = []
    for radius in scales:
        disk_filters_pairs = []
        temp = []
        for orientation in range(orient):
            angle = orientation * 360 / orient
            halfDisk = hdGen(radius, angle)
            temp.append(halfDisk)

        for i in range(int(orient / 2)):
            disk_filters_pairs.append(temp[i])
            disk_filters_pairs.append(temp[i + int((orient) / 2)])

        disk_filters = disk_filters + disk_filters_pairs
    
    return disk_filters


def showHD(hd):
    
    plt.subplots(math.ceil(len(hd) / 5), 5, figsize=(20, 20))
    for d in range(len(hd)):
        plt.subplot(math.ceil(len(hd) / 4), 4, d + 1)
        plt.axis('off')
        plt.imshow(hd[d], cmap='gray')
    plt.savefig('../Results/Filters/HalfDiskBank.png')
    plt.close()
    
    
def chi2Gradient(image, chi_bins, hdBank):
    
    copy = image
    g = []
    h = []
    bank_length = len(hdBank) / 2
    for bl in range(int(bank_length)):
        chi_sqr_dist = image * 0
        mask_1 = hdBank[2 * bl]
        mask_2 = hdBank[2 * bl + 1]
        for eachBin in range(chi_bins):
            mask_image = np.ma.MaskedArray(image, image == eachBin)
            mask_image = mask_image.mask.astype(np.int64)
            g = cv2.filter2D(mask_image, -1, mask_1)
            h = cv2.filter2D(mask_image, -1, mask_2)
            chi_sqr_dist = chi_sqr_dist + ((g - h) ** 2 / (g + h + np.exp(-7)))
            
        copy = np.dstack((copy, chi_sqr_dist / 2))
    t_grad = np.mean(copy, axis = 2)
    
    return t_grad
    

# Loading all Images
all_images = getImages()

# all_images = all_images[:2]

# Displaying Images
# for p,image in enumerate(all_images):
#     cv2.imshow('image', all_images[p])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


def main():

    """ 
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    bank_1 = DoG([4, 7, 10], 49, 15)
    # print(len(bank_1))
    print("\nGenerating DOG Bank -->\n")
    showDoG(bank_1)

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    bank_2 = LM(49, np.asarray([np.sqrt(2), 2 * np.sqrt(2), 3 * np.sqrt(2)]), 6, 12)
    # print(len(bank_2))
    print("\nGenerating LM Bank -->\n")
    showLM(bank_2)

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    bank_3 = gaborBank([9, 16, 25], 0.25, 1, 1, 1, 15)
    # print(len(bank_3))
    print("\nGenerating GABOR Bank -->\n")
    showGabor(bank_3)

    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    bank_4 = halfDiskMasks([5, 7, 16], 16)
    # print(len(bank_4))
    print("\nGenerating Half Disk Masks -->\n")
    showHD(bank_4)

    
    for i,image in enumerate(all_images):
        
        """
        Generate Texton Map
        Filter image using oriented gaussian filter bank
        """
        t_1 = textonMap(64, all_images[i], bank_1, bank_2, bank_3)
        plt.imsave('../Results/Maps/texton_map_' + str(i + 1) +'.png', t_1)

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
        T_gradient = chi2Gradient(t_1, 64, bank_4)
        plt.imsave('../Results/Gradients/T_g_' + str(i + 1) +'.png', T_gradient)

        """
        Generate Brightness Map
        Perform brightness binning 
        """
        b_1 = brightnessMap(16, all_images[i])
        plt.imsave('../Results/Maps/brightness_map_' + str(i + 1) +'.png', b_1, cmap='binary')

        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        B_gradient = chi2Gradient(b_1, 16, bank_4)
        plt.imsave('../Results/Gradients/B_g_' + str(i + 1) +'.png', B_gradient, cmap='binary')

        """
        Generate Color Map
        Perform color binning or clustering
        """
        c_1 = colorMap(16, all_images[i])
        plt.imsave('../Results/Maps/color_map_' + str(i + 1) +'.png', c_1)

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        C_gradient = chi2Gradient(c_1, 16, bank_4)
        plt.imsave('../Results/Gradients/C_g_' + str(i + 1) +'.png', C_gradient)

        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        sobelPb = cv2.imread('../BSDS500/SobelBaseline/' + str(i + 1) +'.png', 0)
        
        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        cannyPb = cv2.imread('../BSDS500/CannyBaseline/' + str(i + 1) +'.png', 0)
    
        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        pb = (T_gradient + B_gradient + C_gradient) / 3
        
        pblite_out = np.multiply(pb, (0.9 * sobelPb + 0.1 * cannyPb))
        
        print("\nSaving Pb-Lite Output ...\n")
        plt.imsave('../Results/PB-Lite/outputPB_' + str(i + 1) +'.png', pblite_out)
    
    
if __name__ == '__main__':
    main()
 


