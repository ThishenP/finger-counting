import cv2 as cv
import numpy as np
from skimage import morphology
import glob

#convexity defect method for finger counting
def hull_count(f):
    #finding contours
    contours, hierarchy = cv.findContours(f, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    f =  cv.cvtColor(f, cv.COLOR_GRAY2RGB)
    
    #finding the contour taht surrounds the largest area
    max_idx = 0
    max_area = 0
    for i in range(len(contours)):
        contour_area = cv.contourArea(contours[i])
        if contour_area > max_area:
            max_idx = i
            max_area = contour_area
    max_contour =  contours[max_idx]       
    
    #find convex hull
    hull = cv.convexHull(max_contour, returnPoints = False)
    
    defects = cv.convexityDefects(max_contour, hull)

    #find the largest defects corresponding to space between fingers
    large_defects = defects[defects[:,0,3]>4500]
    
    #return num fingers
    return len(large_defects)+1

#crop The image to a square
def square_crop(f):
    #finds longest side and crops that side to the size of shorter side to create square image
    if f.shape[0] > f.shape[1]:
        cut_size = int((f.shape[0] - f.shape[1])/2)
        cropped = f[cut_size : cut_size + f.shape[1]]
    elif f.shape[0] > f.shape[1]:
        cut_size = int((f.shape[1] - f.shape[0])/2)
        cropped = f[:,cut_size : cut_size + f.shape[0]]
    else:
        return f  
    return cropped

def preprocess_image(f):
    #crops image
    cropped = square_crop(f)

    #scales image down to 128x128
    small =  cv.resize(cropped,[128,128], interpolation = cv.INTER_AREA)

    #converts image to hsv
    hsv_hand =  cv.cvtColor(small, cv.COLOR_BGR2HSV)
    return hsv_hand

#returns the most common value in a box at the center of the image
def center_vals(f):
    center = [int(f.shape[0]/2)+20,int(f.shape[1]/2)]
    box = f[center[0]-5:center[0]+5,center[1]-5:center[1]+5]
    flat = box.flatten()
    values, counts = np.unique(flat, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]

#segments and thresholds the hand out of the hand image
def colour_image_thresholding(f):
    saturation = f[:,:,1]

    #blur to reduce noise and benefit thresholding
    blur = cv.GaussianBlur(saturation,(5,5),0)

    #otsu thresholding
    _ ,binary_hand = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    #if inverted
    if center_vals(binary_hand)==0:
        binary_hand = ~binary_hand
    return binary_hand

#find final count 
def count_fingers(f):
    hand = preprocess_image(f)
    binary_hand = colour_image_thresholding(hand)
    count = hull_count(binary_hand)
    return count


input_folder = "input"
#read and display result for all images in input folder
hands =[]
for file in glob.glob(f"{input_folder}/*.jpg"):
    hand = cv.imread(file)
    print(f"image: {file[len(input_folder)+1:]}, num fingers: {count_fingers(hand)}")
