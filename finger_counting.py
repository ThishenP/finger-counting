import cv2 as cv
import numpy as np
from skimage import morphology
import glob

def hull_count(f):
    contours, hierarchy = cv.findContours(f, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = max(contours, key=lambda x: cv.contourArea(x))

    hull = cv.convexHull(contours, returnPoints=False)

    defects = cv.convexityDefects(contours, hull)
    defect_lengths = defects[:,0,3]
    
    return len(defect_lengths[defect_lengths>4500])+1

def square_crop(f):
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
    cropped = square_crop(f)
    small =  cv.resize(cropped,[128,128], interpolation = cv.INTER_AREA)
    hsv_hand =  cv.cvtColor(small, cv.COLOR_BGR2HSV)
    return hsv_hand

def center_vals(f):
    center = [int(f.shape[0]/2)+20,int(f.shape[1]/2)]
    box = f[center[0]-5:center[0]+5,center[1]-5:center[1]+5]
    flat = box.flatten()
    values, counts = np.unique(flat, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]

def colour_image_thresholding(f):
    saturation = f[:,:,1]

    blur = cv.GaussianBlur(saturation,(5,5),0)
    _ ,binary_hand = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    if center_vals(binary_hand)==0:
        binary_hand = ~binary_hand
    return binary_hand

def count_fingers(f):
    hand = preprocess_image(f)
    binary_hand = colour_image_thresholding(hand)
    count = hull_count(binary_hand)
    return count


input_folder = "input"

hands =[]
for file in glob.glob(f"{input_folder}/*.jpg"):
    hand = cv.imread(file)
    print(f"image: {file[len(input_folder)+1:]}, num fingers: {count_fingers(hand)}")
