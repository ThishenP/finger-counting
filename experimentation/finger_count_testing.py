import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import glob
from skimage import morphology
from scipy import ndimage
from skimage import measure

plt.rcParams["figure.figsize"] = (15, 10)

def morph_count(f, save_process = False, plot_save_name = None):
    
    binary_hand = morphology.closing(f)
    binary_hand = morphology.opening(binary_hand)
    
    palm_small = morphology.binary_erosion(binary_hand,np.ones([10,10]))
    palm_large = morphology.binary_dilation(palm_small, np.ones([31,31]))
    
    fingers = np.bitwise_and(binary_hand,np.invert(palm_large))
    _,num_fingers = measure.label(fingers,return_num=True)
    
    
    if save_process:
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(binary_hand,cmap='gray')
        ax[0].set_title("Binarised Image", fontsize=25)
        ax[1].imshow(palm_small,cmap='gray')
        ax[1].set_title("Erosion", fontsize=25)
        ax[2].imshow(palm_large,cmap='gray')
        ax[2].set_title("Dilation", fontsize=25)
        ax[3].imshow( fingers,cmap='gray')
        ax[3].set_title("Extracted Fingers", fontsize=25)
        
        fig.savefig(f"{plot_save_name}", dpi=fig.dpi)
    return num_fingers

def hull_count(f, save_process = False,plot_save_name = None):
    contours, hierarchy = cv.findContours(f, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    f =  cv.cvtColor(f, cv.COLOR_GRAY2RGB)
    
    max_idx = 0
    max_area = 0
    for i in range(len(contours)):
        contour_area = cv.contourArea(contours[i])
        if contour_area > max_area:
            max_idx = i
            max_area = contour_area
    max_contour =  contours[max_idx]       
    
    hull = cv.convexHull(max_contour, returnPoints = False)
    hull_points = max_contour[hull][:,0]
    
    defects = cv.convexityDefects(max_contour, hull)
    large_defects = defects[defects[:,0,3]>4500]
    
    
    if save_process:
        fig, ax = plt.subplots(1, 4)
        
        
        ax[0].imshow(f)
        ax[0].set_title("Binarised Image", fontsize=25)
        
        f1 = f.copy()
        cv.drawContours(f1, [max_contour], -1, (255,0,0), 2)
        ax[1].imshow(f1)
        ax[1].set_title("Contours", fontsize=25)
        
        f2 = f.copy()
        cv.drawContours(f2, [hull_points], -1, (0,255,0), 2)
        ax[2].imshow(f2)
        ax[2].set_title("Convex Hull", fontsize=25)
        
        f3 = f.copy()
        for defect in large_defects:
            furthest = max_contour[defect[0][-2]][0]
            cv.circle(f3,furthest,4,[255,0,255],-1)
        
        cv.drawContours(f, [hull_points], -1, (0,255,0), 2)
        ax[3].imshow(f3)
        ax[3].set_title("Defects", fontsize=25)

        
        fig.savefig(f"{plot_save_name}", dpi=fig.dpi)

    return len(large_defects)+1

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

def center_vals(f):
    center = [int(f.shape[0]/2)+20,int(f.shape[1]/2)]
    box = f[center[0]-5:center[0]+5,center[1]-5:center[1]+5]
    flat = box.flatten()
    values, counts = np.unique(flat, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]

def gray_image_thresholding(f):
    _,binary_hand = cv.threshold(f, 90, 255, cv.THRESH_BINARY)
    binary_hand = morphology.closing(binary_hand)
    binary_hand = morphology.opening(binary_hand)
    return binary_hand

def preprocess_image(f):
    cropped = square_crop(f)
    small =  cv.resize(cropped,[128,128], interpolation = cv.INTER_AREA)
    hsv_hand =  cv.cvtColor(small, cv.COLOR_BGR2HSV)
    return hsv_hand

def colour_image_thresholding(f):
    saturation = f[:,:,1]

    blur = cv.GaussianBlur(saturation,(5,5),0)
    _ ,binary_hand = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    if center_vals(binary_hand)==0:
        binary_hand = ~binary_hand
    return binary_hand

def count_fingers(f, count_func, save_process=False, plot_save_name = None):
    hand = preprocess_image(f)
    binary_hand = colour_image_thresholding(hand)
    count = count_func(binary_hand, save_process, plot_save_name)
    return count    
    
gray_hands = []
gray_labels = []
for file in sorted(glob.glob("gray_hands/*.png")):
    hand = cv.imread(file , cv.COLOR_BGR2GRAY)
    binary_hand = gray_image_thresholding(hand)
    gray_hands.append(binary_hand)
    gray_labels.append(int(file[len(file)-6]))
    
colour_hands = []
colour_labels = []
for file in sorted(glob.glob("colour_hands/*.jpg")):
    hand = cv.imread(file)
    colour_hands.append(hand)
    colour_labels.append(int(file[len(file)-5]))

n = len(gray_hands)
right_morph = 0
right_hull = 0
for i in range(n):
    if morph_count(gray_hands[i]) == gray_labels[i]:
        right_morph += 1
    if hull_count(gray_hands[i]) == gray_labels[i]:
        right_hull += 1
        
gray_morph_accuracy = (right_morph/n)*100
gray_hull_accuracy = (right_hull/n)*100

n = len(colour_hands)
right_morph = 0
right_hull = 0
for i in range(n):
    if count_fingers(colour_hands[i], morph_count) == colour_labels[i]:
        right_morph += 1
    if count_fingers(colour_hands[i], hull_count) == colour_labels[i]:
        right_hull += 1
        
colour_morph_accuracy = (right_morph/n)*100
colour_hull_accuracy = (right_hull/n)*100

print('percentage accuracies:')
print('gray:')
print('morph: ', gray_morph_accuracy)
print('hull: ', gray_hull_accuracy)

print('colour:')
print('morph: ', colour_morph_accuracy)
print('hull: ', colour_hull_accuracy)

hsv = preprocess_image(colour_hands[14])
fig, ax = plt.subplots(1, 3)
ax[0].imshow(hsv[:,:,0], cmap = 'hsv')
ax[0].set_title("Hue", fontsize=20)

ax[1].imshow(hsv[:,:,1], cmap='hsv')
ax[1].set_title("Saturation", fontsize=20)

ax[2].imshow(hsv[:,:,2], cmap='hsv')
ax[2].set_title("Value", fontsize=20)
fig.savefig("hsv.png", dpi=fig.dpi)


hsv = preprocess_image(colour_hands[13])
seg = colour_image_thresholding(hsv)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(colour_hands[13])
ax[0].set_title("Input", fontsize=20)

ax[1].imshow(seg, cmap='gray')
ax[1].set_title("Segmented and Binarised", fontsize=20)

fig.savefig("segmentation.png", dpi=fig.dpi)

count_fingers(colour_hands[10], hull_count,True, 'convexity_defects_example')

count_fingers(colour_hands[3], morph_count,True, 'morph_example')

count_fingers(colour_hands[13], hull_count,True, 'noise_conv')

count_fingers(colour_hands[12], morph_count, True, "scale_morph_fail") 

count_fingers(colour_hands[13], morph_count, True, "noise_morph_fail") 

fig, ax = plt.subplots(1, 3)
for i in range(3):
    num = count_fingers(colour_hands[8+i], hull_count)
    
    ax[i].imshow(colour_hands[8+i],cmap='hsv')
    ax[i].set_title(f"num fingers (output): {num}", fontsize=20)
        
fig.savefig("io.png", dpi=fig.dpi)