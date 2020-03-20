"""
- debugging code for find circles using scikit image
- debugging code for getting the rotation angle
- using subtraction to evaluate the number of rafts that are stepped out
- track circle using Hungarian Algorithm; effused rafts function definition
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as scipy_distance

import os
import glob
import shelve
import progressbar

import scripts.functions_spinning_rafts as fsr

# %% debugging code for find circles using scikit image

# Reading files
tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()
currentFrameNum = 0
currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
# plt.imshow(currentFrameBGR[:,:,::-1])
currentFrameGray = currentFrameBGR[:, :, 1]  # green channel has the highest contrast
# currentFrameGray = cv.imread(tiffFileList[currentFrameNum], 0)
# currentFrameGrayContAdj = cv.equalizeHist(currentFrameGray)
# plt.imshow(currentFrameBGR[:,:,1], 'gray')
plt.imshow(currentFrameGray, 'gray')

# look at the FFT
f = np.fft.fft2(currentFrameGray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))
magnitude_spectrumNormalized = magnitude_spectrum / magnitude_spectrum.max()
magnitude_spectrumEnhanced = np.uint8(magnitude_spectrumNormalized * 255)
# magnitude_spectrumEnhanced =  cv.equalizeHist(magnitude_spectrum)
plt.imshow(magnitude_spectrumEnhanced, 'gray')
outputImageName = tiffFileList[currentFrameNum].partition('.')[0] + '_fft.jpg'
cv.imwrite(outputImageName, magnitude_spectrumEnhanced)

# Reading video frames
videoFileList = glob.glob('*.MOV')
numOfExp = len(videoFileList)
expID = 0
cap = cv.VideoCapture(videoFileList[expID])
numOfFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
retval, currentFrameBGR = cap.read()
currentFrameGray = cv.cvtColor(currentFrameBGR, cv.COLOR_BGR2GRAY)
cap.release()

# below is to expanded version of the function FindCircles

# parameters for various find-circle functions
# frequently-adjusted:
num_of_rafts = 100
radii_Hough = [14, 18]  # [71, 77] for 2.5x [21, 25] for 0.8x, [14, 18] for 0.57x for 5x using coaxial illumination
adaptiveThres_blocksize = 5  # 5, 11, 9, 19
adaptiveThres_const = -13  # -9, -11, -13
raft_center_threshold = 40
min_sep_dist = 40
# cropping
topLeft_x = 0  # 650 #1300
topLeft_y = 0  # 700 #160
width_x = 1250  # 70 #100 # 70
height_y = 1250  # 550 #550
# not used find_circles_adaptive, but in the find_circles_thres and FindAndSortCircles
thres_value = 33
sigma_Canny = 1
low_threshold_canny = 25
high_threshold_canny = 127
# an old parameter that resists all circles within a certain radius, not used anymore
lookup_radius = 880  # unit: pixel
error_message = ' '

# key data set initialization
raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
raft_radii = np.zeros(num_of_rafts, dtype=int)
# raft_centerPixelValue = np.zeros(num_of_rafts)
# raft_InscribedSquareMeanValue = np.zeros(num_of_rafts)
# raft_accumScore = np.zeros(num_of_rafts)
# raft_thresholdValues = np.zeros(num_of_rafts)

# cropping.
image_cropped = currentFrameGray[topLeft_y: topLeft_y + height_y, topLeft_x: topLeft_x + width_x]
plt.imshow(image_cropped, 'gray')

# normal thresholding
# retval, image_thres = cv.threshold(image_cropped, thres_value, 255, 0)
# plt.imshow(image_thres, 'gray')

# adaptive thresholding
image_thres = cv.adaptiveThreshold(image_cropped, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                   adaptiveThres_blocksize, adaptiveThres_const)
plt.imshow(image_thres, 'gray')

# try some morphological computation
# kernel = np.ones((3,3),np.uint8)
# image_close = cv.morphologyEx(image_thres, cv.MORPH_CLOSE, kernel)
# plt.imshow(image_close, 'gray')
# image_dist_transform = cv.distanceTransform(image_thres,cv.DIST_L2,5)
# plt.imshow(image_dist_transform, 'gray')

# canny edge to detect edge
# image_edges = canny(image_thres, sigma=sigma_Canny, low_threshold=low_threshold_canny,
#                     high_threshold=high_threshold_canny)
# plt.imshow(image_edges,'gray')

# hough transform to find circles
hough_results = hough_circle(image_thres, np.arange(*radii_Hough))
accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_Hough))

# assuming that the first raft (highest accumulator score) is a good one
# raft_centers[0,0] = cx[0]
# raft_centers[0,1] = cy[0]
# raft_radii[0] = radii[0]
# raft_centerPixelValue[0] = currentFrameGrayCropped[cy[0], cx[0]]
# raft_InscribedSquareMeanValue[0] = \
#     currentFrameGrayCropped[cy[0]-radii[0]//2:cy[0]+radii[0]//2, cx[0]-radii[0]//2:cx[0]+radii[0]//2].mean()
# raft_accumScore[0] = accums[0]
# raft_thresholdValues[0] = thres_value
raft_count = 0  # starting from 1!

# remove circles that belong to the same raft and circles
# that happened to be in between rafts and rafts outside lookup radius
t1 = time.perf_counter()
for accumScore, detected_cx, detected_cy, detected_radius in zip(accums, cx, cy, radii):
    new_raft = 1
    if image_cropped[detected_cy, detected_cx] < raft_center_threshold:
        new_raft = 0
    elif image_cropped[detected_cy - detected_radius // 2: detected_cy + detected_radius // 2,
                       detected_cx - detected_radius // 2:detected_cx + detected_radius // 2].mean() \
            < raft_center_threshold:
        new_raft = 0
    #    elif  (detected_cx - width_x/2)**2 +  (detected_cy - height_y/2)**2 > lookup_radius**2:
    #        new_raft = 0
    else:
        costMatrix = scipy_distance.cdist(np.array([detected_cx, detected_cy], ndmin=2),
                                          raft_centers[:raft_count, :], 'euclidean')
        if np.any(costMatrix < min_sep_dist):  # raft still exist
            new_raft = 0
    if new_raft == 1:
        raft_centers[raft_count, 0] = detected_cx
        # note that raft_count starts with 1, also note that cx corresonds to columns number
        raft_centers[raft_count, 1] = detected_cy
        # cy is row number
        raft_radii[raft_count] = detected_radius
        # raft_centerPixelValue[raft_count] = currentFrameGrayCropped[detected_cy, detected_cx]
        # raft_InscribedSquareMeanValue[raft_count] = \
        #     currentFrameGrayCropped[detected_cy-detected_radius//2 : detected_cy+detected_radius//2,
        #     detected_cx-detected_radius//2:detected_cx+detected_radius//2].mean()
        # raft_accumScore[raft_count] = accumScore
        # raft_thresholdValues[raft_count] = thres_value
        raft_count = raft_count + 1
    if raft_count == num_of_rafts:
        error_message = 'all rafts found'
        break

# convert the xy coordinates of the cropped image into the coordinates of the original image
raft_centers[:, 0] = raft_centers[:, 0] + topLeft_x
raft_centers[:, 1] = raft_centers[:, 1] + topLeft_y

t2 = time.perf_counter()
timeTotal = t2 - t1  # in seconds
print(timeTotal)
print(error_message)
print(raft_count)

# FindCircles function
raft_centers, raft_radii, raft_count = fsr.find_circles_thres(currentFrameGray, num_of_rafts, radii_hough=radii_Hough,
                                                              thres_value=thres_value, sigma_canny=sigma_Canny,
                                                              low_threshold_canny=low_threshold_canny,
                                                              high_threshold_canny=high_threshold_canny,
                                                              min_sep_dist=min_sep_dist,
                                                              raft_center_threshold=raft_center_threshold,
                                                              top_left_x=topLeft_x, top_left_y=topLeft_y, width_x=width_x,
                                                              height_y=height_y)
print(raft_count)

# FindCircles function with adaptive thresholding
raft_centers, raft_radii, raft_count = fsr.find_circles_adaptive(currentFrameGray, num_of_rafts,
                                                                 radii_hough=radii_Hough,
                                                                 adaptive_thres_blocksize=adaptiveThres_blocksize,
                                                                 adaptive_thres_const=adaptiveThres_const,
                                                                 min_sep_dist=min_sep_dist,
                                                                 raft_center_threshold=raft_center_threshold,
                                                                 top_left_x=topLeft_x, top_left_y=topLeft_y,
                                                                 width_x=width_x, height_y=height_y)
print(raft_count)
# below is for FindAndSortCircles, after using FindCircles to get the first centers
prev_pos = raft_centers
max_displ = 30
currentFrameNum += 1
currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
currentFrameGray = cv.imread(tiffFileList[currentFrameNum], 0)
currentFrameGrayContAdj = cv.equalizeHist(currentFrameGray)
plt.imshow(currentFrameGray, 'gray')

raft_centers, raft_radii, raft_count = fsr.find_and_sort_circles(currentFrameGray, num_of_rafts, prev_pos=prev_pos,
                                                                 radii_hough=radii_Hough, thres_value=thres_value,
                                                                 sigma_Canny=sigma_Canny,
                                                                 low_threshold_canny=low_threshold_canny,
                                                                 high_threshold_canny=high_threshold_canny,
                                                                 max_displ=max_displ)

# below is for detect_by_contours
raft_centers, raft_radii = fsr.detect_by_contours(currentFrameGray)

original = currentFrameGray.copy()
lowcut = original.mean() + 1.0 * original.std()
retval, image_thres = cv.threshold(original, lowcut, 255, cv.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
image = cv.morphologyEx(image_thres, cv.MORPH_OPEN, kernel)

_, contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

centers = []
radii = []
for contour in contours:
    area = cv.contourArea(contour)
    # there is one contour that contains all others, filter it out, Area can be moved to function definition also.
    if area < 500:
        continue
    center, br2 = cv.minEnclosingCircle(contour)
    # I tried to match the raft boundary using this 0.9
    radii.append(br2)
    centers.append(center)
    raft_centers = np.array(centers, dtype=int)
    raft_radii = np.array(radii, dtype=int)

# raft_centers, raft_radii = detect_by_contours(currentFrameGray)


# draw circles.
curr_frame_draw = currentFrameBGR.copy()
curr_frame_draw = fsr.draw_rafts(curr_frame_draw, raft_centers, raft_radii, num_of_rafts)
# curr_frame_draw = cv.circle(curr_frame_draw, (int(currentFrameGray.shape[1]/2), int(currentFrameGray.shape[0]/2)),
#                             lookup_radius, (255, 0, 0), int(2))
curr_frame_draw = fsr.draw_raft_number(curr_frame_draw, raft_centers, num_of_rafts)
# cv.imshow('analyzed image with circles', curr_frame_draw)
plt.imshow(curr_frame_draw[:, :, ::-1])

curr_frame_draw = currentFrameBGR.copy()
curr_frame_draw = fsr.draw_rafts(curr_frame_draw, (currentFrameGray.shape[1] / 2, currentFrameGray.shape[0] / 2),
                                 lookup_radius, 1)
cv.imshow('analyzed image', curr_frame_draw)

# %% debugging code for getting the rotation angle

# read two frames:
# Reading files
tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()
frameNum1 = 2
frameNum2 = 3
frame1BGR = cv.imread(tiffFileList[frameNum1])
frame2BGR = cv.imread(tiffFileList[frameNum2])
frame1Gray = frame1BGR[:, :, 1]  # green channel has the highest contrast
frame2Gray = frame2BGR[:, :, 1]
plt.imshow(frame1Gray, 'gray')
plt.figure()
plt.imshow(frame2Gray, 'gray')

raft_centers_1, raft_radii_1, raft_count_1 = fsr.find_circles_adaptive(frame1Gray, num_of_rafts,
                                                                       radii_hough=radii_Hough,
                                                                       adaptive_thres_blocksize=adaptiveThres_blocksize,
                                                                       adaptive_thres_const=adaptiveThres_const,
                                                                       min_sep_dist=min_sep_dist,
                                                                       raft_center_threshold=raft_center_threshold,
                                                                       top_left_x=topLeft_x, top_left_y=topLeft_y,
                                                                       width_x=width_x, height_y=height_y)
print(raft_count_1)

raft_centers_2, raft_radii_2, raft_count_2 = fsr.find_circles_adaptive(frame2Gray, num_of_rafts,
                                                                       radii_hough=radii_Hough,
                                                                       adaptive_thres_blocksize=adaptiveThres_blocksize,
                                                                       adaptive_thres_const=adaptiveThres_const,
                                                                       min_sep_dist=min_sep_dist,
                                                                       raft_center_threshold=raft_center_threshold,
                                                                       top_left_x=topLeft_x, top_left_y=topLeft_y,
                                                                       width_x=width_x, height_y=height_y)
print(raft_count_2)

# setting parameters and initialize dataset
numOfRafts = 2
sizeOfCroppedRaftImage = 150
raftLocations = np.zeros((numOfRafts, 2, 2), dtype=int)
# (raftNum, frameNum, x(columns)&y(rows), note that only two frames
raftOrientations = np.zeros((numOfRafts, 2))  # (raftNum, frameNum)
raftRadii = np.zeros((numOfRafts, 2), dtype=int)  # (raftNUm, frameNum)
firstImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))
currImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))
raftInitialOrientation = 0

# tracking part
raftLocations[:, 0, :] = raft_centers_1
raftRadii[:, 0] = raft_radii_1
targetID = fsr.tracking_rafts(raft_centers_1, raft_centers_2)
for raftID in np.arange(numOfRafts):
    raftLocations[raftID, 1, :] = raft_centers_2[targetID[raftID], :]
    raftRadii[raftID, 1] = raft_radii_2[targetID[raftID]]

# setting the orientation in the first image as initial orientation 0
currentFrameNum = 0
for raftID in np.arange(numOfRafts):
    firstImages[raftID, :, :] = fsr.crop_image(frame1Gray, raftLocations[raftID, currentFrameNum, :],
                                               sizeOfCroppedRaftImage)
    raftOrientations[raftID, currentFrameNum] = raftInitialOrientation
frame1Draw = frame1BGR.copy()
frame1Draw = fsr.draw_raft_orientations(frame1Draw, raftLocations[:, currentFrameNum, :],
                                        raftOrientations[:, currentFrameNum], raftRadii[:, currentFrameNum], numOfRafts)
plt.figure()
plt.imshow(frame1Draw[:, :, ::-1])

# obtain the orientation of rafts in the second image
currentFrameNum = 1
for raftID in np.arange(numOfRafts):
    currImages[raftID, :, :] = fsr.crop_image(frame2Gray, raftLocations[raftID, currentFrameNum, :],
                                              sizeOfCroppedRaftImage)
    rotationAngle = fsr.get_rotation_angle(firstImages[raftID, :, :], currImages[raftID, :, :], 15)
    raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, 0] + rotationAngle
    while raftOrientations[raftID, currentFrameNum] < 0:
        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] + 360
    while raftOrientations[raftID, currentFrameNum] > 360:
        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] - 360
frame2Draw = frame2BGR.copy()
frame2Draw = fsr.draw_raft_orientations(frame2Draw, raftLocations[:, currentFrameNum, :],
                                        raftOrientations[:, currentFrameNum], raftRadii[:, currentFrameNum], numOfRafts)
plt.figure()
plt.imshow(frame2Draw[:, :, ::-1])


def get_rotation_angle(prev_image, curr_image):
    """
    extract the angle of rotation theta between two frames
    """
    max_value = np.amax(prev_image)

    if prev_image.dtype == 'float' and max_value <= 1:
        prev_image = np.uint8(prev_image * 255)
        curr_image = np.uint8(curr_image * 255)

    if prev_image.dtype == 'float' and max_value > 1:
        prev_image = np.uint8(prev_image)
        curr_image = np.uint8(curr_image)

    prev_image = cv.equalizeHist(prev_image)
    curr_image = cv.equalizeHist(curr_image)

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=20)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(prev_image, None)
    kp2, des2 = orb.detectAndCompute(curr_image, None)

    # do feature matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # calculate perspective transform matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    transformMatrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    if transformMatrix is None:
        transformMatrix, mask = cv.findHomography(src_pts, dst_pts, 0)

    if transformMatrix is None:
        transformMatrix, mask = cv.findHomography(src_pts, dst_pts, 0)

    vector_along_x_axis_from_center = np.float32([[sizeOfCroppedRaftImage / 2, sizeOfCroppedRaftImage / 2],
                                                  [sizeOfCroppedRaftImage, sizeOfCroppedRaftImage / 2]]).reshape(-1, 1,
                                                                                                                 2)
    vector_transformed = cv.perspectiveTransform(vector_along_x_axis_from_center, transformMatrix)

    theta = - np.arctan2(vector_transformed[1, 0, 1] - vector_transformed[0, 0, 1],
                         vector_transformed[1, 0, 0] - vector_transformed[0, 0, 0]) * 180 / np.pi
    # negative sign is to make the sign of the angle in the right-handed coordinate

    return theta


raftID = 0

prev_image = firstImages[raftID, :, :]
plt.figure()
plt.imshow(prev_image, 'gray')

curr_image = currImages[raftID, :, :]
plt.figure()
plt.imshow(curr_image, 'gray')

rotationAngle = get_rotation_angle(prev_image, curr_image)

max_value = np.amax(prev_image)

if prev_image.dtype == 'float' and max_value <= 1:
    img1 = np.uint8(prev_image * 255)
    img2 = np.uint8(curr_image * 255)

if prev_image.dtype == 'float' and max_value > 1:
    img1 = np.uint8(prev_image)
    img2 = np.uint8(curr_image)

img1_hist = cv.equalizeHist(img1)
img2_hist = cv.equalizeHist(img2)

plt.figure()
plt.imshow(img1_hist, 'gray')
plt.figure()
plt.imshow(img2_hist, 'gray')

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# draw key points to see what features are:
img1_kp1 = cv.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
plt.figure()
plt.imshow(img1_kp1)

img2_kp2 = cv.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
plt.figure()
plt.imshow(img2_kp2)

# do feature matching
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# draw matches
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

plt.figure()
plt.imshow(img3)

# calculate perspective transform matrix
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
transformMatrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

if transformMatrix is None:
    transformMatrix, mask = cv.findHomography(src_pts, dst_pts, 0)

if transformMatrix is None:
    transformMatrix, mask = cv.findHomography(src_pts, dst_pts, 0)

vector_along_x_axis_from_center = np.float32([[sizeOfCroppedRaftImage / 2, sizeOfCroppedRaftImage / 2],
                                              [sizeOfCroppedRaftImage, sizeOfCroppedRaftImage / 2]]).reshape(-1, 1, 2)
vector_transformed = cv.perspectiveTransform(vector_along_x_axis_from_center, transformMatrix)

theta = - np.arctan2(vector_transformed[1, 0, 1] - vector_transformed[0, 0, 1],
                     vector_transformed[1, 0, 0] - vector_transformed[0, 0, 0]) * 180 / np.pi
# negative sign is to make the sign of the angle the same as in rhino, i.e. counter-clock wise from x-axis is positive

# cv.imshow('first image', firstImages[0,:,:])
plt.imshow(firstImages[0, :, :], 'gray')

plt.imshow(currImages[0, :, :], 'gray')

# %% using subtraction to evaluate the number of rafts that are stepped out.

# read two frames:
# Reading files
tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()
frameNum1 = 2
frameNum2 = 3
frame1BGR = cv.imread(tiffFileList[frameNum1])
frame2BGR = cv.imread(tiffFileList[frameNum2])
frame1Gray = frame1BGR[:, :, 1]  # green channel has the highest contrast
frame2Gray = frame2BGR[:, :, 1]
# plt.figure()
# plt.imshow(frame1Gray, 'gray')
# plt.figure()
# plt.imshow(frame2Gray, 'gray')

raft_centers_1, raft_radii_1, raft_count_1 = \
    fsr.find_circles_adaptive(frame1Gray, num_of_rafts, radii_hough=radii_Hough,
                              adaptive_thres_blocksize=adaptiveThres_blocksize,
                              adaptive_thres_const=adaptiveThres_const,
                              min_sep_dist=min_sep_dist, raft_center_threshold=raft_center_threshold,
                              top_left_x=topLeft_x, top_left_y=topLeft_y, width_x=width_x, height_y=height_y)
print(raft_count_1)

raft_centers_2, raft_radii_2, raft_count_2 = \
    fsr.find_circles_adaptive(frame2Gray, num_of_rafts, radii_hough=radii_Hough,
                              adaptive_thres_blocksize=adaptiveThres_blocksize,
                              adaptive_thres_const=adaptiveThres_const,
                              min_sep_dist=min_sep_dist,
                              raft_center_threshold=raft_center_threshold,
                              top_left_x=topLeft_x, top_left_y=topLeft_y, width_x=width_x, height_y=height_y)
print(raft_count_2)

# initialize key varialbes
numOfRafts = 211
sizeOfCroppedRaftImage = 40
raftLocations = np.zeros((numOfRafts, 2, 2), dtype=int)  # (raftNum, frameNum, x(columns)&y(rows)
raftOrientationDiff = np.zeros((numOfRafts, 2))  # (raftNum, numOfframes)
shiftingLength = 7
raftMatchingShifts = np.zeros((numOfRafts, 2, 2), dtype=int)  # (numOfRafts, numOfFrames, shiftingInX & shiftingInY)
firstImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))
currImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))

# tracking the rafts
raftLocations[:, 0, :] = raft_centers_1
targetID = fsr.tracking_rafts(raft_centers_1, raft_centers_2)
for raftID in np.arange(numOfRafts):
    raftLocations[raftID, 1, :] = raft_centers_2[targetID[raftID], :]

# crop and store firstImages
currentFrameNum = 0
for raftID in np.arange(numOfRafts):
    firstImages[raftID, :, :] = fsr.crop_image(frame1Gray, raftLocations[raftID, currentFrameNum, :],
                                               sizeOfCroppedRaftImage)
frame1Draw = frame1BGR.copy()

# crop and store currImages
currentFrameNum = 1
for raftID in np.arange(numOfRafts):
    currImages[raftID, :, :] = fsr.crop_image(frame2Gray, raftLocations[raftID, currentFrameNum, :],
                                              sizeOfCroppedRaftImage)

# calculate the difference between two successive frames with possible shifts
for raftID in np.arange(numOfRafts):
    rollImagesTest = np.zeros((shiftingLength, shiftingLength))
    for indexX in np.arange(shiftingLength):
        for indexY in np.arange(shiftingLength):
            shiftInX = indexX - shiftingLength // 2
            shiftInY = indexY - shiftingLength // 2
            rollImagesTest[indexX, indexY] = np.abs(
                firstImages[raftID, :, :] - np.roll(currImages[raftID, :, :], (shiftInX, shiftInY), (0, 1))).mean()
    raftOrientationDiff[raftID, 1] = rollImagesTest.min()
    raftMatchingShifts[raftID, 1, 0] = np.nonzero(rollImagesTest == rollImagesTest.min())[0][0] - shiftingLength // 2
    raftMatchingShifts[raftID, 1, 1] = np.nonzero(rollImagesTest == rollImagesTest.min())[1][0] - shiftingLength // 2

plt.figure()
plt.plot(raftOrientationDiff[:, 1], '-o')

thresholdDiff = 9

rotatedRaftIDs = np.nonzero(raftOrientationDiff[:, 1] > thresholdDiff)
print(len(rotatedRaftIDs[0]))

# fig = plt.figure(figsize = (len(rotatedRaftIDs[0])*9,18))
numOfRaftsSteppedOut = len(rotatedRaftIDs[0])
fig, ax = plt.subplots(nrows=2, ncols=numOfRaftsSteppedOut, figsize=(4 * numOfRaftsSteppedOut, 4), sharex=True)
for counter, raftID in enumerate(rotatedRaftIDs[0]):
    if numOfRaftsSteppedOut == 1:
        ax[0].imshow(firstImages[raftID, :, :], 'gray')
        ax[0].set_title('raftID = {}'.format(raftID))
        ax[1].imshow(np.roll(currImages[raftID, :, :],
                             (raftMatchingShifts[raftID, 1, 0], raftMatchingShifts[raftID, 1, 1]), (0, 1)), 'gray')
    elif numOfRaftsSteppedOut > 1:
        ax[0, counter].imshow(firstImages[raftID, :, :], 'gray')
        ax[0, counter].set_title('raftID = {}'.format(raftID))
        ax[1, counter].imshow(np.roll(currImages[raftID, :, :],
                                      (raftMatchingShifts[raftID, 1, 0], raftMatchingShifts[raftID, 1, 1]),
                                      (0, 1)), 'gray')

fig.savefig('steppedOutRafts')

raftID = 6

prev_image = firstImages[raftID, :, :]
plt.figure()
plt.imshow(prev_image, 'gray')

print(raftMatchingShifts[raftID, 1, :])
curr_image = np.roll(currImages[raftID, :, :], (raftMatchingShifts[raftID, 1, 0], raftMatchingShifts[raftID, 1, 1]),
                     (0, 1))
plt.figure()
plt.imshow(curr_image, 'gray')

diff_image = np.abs(prev_image - curr_image)
plt.figure()
plt.imshow(diff_image, 'gray')

plt.figure()
curr_frame_draw = frame1BGR.copy()
curr_frame_draw = fsr.draw_rafts(curr_frame_draw, raft_centers_1, raft_radii_1, num_of_rafts)
# curr_frame_draw = cv.circle(curr_frame_draw, (int(currentFrameGray.shape[1]/2), int(currentFrameGray.shape[0]/2)),
#                             lookup_radius, (255,0,0), int(2))
curr_frame_draw = fsr.draw_raft_number(curr_frame_draw, raft_centers_1, num_of_rafts)
# cv.imshow('analyzed image with circles', curr_frame_draw)
plt.imshow(curr_frame_draw[:, :, ::-1])

# %% track circle using Hungarian Algorithm; effused rafts function definition
# Reading files
tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()
prevFrameNum = 0
currFrameNum = prevFrameNum + 1
prevFrameBGR = cv.imread(tiffFileList[prevFrameNum])
currFrameBGR = cv.imread(tiffFileList[currFrameNum])
# prevFrameGray = cv.imread(tiffFileList[prevFrameNum], 0)
# currFrameGray = cv.imread(tiffFileList[currFrameNum], 0)
prevFrameGray = prevFrameBGR[:, :, 1]  # green channel has the highest contrast
currFrameGray = currFrameBGR[:, :, 1]
plt.imshow(prevFrameGray, 'gray')
plt.imshow(currFrameGray, 'gray')

numOfTotalRafts = 232
numOfFrames = len(tiffFileList)

# parameters initialization
num_of_rafts = 5
radii_Hough = [14, 17]
thres_value = 50
adaptiveThres_blocksize = 9
adaptiveThres_const = -15
raft_center_threshold = 74
sigma_Canny = 1
low_threshold_canny = 25
high_threshold_canny = 127
min_sep_dist = 40
lookup_radius = 880  # unit: pixel
topLeft_x = 150
topLeft_y = 970
width_x = 150
height_y = 250
max_displacement = 15
error_message = ' '

raftLocations = np.zeros((numOfTotalRafts, numOfFrames, 2), dtype=int)  # (raftNum, frameNum, x(columns)&y(rows)
raftRadii = np.zeros((numOfTotalRafts, numOfFrames), dtype=int)

prev_centers = np.zeros((num_of_rafts, 2), dtype=int)
prev_radii = np.zeros(num_of_rafts, dtype=int)
prev_flags = np.zeros(num_of_rafts, dtype=int)  # -1 meaning enter from left, 1 meaning enter from right
curr_centers = np.zeros((num_of_rafts, 2), dtype=int)
curr_radii = np.zeros(num_of_rafts, dtype=int)
curr_flags = np.zeros(num_of_rafts, dtype=int)

curr_centers_tracked = np.zeros((num_of_rafts, 2), dtype=int)
curr_radii_tracked = np.zeros(num_of_rafts, dtype=int)

# FindCircles in the previous frame
prev_centers, prev_radii, prev_count = fsr.find_circles_thres(prevFrameGray, num_of_rafts, radii_hough=radii_Hough,
                                                              thres_value=thres_value, sigma_canny=sigma_Canny,
                                                              low_threshold_canny=low_threshold_canny,
                                                              high_threshold_canny=high_threshold_canny,
                                                              min_sep_dist=min_sep_dist,
                                                              raft_center_threshold=raft_center_threshold,
                                                              top_left_x=topLeft_x, top_left_y=topLeft_y, width_x=width_x,
                                                              height_y=height_y)
# FindCircles in the current frame
curr_centers, curr_radii, curr_count = fsr.find_circles_adaptive(currFrameGray, num_of_rafts, radii_hough=radii_Hough,
                                                                 adaptive_thres_blocksize=adaptiveThres_blocksize,
                                                                 adaptive_thres_const=adaptiveThres_const,
                                                                 min_sep_dist=min_sep_dist,
                                                                 raft_center_threshold=raft_center_threshold,
                                                                 top_left_x=topLeft_x, top_left_y=topLeft_y,
                                                                 width_x=width_x, height_y=height_y)

boudnary_x = topLeft_x + width_x // 2


def counting_effused_rafts(prev_centers, prev_count, curr_centers, curr_count, boundary_x, max_dispacement):
    """
    test if the raft crosses the boundary of container
    """
    effused_raft_change = 0
    cost_matrix = scipy_distance.cdist(prev_centers[:prev_count], curr_centers[:curr_count], 'euclidean')
    #  note that row index refers to previous raft number, column index refers to current raft number

    # select the boundary crossing to be in the middle of the cropped image, so only deals with existing rafts
    for raft_id in np.arange(prev_count):
        if np.any(cost_matrix[raft_id, :] < max_displacement):  # raft still exist
            curr_raft_id = np.nonzero(cost_matrix[raft_id, :] < max_displacement)[0][
                0]  # [0][0] is to convert array into scalar
            if (prev_centers[raft_id, 0] > boundary_x) and (curr_centers[curr_raft_id, 0] <= boundary_x):
                effused_raft_change = effused_raft_change + 1
            elif (prev_centers[raft_id, 0] < boundary_x) and (curr_centers[curr_raft_id, 0] >= boundary_x):
                effused_raft_change = effused_raft_change - 1
    return effused_raft_change


# targetID = tracking_rafts(prev_centers, curr_centers, num_of_rafts)

costMatrix = scipy_distance.cdist(prev_centers[:prev_count], curr_centers[:curr_count], 'euclidean')
row_ind, col_ind = linear_sum_assignment(costMatrix)

targetID = col_ind

# Hungarian algorithm to track
# def tracking_rafts(prev_rafts_locations, detected_centers):
#    ''' sort the detected_centers according to the locations of rafts in the previous frame
#
#    the row number of col_ind is raft number in prev_rafts_locations,
#    the value in col_ind is the corresponding raft number in the detected_centers
#    '''
#    costMatrix = scipyDistance.cdist(prev_rafts_locations, detected_centers, 'euclidean')
#    row_ind, col_ind = linear_sum_assignment(costMatrix)
#
#    return col_ind
#
#
# old tracking algorithm based on minimal distance.
# def tracking_rafts(prev_rafts_locations, detected_centers, num_of_rafts):
#    ''' sort the detected_centers according to the locations of rafts in the previous frame
#    '''
#    target_id = np.zeros((num_of_rafts,1), dtype = int)
#
#    for prev_raft_id in np.arange(num_of_rafts): # looping through the rafs location in the prev_rafts_locations
#        min_seperation = 10000
#        for test_id in np.arange(num_of_rafts): # looping through the rafs location in the detected_centers
#            seperation = calculate_distance(prev_rafts_locations[prev_raft_id,:], detected_centers[test_id,:])
#            if seperation < min_seperation:
#                min_seperation = seperation
#                target_id[prev_raft_id] = test_id
#
#    return target_id


for raftID in np.arange(num_of_rafts):
    if raftID in targetID:
        curr_centers_tracked[raftID, :] = curr_centers[targetID[raftID], :]
        curr_radii_tracked[raftID] = curr_radii[targetID[raftID]]
    else:
        curr_centers_tracked[raftID, :] = curr_centers[raftID, :]
        curr_radii_tracked[raftID] = curr_radii[raftID]

prev_frame_draw = prevFrameBGR.copy()
prev_frame_draw = fsr.draw_rafts(prev_frame_draw, prev_centers, prev_radii, num_of_rafts)
# prev_frame_draw = cv.circle(prev_frame_draw, (int(prevFrameGray.shape[1]/2),
#                                               int(prevFrameGray.shape[0]/2)), lookup_radius, (255,0,0), int(2))
prev_frame_draw = fsr.draw_raft_number(prev_frame_draw, prev_centers, num_of_rafts)
# cv.imshow('analyzed image with first two circles', prev_frame_draw)
plt.imshow(prev_frame_draw[:, :, ::-1])

curr_frame_draw = currFrameBGR.copy()
curr_frame_draw = fsr.draw_rafts(curr_frame_draw, curr_centers_tracked, curr_radii_tracked, num_of_rafts)
# curr_frame_draw = cv.circle(curr_frame_draw, (int(currFrameGray.shape[1]/2),
#                                               int(currFrameGray.shape[0]/2)), lookup_radius, (255,0,0), int(2))
curr_frame_draw = fsr.draw_raft_number(curr_frame_draw, curr_centers_tracked, num_of_rafts)
# cv.imshow('analyzed image with first two circles', curr_frame_draw)
plt.imshow(curr_frame_draw[:, :, ::-1])
