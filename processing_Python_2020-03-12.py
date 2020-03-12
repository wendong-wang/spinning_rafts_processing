# -*- coding: utf-8 -*-
"""
@author: wwang

processing script for spinning rafts systems

----------- Table of Contents

- import necessary libraries and define functions
- data processing
- debugging code for find circles using scikit image
- debugging code for getting the rotation angle
- using substraction to evalute the number of rafts that are stepped out
- track circle using Hungarian Algorithm; effused rafts function definition 

"""
# %% import necessary libraries and define functions

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


def FindCirclesThres(current_frame_gray, num_of_rafts, radii_Hough=[17, 19],
                     thres_value=70, adaptiveThres_blocksize=9, adaptiveThres_const=-20,
                     sigma_Canny=1.0, low_threshold_canny=25, high_threshold_canny=127,
                     min_sep_dist=20, lookup_radius=768, raft_center_threshold=60,
                     topLeft_x=390, topLeft_y=450, width_x=850, height_y=850, error_message=' '):
    """
    find the centers of each raft

    current_frame_gray: gray scale image
    num_of_rafts: number of rafts to be located
    radii_Hough: [starting radius, ending radius, intervale], to be unpacked as an argument for hough_circle
    """
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # crop the image
    image_cropped = current_frame_gray[topLeft_y: topLeft_y + height_y, topLeft_x: topLeft_x + width_x]

    # threshold the image
    retval, image_thres = cv.threshold(image_cropped, thres_value, 255, 0)

    # find edges
    image_edges = canny(image_thres, sigma=sigma_Canny, low_threshold=low_threshold_canny,
                        high_threshold=high_threshold_canny)

    # use Hough transform to find circles
    hough_results = hough_circle(image_edges, np.arange(*radii_Hough))
    accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_Hough))

    # assuming that the first raft (highest accumulator score) is a good one
    #    raft_centers[0,0] = cx[0]
    #    raft_centers[0,1] = cy[0]
    #    raft_radii[0] = radii[0]
    raft_count = 0  # starting from 1!

    # remove circles that belong to the same raft and circles that happened to be in between rafts
    for accumScore, detected_cx, detected_cy, detected_radius in zip(accums, cx, cy, radii):
        new_raft = 1
        if image_cropped[detected_cy, detected_cx] < raft_center_threshold:
            new_raft = 0
        elif image_cropped[detected_cy - detected_radius // 2: detected_cy + detected_radius // 2,
             detected_cx - detected_radius // 2:detected_cx + detected_radius // 2].mean() < raft_center_threshold:
            new_raft = 0
        #        elif  (detected_cx - width_x/2)**2 +  (detected_cy - height_y/2)**2 > lookup_radius**2:
        #            new_raft = 0
        else:
            costMatrix = scipy_distance.cdist(np.array([detected_cx, detected_cy], ndmin=2),
                                              raft_centers[:raft_count, :], 'euclidean')
            if np.any(costMatrix < min_sep_dist):  # raft still exist
                new_raft = 0
        if new_raft == 1:
            raft_centers[
                raft_count, 0] = detected_cx  # note that raft_count starts with 1, also note that cx corresonds to columns number
            raft_centers[raft_count, 1] = detected_cy  # cy is row number
            raft_radii[raft_count] = detected_radius
            raft_count = raft_count + 1
        if raft_count == num_of_rafts:
            #            error_message = 'all rafts found'
            break

    # convert the xy coordinates of the cropped image into the coordinates of the original image
    raft_centers[:, 0] = raft_centers[:, 0] + topLeft_x
    raft_centers[:, 1] = raft_centers[:, 1] + topLeft_y

    return raft_centers, raft_radii, raft_count


def FindCirclesAdaptive(current_frame_gray, num_of_rafts, radii_Hough=[17, 19], thres_value=70,
                        adaptiveThres_blocksize=9, adaptiveThres_const=-20, sigma_Canny=1.0, low_threshold_canny=25,
                        high_threshold_canny=127, min_sep_dist=20, lookup_radius=768, raft_center_threshold=60,
                        topLeft_x=390, topLeft_y=450, width_x=850, height_y=850, error_message=' '):
    ''' find the centers of each raft
    
    current_frame_gray: gray scale image
    num_of_rafts: number of rafts to be located 
    radii_Hough: [starting radius, ending radius, intervale], to be unpacked as an argument for hough_circle
    '''
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # crop the image
    image_cropped = current_frame_gray[topLeft_y: topLeft_y + height_y, topLeft_x: topLeft_x + width_x]

    # threshold the image
    image_thres = cv.adaptiveThreshold(image_cropped, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                       adaptiveThres_blocksize, adaptiveThres_const)

    # use Hough transform to find circles
    hough_results = hough_circle(image_thres, np.arange(*radii_Hough))
    accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_Hough))

    # assuming that the first raft (highest accumulator score) is a good one
    #    raft_centers[0,0] = cx[0]
    #    raft_centers[0,1] = cy[0]
    #    raft_radii[0] = radii[0]
    raft_count = 0  # starting from 1!

    # remove circles that belong to the same raft and circles that happened to be in between rafts
    for accumScore, detected_cx, detected_cy, detected_radius in zip(accums, cx, cy, radii):
        new_raft = 1
        if image_cropped[detected_cy, detected_cx] < raft_center_threshold:
            new_raft = 0
        elif image_cropped[detected_cy - detected_radius // 2: detected_cy + detected_radius // 2,
             detected_cx - detected_radius // 2:detected_cx + detected_radius // 2].mean() < raft_center_threshold:
            new_raft = 0
        #        elif  (detected_cx - width_x/2)**2 +  (detected_cy - height_y/2)**2 > lookup_radius**2:
        #            new_raft = 0
        else:
            costMatrix = scipy_distance.cdist(np.array([detected_cx, detected_cy], ndmin=2),
                                              raft_centers[:raft_count, :], 'euclidean')
            if np.any(costMatrix < min_sep_dist):  # raft still exist
                new_raft = 0
        if new_raft == 1:
            raft_centers[
                raft_count, 0] = detected_cx  # note that raft_count starts with 1, also note that cx corresonds to columns number
            raft_centers[raft_count, 1] = detected_cy  # cy is row number
            raft_radii[raft_count] = detected_radius
            raft_count = raft_count + 1
        if raft_count == num_of_rafts:
            #            error_message = 'all rafts found'
            break

    # convert the xy coordinates of the cropped image into the coordinates of the original image
    raft_centers[:, 0] = raft_centers[:, 0] + topLeft_x
    raft_centers[:, 1] = raft_centers[:, 1] + topLeft_y

    return raft_centers, raft_radii, raft_count


def FindAndSortCircles(image_gray, num_of_rafts, prev_pos, radii_Hough=[30, 40], thres_value=30, sigma_Canny=1.0,
                       low_threshold_canny=25, high_threshold_canny=127, max_displ=50):
    """
    For each raft detected in the prev_pos, go through the newly found circles in descending order of scores,
    and the first one within max_displ is the stored as the new position of the raft.

    image_gray: gray scale image
    num_of_rafts: number of rafts to be located
    prev_pos: previous positions of rafts
    radii_Hough: [starting radius, ending radius, intervale], to be unpacked as an argument for hough_circle
    sigma_Canny: the width of the Gaussian filter for Canny edge detection
    low_threshold_canny: low threshold for Canny
    high_threshold_canny: high threshold for Canny
    """
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # threshold the image first
    retval, image_thres = cv.threshold(currentFrameGray, thres_value, 255, 0)
    #    kernel = np.ones((3,3),np.uint8)
    #    image_thres = cv.morphologyEx(image_thres, cv.MORPH_OPEN, kernel)

    # use canny and then Hough transform to find circles
    image_edges = canny(image_thres, sigma=sigma_Canny, low_threshold=low_threshold_canny,
                        high_threshold=high_threshold_canny)
    hough_results = hough_circle(image_edges, np.arange(*radii_Hough))
    accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_Hough))

    raft_count = 0
    for raftID in np.arange(num_of_rafts):
        for accumScore, detected_cx, detected_cy, detected_radius in zip(accums, cx, cy, radii):
            distance = np.sqrt((detected_cx - prev_pos[raftID, 0]) ** 2 + (detected_cy - prev_pos[raftID, 1]) ** 2)
            if distance < max_displ:
                raft_centers[
                    raftID, 0] = detected_cx  # note that raft_count starts with 1, also note that cx corresonds to columns number
                raft_centers[raftID, 1] = detected_cy  # cy is row number
                raft_radii[raftID] = detected_radius
                raft_count += 1
                break

    return raft_centers, raft_radii, raft_count


def DetectByContours(image_gray):
    original = image_gray.copy()
    lowcut = original.mean() + 1.0 * original.std()
    retval, image_thres = cv.threshold(original, lowcut, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.morphologyEx(image_thres, cv.MORPH_OPEN, kernel)
    _, contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #    drawing = test_image.copy()
    centers = []
    radii = []
    for contour in contours:
        area = cv.contourArea(contour)
        # there is one contour that contains all others, filter it out, Area can be moved to function definition also. 
        if area < 2000:
            continue
        center, br2 = cv.minEnclosingCircle(contour)
        # I tried to match the raft boundary using this 0.9
        radii.append(br2)
        centers.append(center)
        raft_centers = np.array(centers, dtype=int)
        raft_radii = np.array(radii, dtype=int)
    return raft_centers, raft_radii


def ParseMainFolderName(main_folder_name):
    ''' parse the name of the main folder here, and return the follwing parts
    date, string
    raft_geometry, string
    thin_film_prop, string
    magnet_field_prop, string
    comments, string
    '''
    parts = main_folder_name.split('_')

    date = parts[0]
    raft_geometry = parts[1]
    thin_film_prop = parts[2]
    magnet_field_prop = parts[3]

    if len(parts) > 4:
        comments = parts[4]
    else:
        comments = 'none'

    return date, raft_geometry, thin_film_prop, magnet_field_prop, comments


def ParseSubfolderName(subfolder_name):
    """ parse the subfolder name here, and return the following variables
    num_of_rafts, int
    batch_number, int
    spin_speed, int
    magnification, int
    comments, string
    """
    name_lowercase = subfolder_name.lower()
    parts = name_lowercase.split('_')

    num_of_rafts = int(parts[0].partition('raft')[0])
    batch_number = int(parts[1])
    spin_speed = float(parts[2].partition('rp')[0])
    spin_unit = ''.join(parts[2].partition('rp')[1:])
    #    if parts[2].partition('rp')[0].isdigit():
    #        spin_speed = int(parts[2].partition('rp')[0])
    #        spin_unit = ''.join(parts[2].partition('rp')[1:])
    #    elif parts[2].partition('hz')[0].isdigit():
    #        spin_speed = int(parts[2].partition('hz')[0])
    #        spin_unit = ''.join(parts[2].partition('hz')[1:])
    magnification = float(parts[3].partition('x')[0])

    if len(parts) > 4:
        comments = ''.join(parts[4:])
    else:
        comments = 'none'

    return num_of_rafts, batch_number, spin_speed, spin_unit, magnification, comments


def CalculateDistance(p1, p2):
    """
    calculate the distance between p1 and p2
    """

    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    return dist


def CalculateOrbitingAngle(orbiting_center, raft):
    """
    calculate the orbiting angle of a raft with respect to a center
    """

    # note the negative sign before the first component, which is y component 
    # the y in scikit-image is flipped. 
    # it is to make the value of the angle appears natural, as in Rhino, with x-axis pointing right, and y-axis pointing up. 
    angle = np.arctan2(-(raft[1] - orbiting_center[1]), (raft[0] - orbiting_center[0])) * 180 / np.pi

    return angle


def NumberingRafts(rafts_loc, rafts_radii, num_of_rafts):
    """
    sort the rafts into layers and number them sequentially from inner most layer to the outmost layer
    return sorted rafts_loc and rafts_radii, and layer index
    """
    orbiting_center = np.mean(rafts_loc, axis=0)
    orbiting_dist = np.sqrt((rafts_loc[:, 0] - orbiting_center[0]) ** 2 + (rafts_loc[:, 1] - orbiting_center[1]) ** 2)
    sorted_index = orbiting_dist.argsort()
    dist_sorted = orbiting_dist[sorted_index]
    rafts_loc_sorted = rafts_loc[sorted_index, :]
    rafts_radii_sorted = rafts_radii[sorted_index]

    # assign layer
    layer_index = np.ones(num_of_rafts, dtype=int)
    layer_num = 1
    for raft_id in np.arange(1, num_of_rafts):
        if dist_sorted[raft_id] - dist_sorted[raft_id - 1] > rafts_radii_sorted[raft_id]:
            layer_num = layer_num + 1
        layer_index[raft_id] = layer_num

    # calcuate orbiting angle, note the two negative signs in front of both y- and x- components. 
    # For y-component, it is for flipping image axis. For x-component, it is make the counting start at x-axis and go clockwise. Note the value of arctan2 is  [-pi, pi]
    orbiting_angles = np.arctan2(-(rafts_loc_sorted[:, 1] - orbiting_center[1]),
                                 -(rafts_loc_sorted[:, 0] - orbiting_center[0])) * 180 / np.pi

    # concatenate and sort
    rafts_loc_radii_dist_angle_layer = np.column_stack(
        (rafts_loc_sorted[:, 0], rafts_loc_sorted[:, 1], rafts_radii_sorted, dist_sorted, orbiting_angles, layer_index))

    sorted_index2 = np.lexsort((orbiting_angles, layer_index))

    rafts_loc_radii_dist_angle_layer_sorted = rafts_loc_radii_dist_angle_layer[sorted_index2]

    rafts_loc_sorted2 = rafts_loc_radii_dist_angle_layer_sorted[:, 0:2].astype(int)
    rafts_radii_sorted2 = rafts_loc_radii_dist_angle_layer_sorted[:, 2].astype(int)
    dist_sorted2 = rafts_loc_radii_dist_angle_layer_sorted[:, 3]
    angles_sorted2 = rafts_loc_radii_dist_angle_layer_sorted[:, 4]
    layer_index_sorted2 = rafts_loc_radii_dist_angle_layer_sorted[:, 5]

    return rafts_loc_sorted2, rafts_radii_sorted2, dist_sorted2, angles_sorted2, layer_index_sorted2


def CropImage(grayscale_image, raft_center, width):
    """
    crop the area of the raft
    """
    topRow = int(raft_center[
                     1] - width / 2)  # note that y corresponds to rows, and is directed from top to bottom in scikit-image
    bottomRow = int(raft_center[1] + width / 2)

    leftColumn = int(raft_center[0] - width / 2)
    rightColumn = int(raft_center[0] + width / 2)

    raft_image = grayscale_image[topRow:bottomRow, leftColumn:rightColumn]
    return raft_image


def TrackingRafts(prev_rafts_locations, detected_centers, num_of_rafts):
    """
    sort the detected_centers according to the locations of rafts in the previous frame

    the row number of col_ind is raft number in prev_rafts_locations,
    the value in col_ind is the corresponding raft number in the detected_centers
    """
    costMatrix = scipy_distance.cdist(prev_rafts_locations, detected_centers, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(costMatrix)

    return col_ind


def CountingEffusedRafts(prev_centers, prev_count, curr_centers, curr_count, boundary_x, max_displacement):
    """
    test if the raft crosses the boundary of container
    """
    effused_raft_toLeft = 0
    effused_raft_toRight = 0
    costMatrix = scipy_distance.cdist(prev_centers[:prev_count], curr_centers[:curr_count], 'euclidean')
    #  note that row index refers to previous raft number, column index refers to current raft number

    # select the boundary crossing to be in the middle of the cropped image, so only deals with existing rafts
    for raftID in np.arange(prev_count):
        if np.any(costMatrix[raftID, :] < max_displacement):  # raft still exist
            curr_raftID = np.nonzero(costMatrix[raftID, :] < max_displacement)[0][
                0]  # [0][0] is to convert array into scalar
            if (prev_centers[raftID, 0] >= boundary_x) and (curr_centers[curr_raftID, 0] < boundary_x):
                effused_raft_toLeft = effused_raft_toLeft + 1
            elif (prev_centers[raftID, 0] < boundary_x) and (curr_centers[curr_raftID, 0] >= boundary_x):
                effused_raft_toRight = effused_raft_toRight + 1
    return effused_raft_toLeft, effused_raft_toRight


def GetRotationAngle(prev_image, curr_image):
    """
    extract the angle of rotation theta between two frames
    """

    max_value = np.amax(prev_image)

    if prev_image.dtype == 'float' and max_value <= 1:
        img1 = np.uint8(prev_image * 255)
        img2 = np.uint8(curr_image * 255)

    if prev_image.dtype == 'float' and max_value > 1:
        img1 = np.uint8(prev_image)
        img2 = np.uint8(curr_image)

    img1 = cv.equalizeHist(img1)
    img2 = cv.equalizeHist(img2)

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=200)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

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
    # negative sign is to make the sign of the angle the same as in rhino, i.e. counter-clock wise from x-axis is positive

    return theta


def DrawRafts(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    """
    draw circles around rafts
    """

    circle_thickness = int(2)
    circle_color = (0, 0, 255)  # openCV: BGR

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1]), rafts_radii[raft_id],
                               circle_color, circle_thickness)

    return output_img


def DrawRaftOrientations(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicte the orientation of each raft
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)
    #    line_length = 20

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    int(rafts_loc[raft_id, 1] - np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]))
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def DrawRaftNumber(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    """

    fontFace = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 255)  # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id + 1), fontFace, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1),
                                (rafts_loc[raft_id, 0] - textSize[0] // 2, rafts_loc[raft_id, 1] + textSize[1] // 2),
                                fontFace, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def DrawEffusedRaftCount(img_bgr, raft_effused, raft_to_left, raft_to_right, topLeft_X, topLeft_Y, width_X, height_Y):
    """
    draw effused rafts
    """
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)  # BGR
    font_thickness = 2
    line_color = (0, 0, 255)  # BGR
    line_thickness = 1
    output_img = img_bgr
    output_img = cv.line(output_img, (topLeft_X + width_X // 2, topLeft_Y),
                         (topLeft_X + width_X // 2, topLeft_Y + height_Y), line_color, line_thickness)
    output_img = cv.putText(output_img, 'Effused: ' + str(raft_effused), (topLeftX, topLeftY - 30), fontFace,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'To left: ' + str(raft_to_left), (topLeftX, topLeftY - 60), fontFace,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'To right: ' + str(raft_to_right), (topLeftX, topLeftY - 90), fontFace,
                            font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


# %% data processing

# look into mainFolders and choose which ones to analyze
# rootFolderName = os.getcwd()
rootFolderNameFromWindows = r'D:\\VideoProcessingFolder'  # '/home/gardi/Rafts/Experiments Data/test'
os.chdir(rootFolderNameFromWindows)
rootFolderTreeGen = os.walk(rootFolderNameFromWindows)
_, mainFolders, _ = next(rootFolderTreeGen)

# set up parameters according to the type of data in the main folders
isVideo = 0  # 1: data is video, 0: data is image sequence
videoFormat = '*.MOV'
imageFormat = '*.tiff'

# parameters for various find-circle functions
# frequently-adjusted:
radiusIntervalHough = [14, 18]  # [71, 77] for 2.5x, [21, 25] for 0.8x, [14, 18] for 0.57x for 5x using coaxial illumination,  [45, 55] for 5x using ring light illumination, [10, 20] for multiple rafts 1 or 2x.
adaptiveThresBlocksize = 5  # 5, 13 #9 #19 #9
adaptiveThresConst = -13  # -9, 13
raftCenterThreshold = 40  # 30, 40, 50 #100 #74 #78 #80
minSepDist = 40
# cropping
topLeftX = 1300
topLeftY = 160
widthX = 70  # 1728 #1472 #130
heightY = 280  # 1728 #1400 #280
# maxim displacement, usually twice of the upper radius, used in tracking in effusions and FindAndSortCircles
maxDisplacement = 36
# not used FindCirclesAdaptive, but in the FindCirclesThres and FindAndSortCircles
thresholdingValue = 33
lowThresholdCanny = 25
highThresholdCanny = 127
sigmaCanny = 1.0
# an old parameter that rejects all circles outside a certain radius, not used anymore
lookupRadius = 880

effusionData = 1  # 1- this is an effusion data
effusionBoundaryX = topLeftX + widthX // 2
setRaftCountManual = 1  # override the reading from the subfolder name or movie file name; for effusion use
raftCountManual = 5

regionalSearch = 1  # look at only a small region of the image, to extract velocity of passing rafts
regionTopLeftX = 657
regionTopLeftY = 700
regionWidth = 100
regionHeight = 550
maxNumOfRaftsInRegion = 20

diffusionData = 0  # 1 - this is a diffusion data that only tracks one sigle particle.
diffBoxRadius = 50
diffBoxTopLeftX = topLeftX
diffBoxTopLeftY = topLeftY
diffBoxWidthX = widthX
diffBoxHeightY = heightY

# process rotation parameters
processRotation = 0
sizeOfCroppedRaftImage = 150  # unit pixel, at least twice the size of the radius, for analyzing rafts' orientations
raftInitialOrientation = 0  # the orientation of the rafts in the first frame.

outputImageSeq = 1  # whether to save all the frames
outputVideo = 1  # whether to save results in videos
outputFrameRate = 5.0

errorMessage = '_'

videoFileList = []
subfolders = []

listOfVarialbesToSave = ['batchNum', 'commentsMain', 'commentsSub', 'currentFrameDraw',
                         'currentFrameGray', 'currentFrameNum', 'currentFrameBGR', 'date',
                         'expID', 'errorMessage', 'isVideo', 'imageFormat', 'magneticFieldProp',
                         'magnification', 'mainFolderID', 'mainFolders', 'numOfExp',
                         'numOfFrames', 'numOfRafts', 'orbitingAnglesSorted', 'outputImageName', 'processRotation',
                         'radiusIntervalHough', 'raftGeo', 'raftID', 'raftInitialOrientation', 'raftLocations',
                         'raftOrbitingAngles', 'raftOrbitingCenters', 'raftOrbitingDistances',
                         'raftOrbitingLayerIndices', 'raftOrientations', 'raftRadii',
                         'rootFolderNameFromWindows',
                         'sizeOfCroppedRaftImage', 'spinSpeed', 'spinUnit', 'subfolders',
                         'thinFilmProp', 'videoFormat', 'videoFileList',
                         'adaptiveThresBlocksize', 'adaptiveThresConst',
                         'topLeftX', 'topLeftY', 'widthX', 'heightY', 'minSepDist', 'maxDisplacement',
                         'lookupRadius', 'raftEffused', 'raftToLeft', 'raftToRight',
                         'regionalSearch', 'regionTopLeftX', 'regionTopLeftY', 'regionWidth',
                         'regionHeight', 'maxNumOfRaftsInRegion', 'raftLocationsInRegion', 'raftRadiiInRegion']

for mainFolderID in np.arange(2, 3):
    os.chdir(mainFolders[mainFolderID])
    # parse the main folder name here
    date, raftGeo, thinFilmProp, magneticFieldProp, commentsMain = ParseMainFolderName(mainFolders[mainFolderID])

    if isVideo == 1:
        videoFileList = glob.glob(videoFormat)
        numOfExp = len(videoFileList)
    else:
        mainFolderTreeGen = os.walk(os.getcwd())
        _, subfolders, _ = next(mainFolderTreeGen)
        numOfExp = len(subfolders)

    for expID in range(0, numOfExp):
        if isVideo == 1:
            # parse video file name, initial video, get total number of frames
            numOfRafts, batchNum, spinSpeed, spinUnit, magnification, commentsSub = ParseSubfolderName(
                videoFileList[expID])
            outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps'
            if os.path.isfile(outputDataFileName + '.dat') == True:
                errorMessage = '{0}.dat exists'.format(outputDataFileName)
                print(errorMessage)
                continue
            cap = cv.VideoCapture(videoFileList[expID])
            numOfFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        else:
            # parse the subfolder name, read file list, get total number of frames
            numOfRafts, batchNum, spinSpeed, spinUnit, magnification, commentsSub = ParseSubfolderName(
                subfolders[expID])
            outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
                spinSpeed) + 'rps_' + str(magnification) + 'x_' + commentsSub
            if os.path.isfile(outputDataFileName + '.dat') == True:
                errorMessage = '{0}.dat exists'.format(outputDataFileName)
                print(errorMessage)
                continue
            os.chdir(subfolders[expID])
            tiffFileList = glob.glob(imageFormat)
            tiffFileList.sort()
            numOfFrames = len(tiffFileList)

        # for effusion, only count the raft inside the cropped area
        if setRaftCountManual == 1:
            numOfRafts = raftCountManual

        # initialize key data set
        raftLocations = np.zeros((numOfRafts, numOfFrames, 2), dtype=int)  # (raftNum, frameNum, x(columns)&y(rows)
        raftRadii = np.zeros((numOfRafts, numOfFrames), dtype=int)
        raftOrientations = np.zeros((numOfRafts, numOfFrames))
        raftOrbitingCenters = np.zeros((numOfFrames, 2))  #
        raftOrbitingDistances = np.zeros((numOfRafts, numOfFrames))
        raftOrbitingAngles = np.zeros((numOfRafts, numOfFrames))
        raftOrbitingLayerIndices = np.zeros((numOfRafts, numOfFrames))
        if processRotation == 1:
            firstImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))
            currImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))
        raftEffused = np.zeros((numOfFrames), dtype=int)
        raftToLeft = np.zeros((numOfFrames), dtype=int)
        raftToRight = np.zeros((numOfFrames), dtype=int)
        effusedRaftCount = 0
        raftMovingToLeftCount = 0
        raftMovingToRightCount = 0
        raftLocationsInRegion = np.zeros((maxNumOfRaftsInRegion, numOfFrames, 2),
                                         dtype=int)  # (raftNum, frameNum, x(columns)&y(rows)
        raftRadiiInRegion = np.zeros((maxNumOfRaftsInRegion, numOfFrames), dtype=int)

        ## read and process the first frame
        currentFrameNum = 0
        if isVideo == 1:
            retval, currentFrameBGR = cap.read()
            currentFrameGray = cv.cvtColor(currentFrameBGR, cv.COLOR_BGR2GRAY)
        else:
            currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
            #            currentFrameGray = cv.imread(tiffFileList[currentFrameNum],0)
            currentFrameGray = currentFrameBGR[:, :,
                               1]  # use only green channel. We found green channel has the highest contrast.

        # find cricles in the first frame
        centers, radii, prevCount = FindCirclesAdaptive(currentFrameGray, numOfRafts, radii_Hough=radiusIntervalHough,
                                                        thres_value=thresholdingValue,
                                                        adaptiveThres_blocksize=adaptiveThresBlocksize,
                                                        adaptiveThres_const=adaptiveThresConst,
                                                        sigma_Canny=sigmaCanny, low_threshold_canny=lowThresholdCanny,
                                                        high_threshold_canny=highThresholdCanny,
                                                        min_sep_dist=minSepDist,
                                                        lookup_radius=lookupRadius,
                                                        raft_center_threshold=raftCenterThreshold,
                                                        topLeft_x=topLeftX, topLeft_y=topLeftY, width_x=widthX,
                                                        height_y=heightY, error_message=' ')

        if regionalSearch == 1:
            centersInRegion, radiiInRegion, countInRegion = FindCirclesAdaptive(currentFrameGray, maxNumOfRaftsInRegion,
                                                                                radii_Hough=radiusIntervalHough,
                                                                                thres_value=thresholdingValue,
                                                                                adaptiveThres_blocksize=adaptiveThresBlocksize,
                                                                                adaptiveThres_const=adaptiveThresConst,
                                                                                sigma_Canny=sigmaCanny,
                                                                                low_threshold_canny=lowThresholdCanny,
                                                                                high_threshold_canny=highThresholdCanny,
                                                                                min_sep_dist=minSepDist,
                                                                                lookup_radius=lookupRadius,
                                                                                raft_center_threshold=raftCenterThreshold,
                                                                                topLeft_x=regionTopLeftX,
                                                                                topLeft_y=regionTopLeftY,
                                                                                width_x=regionWidth,
                                                                                height_y=regionHeight,
                                                                                error_message=' ')
            raftLocationsInRegion[:countInRegion, 0, :] = centersInRegion[:countInRegion, :]
            raftRadiiInRegion[:countInRegion, 0] = radiiInRegion[:countInRegion]

        # detect by countours
        #        centers, radii = DetectByContours(currentFrameGray)
        #        numOfContours, _ = centers.shape
        #        if numOfContours < numOfRafts:
        #            continue

        # sorting
        centersSorted, radiiSorted, distSorted, orbitingAnglesSorted, layerIndexSorted = NumberingRafts(centers, radii,
                                                                                                        numOfRafts)

        # transfer data of the first frame to key data set
        raftLocations[:, currentFrameNum, :] = centersSorted
        raftRadii[:, currentFrameNum] = radiiSorted
        raftOrbitingCenters[currentFrameNum, :] = np.mean(centers, axis=0)
        raftOrbitingDistances[:, currentFrameNum] = distSorted
        raftOrbitingAngles[:, currentFrameNum] = orbitingAnglesSorted
        raftOrbitingLayerIndices[:, currentFrameNum] = layerIndexSorted
        if processRotation == 1:
            for raftID in np.arange(numOfRafts):
                firstImages[raftID, :, :] = CropImage(currentFrameGray, raftLocations[raftID, currentFrameNum, :],
                                                      sizeOfCroppedRaftImage)
                raftOrientations[
                    raftID, currentFrameNum] = raftInitialOrientation  # this could change later when we have a external standard to define what degree is.

        # output images
        currentFrameDraw = currentFrameBGR.copy()
        currentFrameDraw = DrawRafts(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                     raftRadii[:, currentFrameNum], numOfRafts)
        currentFrameDraw = DrawRaftNumber(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
        if effusionData == 1:
            currentFrameDraw = DrawEffusedRaftCount(currentFrameDraw, raftEffused[currentFrameNum],
                                                    raftToLeft[currentFrameNum], raftToRight[currentFrameNum], topLeftX,
                                                    topLeftY, widthX, heightY)
        if processRotation == 1:
            currentFrameDraw = DrawRaftOrientations(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                                    raftOrientations[:, currentFrameNum], raftRadii[:, currentFrameNum],
                                                    numOfRafts)
        if outputImageSeq == 1:
            outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
                spinSpeed) + 'rps_' + str(currentFrameNum + 1).zfill(4) + '.jpg'
            cv.imwrite(outputImageName, currentFrameDraw)
        if outputVideo == 1:
            outputVideoName = outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
                spinSpeed) + 'rps_' + str(magnification) + 'x_' + commentsSub + '.mp4'
            fourcc = cv.VideoWriter_fourcc(*'DIVX')
            frameW, frameH, _ = currentFrameDraw.shape
            videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)
            videoOut.write(currentFrameDraw)

        # loop over all the frames
        for currentFrameNum in progressbar.progressbar(range(1, numOfFrames)):
            # Note that the first frame has been dealt with, so currentFrameNum = 0 is omitted
            if isVideo == 1:
                retval, currentFrameBGR = cap.read()
                currentFrameGray = cv.cvtColor(currentFrameBGR, cv.COLOR_BGR2GRAY)
            else:
                currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
                #                currentFrameGray = cv.imread(tiffFileList[currentFrameNum], 0)
                currentFrameGray = currentFrameBGR[:, :,
                                   1]  # use only green channel. We found green channel has the highest contrast.

            if diffusionData == 1:
                # top left cornor of the next search box: x-coordingate
                if raftLocations[0, currentFrameNum - 1, 0] - diffBoxRadius >= topLeftX:
                    diffBoxTopLeftX = raftLocations[0, currentFrameNum - 1, 0] - diffBoxRadius
                else:
                    diffBoxTopLeftX = topLeftX
                # top left cornor of the next search box: y-coordingate
                if raftLocations[0, currentFrameNum - 1, 1] - diffBoxRadius >= topLeftY:
                    diffBoxTopLeftY = raftLocations[0, currentFrameNum - 1, 1] - diffBoxRadius
                else:
                    diffBoxTopLeftY = topLeftY
                # box size of search box, width
                if raftLocations[0, currentFrameNum - 1, 0] + diffBoxRadius <= topLeftX + widthX:
                    diffBoxWidthX = diffBoxRadius + diffBoxRadius
                else:
                    diffBoxWidthX = topLeftX + widthX - raftLocations[0, currentFrameNum - 1, 0]
                # box size of search box, height
                if raftLocations[0, currentFrameNum - 1, 1] + diffBoxRadius <= topLeftY + heightY:
                    diffBoxHeightY = diffBoxRadius + diffBoxRadius
                else:
                    diffBoxHeightY = topLeftY + heightY - raftLocations[0, currentFrameNum - 1, 1]

                centers, radii, currCount = FindCirclesAdaptive(currentFrameGray, numOfRafts,
                                                                radii_Hough=radiusIntervalHough,
                                                                thres_value=thresholdingValue,
                                                                adaptiveThres_blocksize=adaptiveThresBlocksize,
                                                                adaptiveThres_const=adaptiveThresConst,
                                                                sigma_Canny=sigmaCanny,
                                                                low_threshold_canny=lowThresholdCanny,
                                                                high_threshold_canny=highThresholdCanny,
                                                                min_sep_dist=minSepDist,
                                                                lookup_radius=lookupRadius,
                                                                raft_center_threshold=raftCenterThreshold,
                                                                topLeft_x=diffBoxTopLeftX, topLeft_y=diffBoxTopLeftY,
                                                                width_x=diffBoxWidthX, height_y=diffBoxHeightY,
                                                                error_message=' ')

            else:
                # find circles by Hough transform
                centers, radii, currCount = FindCirclesAdaptive(currentFrameGray, numOfRafts,
                                                                radii_Hough=radiusIntervalHough,
                                                                thres_value=thresholdingValue,
                                                                adaptiveThres_blocksize=adaptiveThresBlocksize,
                                                                adaptiveThres_const=adaptiveThresConst,
                                                                sigma_Canny=sigmaCanny,
                                                                low_threshold_canny=lowThresholdCanny,
                                                                high_threshold_canny=highThresholdCanny,
                                                                min_sep_dist=minSepDist,
                                                                lookup_radius=lookupRadius,
                                                                raft_center_threshold=raftCenterThreshold,
                                                                topLeft_x=topLeftX, topLeft_y=topLeftY, width_x=widthX,
                                                                height_y=heightY, error_message=' ')

            if regionalSearch == 1:
                centersInRegion, radiiInRegion, countInRegion = FindCirclesAdaptive(currentFrameGray,
                                                                                    maxNumOfRaftsInRegion,
                                                                                    radii_Hough=radiusIntervalHough,
                                                                                    thres_value=thresholdingValue,
                                                                                    adaptiveThres_blocksize=adaptiveThresBlocksize,
                                                                                    adaptiveThres_const=adaptiveThresConst,
                                                                                    sigma_Canny=sigmaCanny,
                                                                                    low_threshold_canny=lowThresholdCanny,
                                                                                    high_threshold_canny=highThresholdCanny,
                                                                                    min_sep_dist=minSepDist,
                                                                                    lookup_radius=lookupRadius,
                                                                                    raft_center_threshold=raftCenterThreshold,
                                                                                    topLeft_x=regionTopLeftX,
                                                                                    topLeft_y=regionTopLeftY,
                                                                                    width_x=regionWidth,
                                                                                    height_y=regionHeight,
                                                                                    error_message=' ')
                raftLocationsInRegion[:countInRegion, currentFrameNum, :] = centersInRegion[:countInRegion, :]
                raftRadiiInRegion[:countInRegion, currentFrameNum] = radiiInRegion[:countInRegion]

            # find cirlces by detect contours
            #            centers, radii = DetectByContours(currentFrameGray)
            #            numOfContours, _ = centers.shape
            #            if numOfContours < numOfRafts:
            #                continue

            # tracking rafts according to the proximity to the previous frame, and then save to key data set, 
            if effusionData == 1:
                targetID = np.arange(numOfRafts)
                raftMovingToLeft, raftMovingToRight = CountingEffusedRafts(raftLocations[:, currentFrameNum - 1, :],
                                                                           prevCount, centers, currCount,
                                                                           effusionBoundaryX, maxDisplacement)
                raftMovingToLeftCount = raftMovingToLeftCount + raftMovingToLeft
                raftMovingToRightCount = raftMovingToRightCount + raftMovingToRight
                effusedRaftCount = effusedRaftCount + raftMovingToRight - raftMovingToLeft
                raftEffused[currentFrameNum] = effusedRaftCount
                raftToLeft[currentFrameNum] = raftMovingToLeftCount
                raftToRight[currentFrameNum] = raftMovingToRightCount
                prevCount = currCount
            else:
                targetID = TrackingRafts(raftLocations[:, currentFrameNum - 1, :], centers, numOfRafts)

                # find and track rafts together.
            #            centers, radii, detectedNum = FindAndSortCircles(currentFrameGray, numOfRafts, prev_pos = raftLocations[:,currentFrameNum-1,:],
            #                                                             radii_Hough = radiusIntervalHough, thres_value = thresholdingValue,
            #                                                             sigma_Canny = sigmaCanny, low_threshold_canny = lowThresholdCanny,
            #                                                             high_threshold_canny = highThresholdCanny, max_displ = maxDisplacement)

            # filling key dataset after using FindCircles or DetectByCountours
            raftOrbitingCenters[currentFrameNum, :] = np.mean(centers, axis=0)
            raftOrbitingLayerIndices[:, currentFrameNum] = 1
            for raftID in np.arange(numOfRafts):
                raftLocations[raftID, currentFrameNum, :] = centers[targetID[raftID], :]
                raftRadii[raftID, currentFrameNum] = radii[targetID[raftID]]
                raftOrbitingDistances[raftID, currentFrameNum] = CalculateDistance(
                    raftOrbitingCenters[currentFrameNum, :], raftLocations[raftID, currentFrameNum, :])
                raftOrbitingAngles[raftID, currentFrameNum] = CalculateOrbitingAngle(
                    raftOrbitingCenters[currentFrameNum, :], raftLocations[raftID, currentFrameNum, :])

            # filling key dataset after using FindAndSortCircles
            #            raftOrbitingCenters[currentFrameNum,:] = np.mean(centers, axis = 0)
            #            raftOrbitingLayerIndices[:, currentFrameNum] = 1
            #            for raftID in np.arange(numOfRafts):
            #                raftLocations[raftID,currentFrameNum,:] = centers[raftID,:]
            #                raftRadii[raftID,currentFrameNum] = radii[raftID]
            #                raftOrbitingDistances[raftID,currentFrameNum] = CalculateDistance(raftOrbitingCenters[currentFrameNum,:], raftLocations[raftID,currentFrameNum,:])
            #                raftOrbitingAngles[raftID,currentFrameNum] = CalculateOrbitingAngle(raftOrbitingCenters[currentFrameNum,:], raftLocations[raftID,currentFrameNum,:])

            # now deal with rotation
            if processRotation == 1:
                for raftID in np.arange(numOfRafts):
                    currImages[raftID, :, :] = CropImage(currentFrameGray, raftLocations[raftID, currentFrameNum, :],
                                                         sizeOfCroppedRaftImage)
                    rotationAngle = GetRotationAngle(firstImages[raftID, :, :], currImages[raftID, :, :])
                    raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, 0] + rotationAngle
                    while raftOrientations[raftID, currentFrameNum] < 0:
                        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] + 360
                    while raftOrientations[raftID, currentFrameNum] > 360:
                        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] - 360

            # output images
            currentFrameDraw = currentFrameBGR.copy()
            currentFrameDraw = DrawRafts(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                         raftRadii[:, currentFrameNum], numOfRafts)
            currentFrameDraw = DrawRaftNumber(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
            if effusionData == 1:
                currentFrameDraw = DrawEffusedRaftCount(currentFrameDraw, raftEffused[currentFrameNum],
                                                        raftToLeft[currentFrameNum], raftToRight[currentFrameNum],
                                                        topLeftX, topLeftY, widthX, heightY)
            if processRotation == 1:
                currentFrameDraw = DrawRaftOrientations(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                                        raftOrientations[:, currentFrameNum],
                                                        raftRadii[:, currentFrameNum], numOfRafts)
            if outputImageSeq == 1:
                outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
                    spinSpeed) + 'rps_' + str(currentFrameNum + 1).zfill(4) + '.jpg'
                cv.imwrite(outputImageName, currentFrameDraw)
            if outputVideo == 1:
                videoOut.write(currentFrameDraw)

        # save data file
        tempShelf = shelve.open(outputDataFileName, 'n')  # 'n' for new
        for key in listOfVarialbesToSave:
            try:
                tempShelf[key] = globals()[key]
            except TypeError:
                #
                # __builtins__, tempShelf, and imported modules can not be shelved.
                #
                # print('ERROR shelving: {0}'.format(key))
                pass
        tempShelf.close()

        if outputVideo == 1:
            videoOut.release()

        if isVideo == 1:
            cap.release()
        else:
            oldFilePath = os.getcwd()
            os.chdir('..')  # go the the main folder
            newFilePath = os.getcwd()
            os.rename(oldFilePath + '/' + outputDataFileName + '.dat', newFilePath + '/' + outputDataFileName + '.dat')
            os.rename(oldFilePath + '/' + outputDataFileName + '.bak', newFilePath + '/' + outputDataFileName + '.bak')
            os.rename(oldFilePath + '/' + outputDataFileName + '.dir', newFilePath + '/' + outputDataFileName + '.dir')
            if outputVideo == 1:
                os.rename(oldFilePath + '/' + outputVideoName, newFilePath + '/' + outputVideoName)

    # go one level up to the root folder
    os.chdir('..')

# %% debugging code for find circles using scikit image

## Reading files
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

## look at the FFT

f = np.fft.fft2(currentFrameGray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))
magnitude_spectrumNormalized = magnitude_spectrum / magnitude_spectrum.max()
magnitude_spectrumEnhanced = np.uint8(magnitude_spectrumNormalized * 255)
# magnitude_spectrumEnhanced =  cv.equalizeHist(magnitude_spectrum)
plt.imshow(magnitude_spectrumEnhanced, 'gray')
outputImageName = tiffFileList[currentFrameNum].partition('.')[0] + '_fft.jpg'
cv.imwrite(outputImageName, magnitude_spectrumEnhanced)

## Reading video frames
videoFileList = glob.glob('*.MOV')
numOfExp = len(videoFileList)
expID = 0
cap = cv.VideoCapture(videoFileList[expID])
numOfFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
retval, currentFrameBGR = cap.read()
currentFrameGray = cv.cvtColor(currentFrameBGR, cv.COLOR_BGR2GRAY)
cap.release()

## below is to expanded version of the function FindCircles

## parameters for various find-circle functions
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
# not used FindCirclesAdaptive, but in the FindCirclesThres and FindAndSortCircles
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
# image_edges = canny(image_thres, sigma = sigma_Canny, low_threshold = low_threshold_canny, high_threshold = high_threshold_canny)
# plt.imshow(image_edges,'gray')

# hough transform to find circles
hough_results = hough_circle(image_thres, np.arange(*radii_Hough))
accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_Hough))

# assuming that the first raft (highest accumulator score) is a good one
# raft_centers[0,0] = cx[0]
# raft_centers[0,1] = cy[0]
# raft_radii[0] = radii[0]
# raft_centerPixelValue[0] = currentFrameGrayCropped[cy[0], cx[0]]
# raft_InscribedSquareMeanValue[0] = currentFrameGrayCropped[cy[0]-radii[0]//2 : cy[0]+radii[0]//2 , cx[0]-radii[0]//2:cx[0]+radii[0]//2].mean()
# raft_accumScore[0] = accums[0]
# raft_thresholdValues[0] = thres_value
raft_count = 0  # starting from 1!

# remove circles that belong to the same raft and circles that happened to be in between rafts and rafts outside lookup radius
t1 = time.perf_counter()
for accumScore, detected_cx, detected_cy, detected_radius in zip(accums, cx, cy, radii):
    new_raft = 1
    if image_cropped[detected_cy, detected_cx] < raft_center_threshold:
        new_raft = 0
    elif image_cropped[detected_cy - detected_radius // 2: detected_cy + detected_radius // 2,
         detected_cx - detected_radius // 2:detected_cx + detected_radius // 2].mean() < raft_center_threshold:
        new_raft = 0
    #    elif  (detected_cx - width_x/2)**2 +  (detected_cy - height_y/2)**2 > lookup_radius**2:
    #        new_raft = 0
    else:
        costMatrix = scipy_distance.cdist(np.array([detected_cx, detected_cy], ndmin=2), raft_centers[:raft_count, :],
                                          'euclidean')
        if np.any(costMatrix < min_sep_dist):  # raft still exist
            new_raft = 0
    if new_raft == 1:
        raft_centers[
            raft_count, 0] = detected_cx  # note that raft_count starts with 1, also note that cx corresonds to columns number
        raft_centers[raft_count, 1] = detected_cy  # cy is row number
        raft_radii[raft_count] = detected_radius
        #            raft_centerPixelValue[raft_count] = currentFrameGrayCropped[detected_cy, detected_cx]
        #            raft_InscribedSquareMeanValue[raft_count] = currentFrameGrayCropped[detected_cy-detected_radius//2 : detected_cy+detected_radius//2 , detected_cx-detected_radius//2:detected_cx+detected_radius//2].mean()
        #            raft_accumScore[raft_count] = accumScore
        #            raft_thresholdValues[raft_count] = thres_value
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

## FindCircles function
raft_centers, raft_radii, raft_count = FindCirclesThres(currentFrameGray, num_of_rafts, thres_value=thres_value,
                                                        adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                        adaptiveThres_const=adaptiveThres_const,
                                                        radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                        low_threshold_canny=low_threshold_canny,
                                                        high_threshold_canny=high_threshold_canny,
                                                        min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                        raft_center_threshold=raft_center_threshold,
                                                        topLeft_x=topLeft_x, topLeft_y=topLeft_y, width_x=width_x,
                                                        height_y=height_y, error_message=' ')
print(raft_count)

## FindCircles function with adaptive thresholding
raft_centers, raft_radii, raft_count = FindCirclesAdaptive(currentFrameGray, num_of_rafts, thres_value=thres_value,
                                                           adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                           adaptiveThres_const=adaptiveThres_const,
                                                           radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                           low_threshold_canny=low_threshold_canny,
                                                           high_threshold_canny=high_threshold_canny,
                                                           min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                           raft_center_threshold=raft_center_threshold,
                                                           topLeft_x=topLeft_x, topLeft_y=topLeft_y, width_x=width_x,
                                                           height_y=height_y, error_message=' ')
print(raft_count)
## below is for FindAndSortCircles, after using FindCircles to get the first centers
prev_pos = raft_centers
max_displ = 30
currentFrameNum += 1
currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
currentFrameGray = cv.imread(tiffFileList[currentFrameNum], 0)
currentFrameGrayContAdj = cv.equalizeHist(currentFrameGray)
plt.imshow(currentFrameGray, 'gray')

raft_centers, raft_radii, raft_count = FindAndSortCircles(currentFrameGray, num_of_rafts, prev_pos=prev_pos,
                                                          thres_value=thres_value, radii_Hough=radii_Hough,
                                                          sigma_Canny=sigma_Canny,
                                                          low_threshold_canny=low_threshold_canny,
                                                          high_threshold_canny=high_threshold_canny,
                                                          max_displ=max_displ)

## below is for DetectByContours
raft_centers, raft_radii = DetectByContours(currentFrameGray)

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

# raft_centers, raft_radii = DetectByContours(currentFrameGray)


# draw circles. 
curr_frame_draw = currentFrameBGR.copy()
curr_frame_draw = DrawRafts(curr_frame_draw, raft_centers, raft_radii, num_of_rafts)
# curr_frame_draw = cv.circle(curr_frame_draw, (int(currentFrameGray.shape[1]/2), int(currentFrameGray.shape[0]/2)), lookup_radius, (255,0,0), int(2))
curr_frame_draw = DrawRaftNumber(curr_frame_draw, raft_centers, num_of_rafts)
# cv.imshow('analyzed image with circles', curr_frame_draw)
plt.imshow(curr_frame_draw[:, :, ::-1])

curr_frame_draw = currentFrameBGR.copy()
curr_frame_draw = DrawRafts(curr_frame_draw, (currentFrameGray.shape[1] / 2, currentFrameGray.shape[0] / 2),
                            lookup_radius, 1)
cv.imshow('analyzed image', curr_frame_draw)

# %% debugging code for getting the rotation angle

# read two frames: 
## Reading files
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

raft_centers_1, raft_radii_1, raft_count_1 = FindCirclesAdaptive(frame1Gray, num_of_rafts, thres_value=thres_value,
                                                                 adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                                 adaptiveThres_const=adaptiveThres_const,
                                                                 radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                                 low_threshold_canny=low_threshold_canny,
                                                                 high_threshold_canny=high_threshold_canny,
                                                                 min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                                 raft_center_threshold=raft_center_threshold,
                                                                 topLeft_x=topLeft_x, topLeft_y=topLeft_y,
                                                                 width_x=width_x, height_y=height_y, error_message=' ')
print(raft_count_1)

raft_centers_2, raft_radii_2, raft_count_2 = FindCirclesAdaptive(frame2Gray, num_of_rafts, thres_value=thres_value,
                                                                 adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                                 adaptiveThres_const=adaptiveThres_const,
                                                                 radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                                 low_threshold_canny=low_threshold_canny,
                                                                 high_threshold_canny=high_threshold_canny,
                                                                 min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                                 raft_center_threshold=raft_center_threshold,
                                                                 topLeft_x=topLeft_x, topLeft_y=topLeft_y,
                                                                 width_x=width_x, height_y=height_y, error_message=' ')
print(raft_count_2)

# setting parameters and initialize dataset
numOfRafts = 2
sizeOfCroppedRaftImage = 150
raftLocations = np.zeros((numOfRafts, 2, 2),
                         dtype=int)  # (raftNum, frameNum, x(columns)&y(rows), note that only two frames
raftOrientations = np.zeros((numOfRafts, 2))  # (raftNum, frameNum)
raftRadii = np.zeros((numOfRafts, 2), dtype=int)  # (raftNUm, frameNum)
firstImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))
currImages = np.zeros((numOfRafts, sizeOfCroppedRaftImage, sizeOfCroppedRaftImage))
raftInitialOrientation = 0

# tracking part
raftLocations[:, 0, :] = raft_centers_1
raftRadii[:, 0] = raft_radii_1
targetID = TrackingRafts(raft_centers_1, raft_centers_2, numOfRafts)
for raftID in np.arange(numOfRafts):
    raftLocations[raftID, 1, :] = raft_centers_2[targetID[raftID], :]
    raftRadii[raftID, 1] = raft_radii_2[targetID[raftID]]

# setting the orientation in the first image as initial orientation 0
currentFrameNum = 0
for raftID in np.arange(numOfRafts):
    firstImages[raftID, :, :] = CropImage(frame1Gray, raftLocations[raftID, currentFrameNum, :], sizeOfCroppedRaftImage)
    raftOrientations[raftID, currentFrameNum] = raftInitialOrientation
frame1Draw = frame1BGR.copy()
frame1Draw = DrawRaftOrientations(frame1Draw, raftLocations[:, currentFrameNum, :],
                                  raftOrientations[:, currentFrameNum], raftRadii[:, currentFrameNum], numOfRafts)
plt.figure()
plt.imshow(frame1Draw[:, :, ::-1])

# obtain the orientation of rafts in the second image
currentFrameNum = 1
for raftID in np.arange(numOfRafts):
    currImages[raftID, :, :] = CropImage(frame2Gray, raftLocations[raftID, currentFrameNum, :], sizeOfCroppedRaftImage)
    rotationAngle = GetRotationAngle(firstImages[raftID, :, :], currImages[raftID, :, :])
    raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, 0] + rotationAngle
    while raftOrientations[raftID, currentFrameNum] < 0:
        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] + 360
    while raftOrientations[raftID, currentFrameNum] > 360:
        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] - 360
frame2Draw = frame2BGR.copy()
frame2Draw = DrawRaftOrientations(frame2Draw, raftLocations[:, currentFrameNum, :],
                                  raftOrientations[:, currentFrameNum], raftRadii[:, currentFrameNum], numOfRafts)
plt.figure()
plt.imshow(frame2Draw[:, :, ::-1])


def GetRotationAngle(prev_image, curr_image):
    ''' extract the angle of rotation theta between two frames
    '''

    max_value = np.amax(prev_image)

    if prev_image.dtype == 'float' and max_value <= 1:
        img1 = np.uint8(prev_image * 255)
        img2 = np.uint8(curr_image * 255)

    if prev_image.dtype == 'float' and max_value > 1:
        img1 = np.uint8(prev_image)
        img2 = np.uint8(curr_image)

    img1 = cv.equalizeHist(img1)
    img2 = cv.equalizeHist(img2)

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=20)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

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
    # negative sign is to make the sign of the angle the same as in rhino, i.e. counter-clock wise from x-axis is positive

    return theta


raftID = 0

prev_image = firstImages[raftID, :, :]
plt.figure()
plt.imshow(prev_image, 'gray')

curr_image = currImages[raftID, :, :]
plt.figure()
plt.imshow(curr_image, 'gray')

rotationAngle = GetRotationAngle(prev_image, curr_image)

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

# %% using substraction to evulate the number of rafts that are stepped out.

# read two frames: 
## Reading files
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

raft_centers_1, raft_radii_1, raft_count_1 = FindCirclesAdaptive(frame1Gray, num_of_rafts, thres_value=thres_value,
                                                                 adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                                 adaptiveThres_const=adaptiveThres_const,
                                                                 radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                                 low_threshold_canny=low_threshold_canny,
                                                                 high_threshold_canny=high_threshold_canny,
                                                                 min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                                 raft_center_threshold=raft_center_threshold,
                                                                 topLeft_x=topLeft_x, topLeft_y=topLeft_y,
                                                                 width_x=width_x, height_y=height_y, error_message=' ')
print(raft_count_1)

raft_centers_2, raft_radii_2, raft_count_2 = FindCirclesAdaptive(frame2Gray, num_of_rafts, thres_value=thres_value,
                                                                 adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                                 adaptiveThres_const=adaptiveThres_const,
                                                                 radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                                 low_threshold_canny=low_threshold_canny,
                                                                 high_threshold_canny=high_threshold_canny,
                                                                 min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                                 raft_center_threshold=raft_center_threshold,
                                                                 topLeft_x=topLeft_x, topLeft_y=topLeft_y,
                                                                 width_x=width_x, height_y=height_y, error_message=' ')
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
targetID = TrackingRafts(raft_centers_1, raft_centers_2, numOfRafts)
for raftID in np.arange(numOfRafts):
    raftLocations[raftID, 1, :] = raft_centers_2[targetID[raftID], :]

# crop and store firstImages
currentFrameNum = 0
for raftID in np.arange(numOfRafts):
    firstImages[raftID, :, :] = CropImage(frame1Gray, raftLocations[raftID, currentFrameNum, :], sizeOfCroppedRaftImage)
frame1Draw = frame1BGR.copy()

# crop and store currImages
currentFrameNum = 1
for raftID in np.arange(numOfRafts):
    currImages[raftID, :, :] = CropImage(frame2Gray, raftLocations[raftID, currentFrameNum, :], sizeOfCroppedRaftImage)

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
        ax[1].imshow(
            np.roll(currImages[raftID, :, :], (raftMatchingShifts[raftID, 1, 0], raftMatchingShifts[raftID, 1, 1]),
                    (0, 1)), 'gray')
    elif numOfRaftsSteppedOut > 1:
        ax[0, counter].imshow(firstImages[raftID, :, :], 'gray')
        ax[0, counter].set_title('raftID = {}'.format(raftID))
        ax[1, counter].imshow(
            np.roll(currImages[raftID, :, :], (raftMatchingShifts[raftID, 1, 0], raftMatchingShifts[raftID, 1, 1]),
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
curr_frame_draw = DrawRafts(curr_frame_draw, raft_centers_1, raft_radii_1, num_of_rafts)
# curr_frame_draw = cv.circle(curr_frame_draw, (int(currentFrameGray.shape[1]/2), int(currentFrameGray.shape[0]/2)), lookup_radius, (255,0,0), int(2))
curr_frame_draw = DrawRaftNumber(curr_frame_draw, raft_centers_1, num_of_rafts)
# cv.imshow('analyzed image with circles', curr_frame_draw)
plt.imshow(curr_frame_draw[:, :, ::-1])

# %% track circle using Hungarian Algorithm; effused rafts function definition
## Reading files
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

## FindCircles in the previous frame
prev_centers, prev_radii, prev_count = FindCirclesThres(prevFrameGray, num_of_rafts, thres_value=thres_value,
                                                        adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                        adaptiveThres_const=adaptiveThres_const,
                                                        radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                        low_threshold_canny=low_threshold_canny,
                                                        high_threshold_canny=high_threshold_canny,
                                                        min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                        raft_center_threshold=raft_center_threshold,
                                                        topLeft_x=topLeft_x, topLeft_y=topLeft_y, width_x=width_x,
                                                        height_y=height_y, error_message=' ')
## FindCircles in the current frame
curr_centers, curr_radii, curr_count = FindCirclesAdaptive(currFrameGray, num_of_rafts, thres_value=thres_value,
                                                           adaptiveThres_blocksize=adaptiveThres_blocksize,
                                                           adaptiveThres_const=adaptiveThres_const,
                                                           radii_Hough=radii_Hough, sigma_Canny=sigma_Canny,
                                                           low_threshold_canny=low_threshold_canny,
                                                           high_threshold_canny=high_threshold_canny,
                                                           min_sep_dist=min_sep_dist, lookup_radius=lookup_radius,
                                                           raft_center_threshold=raft_center_threshold,
                                                           topLeft_x=topLeft_x, topLeft_y=topLeft_y, width_x=width_x,
                                                           height_y=height_y, error_message=' ')

boudnary_x = topLeft_x + width_x // 2


def CountingEffusedRafts(prev_centers, prev_count, curr_centers, curr_count, boundary_x, max_dispacement):
    '''
    test if the raft crosses the boundary of container
    '''
    effused_raft_change = 0
    costMatrix = scipy_distance.cdist(prev_centers[:prev_count], curr_centers[:curr_count], 'euclidean')
    #  note that row index refers to previous raft number, column index refers to current raft number

    # select the boundary crossing to be in the middle of the cropped image, so only deals with existing rafts
    for raftID in np.arange(prev_count):
        if np.any(costMatrix[raftID, :] < max_displacement):  # raft still exist
            curr_raftID = np.nonzero(costMatrix[raftID, :] < max_displacement)[0][
                0]  # [0][0] is to convert array into scalar
            if (prev_centers[raftID, 0] > boundary_x) and (curr_centers[curr_raftID, 0] <= boundary_x):
                effused_raft_change = effused_raft_change + 1
            elif (prev_centers[raftID, 0] < boundary_x) and (curr_centers[curr_raftID, 0] >= boundary_x):
                effused_raft_change = effused_raft_change - 1
    return effused_raft_change


# targetID = TrackingRafts(prev_centers, curr_centers, num_of_rafts)

costMatrix = scipy_distance.cdist(prev_centers[:prev_count], curr_centers[:curr_count], 'euclidean')
row_ind, col_ind = linear_sum_assignment(costMatrix)

targetID = col_ind

## Hungarian algorithm to track
# def TrackingRafts(prev_rafts_locations, detected_centers):
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
## old tracking algorithm based on minimal distance. 
# def TrackingRafts(prev_rafts_locations, detected_centers, num_of_rafts):
#    ''' sort the detected_centers according to the locations of rafts in the previous frame
#    '''
#    target_id = np.zeros((num_of_rafts,1), dtype = int)
#    
#    for prev_raft_id in np.arange(num_of_rafts): # looping through the rafs location in the prev_rafts_locations
#        min_seperation = 10000
#        for test_id in np.arange(num_of_rafts): # looping through the rafs location in the detected_centers
#            seperation = CalculateDistance(prev_rafts_locations[prev_raft_id,:], detected_centers[test_id,:])
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
prev_frame_draw = DrawRafts(prev_frame_draw, prev_centers, prev_radii, num_of_rafts)
# prev_frame_draw = cv.circle(prev_frame_draw, (int(prevFrameGray.shape[1]/2), int(prevFrameGray.shape[0]/2)), lookup_radius, (255,0,0), int(2))
prev_frame_draw = DrawRaftNumber(prev_frame_draw, prev_centers, num_of_rafts)
# cv.imshow('analyzed image with first two circles', prev_frame_draw)
plt.imshow(prev_frame_draw[:, :, ::-1])

curr_frame_draw = currFrameBGR.copy()
curr_frame_draw = DrawRafts(curr_frame_draw, curr_centers_tracked, curr_radii_tracked, num_of_rafts)
# curr_frame_draw = cv.circle(curr_frame_draw, (int(currFrameGray.shape[1]/2), int(currFrameGray.shape[0]/2)), lookup_radius, (255,0,0), int(2))
curr_frame_draw = DrawRaftNumber(curr_frame_draw, curr_centers_tracked, num_of_rafts)
# cv.imshow('analyzed image with first two circles', curr_frame_draw)
plt.imshow(curr_frame_draw[:, :, ::-1])
