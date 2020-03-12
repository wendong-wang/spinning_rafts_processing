import numpy as np

import cv2 as cv
import time

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as scipy_distance




def find_circles_thres(current_frame_gray, num_of_rafts, radii_Hough=[17, 19],
                       thres_value=70, sigma_Canny=1.0, low_threshold_canny=25, high_threshold_canny=127,
                       min_sep_dist=20, raft_center_threshold=60,
                       topLeft_x=390, topLeft_y=450, width_x=850, height_y=850):
    """
    find the centers of each raft
    :param current_frame_gray: image in grayscale
    :param num_of_rafts:
    :param radii_Hough: the range of Hough radii
    :param thres_value: threshold value
    :param sigma_Canny:
    :param low_threshold_canny:
    :param high_threshold_canny:
    :param min_sep_dist:
    :param raft_center_threshold:
    :param topLeft_x:
    :param topLeft_y:
    :param width_x:
    :param height_y:
    :return: raft_centers, raft_radii, raft_count
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
            raft_centers[raft_count, 0] = detected_cx
            # note that raft_count starts with 1, also note that cx corresponds to columns number
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


def find_circles_adaptive(current_frame_gray, num_of_rafts, radii_hough=[17, 19],
                          adaptive_thres_blocksize=9, adaptive_thres_const=-20,
                          min_sep_dist=20, raft_center_threshold=60,
                          topLeft_x=390, topLeft_y=450, width_x=850, height_y=850):
    """
    find the centers of each raft
    :param current_frame_gray:
    :param num_of_rafts:
    :param radii_hough:
    :param adaptive_thres_blocksize:
    :param adaptive_thres_const:
    :param min_sep_dist:
    :param raft_center_threshold:
    :param topLeft_x:
    :param topLeft_y:
    :param width_x:
    :param height_y:
    :return: raft_centers, raft_radii, raft_count

    """
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # crop the image
    image_cropped = current_frame_gray[topLeft_y: topLeft_y + height_y, topLeft_x: topLeft_x + width_x]

    # threshold the image
    image_thres = cv.adaptiveThreshold(image_cropped, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                       adaptive_thres_blocksize, adaptive_thres_const)

    # use Hough transform to find circles
    hough_results = hough_circle(image_thres, np.arange(*radii_hough))
    accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_hough))

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
            raft_centers[raft_count, 0] = detected_cx
            # note that raft_count starts with 1, also note that cx corresonds to columns number
            raft_centers[raft_count, 1] = detected_cy
            # cy is row number
            raft_radii[raft_count] = detected_radius
            raft_count = raft_count + 1
        if raft_count == num_of_rafts:
            #            error_message = 'all rafts found'
            break

    # convert the xy coordinates of the cropped image into the coordinates of the original image
    raft_centers[:, 0] = raft_centers[:, 0] + topLeft_x
    raft_centers[:, 1] = raft_centers[:, 1] + topLeft_y

    return raft_centers, raft_radii, raft_count


def find_and_sort_circles(image_gray, num_of_rafts, prev_pos, radii_Hough=[30, 40], thres_value=30, sigma_Canny=1.0,
                          low_threshold_canny=25, high_threshold_canny=127, max_displ=50):
    """
    For each raft detected in the prev_pos, go through the newly found circles in descending order of scores,
    and the first one within max_displ is the stored as the new position of the raft.

    :param image_gray: gray scale image
    :param num_of_rafts: number of rafts to be located
    :param prev_pos: previous positions of rafts
    :param radii_Hough: [starting radius, ending radius], to be unpacked as an argument for hough_circle
    :param thres_value:
    :param sigma_Canny: the width of the Gaussian filter for Canny edge detection
    :param low_threshold_canny: low threshold for Canny
    :param high_threshold_canny: high threshold for Canny
    :param max_displ: maximum displacement
    :return:

    """
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # threshold the image first
    retval, image_thres = cv.threshold(image_gray, thres_value, 255, 0)
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
                raft_centers[raftID, 0] = detected_cx
                # note that raft_count starts with 1, also note that cx corresonds to columns number
                raft_centers[raftID, 1] = detected_cy
                # cy is row number
                raft_radii[raftID] = detected_radius
                raft_count += 1
                break

    return raft_centers, raft_radii, raft_count


def detect_by_contours(image_gray):
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


def parse_main_folder_name(main_folder_name):
    """
    parse the name of the main folder here, and return the follwing parts
    date, string
    raft_geometry, string
    thin_film_prop, string
    magnet_field_prop, string
    comments, string
    """
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


def parse_subfolder_name(subfolder_name):
    """
    parse the subfolder name here, and return the following variables
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


def calculate_distance(p1, p2):
    """
    calculate the distance between p1 and p2
    """

    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    return dist


def calculate_orbiting_angle(orbiting_center, raft):
    """
    calculate the orbiting angle of a raft with respect to a center
    """

    # note the negative sign before the first component, which is y component
    # the y in scikit-image is flipped.
    # it is to make the value of the angle appears natural, as in Rhino, with x-axis pointing right, and y-axis pointing up.
    angle = np.arctan2(-(raft[1] - orbiting_center[1]), (raft[0] - orbiting_center[0])) * 180 / np.pi

    return angle


def numbering_rafts(rafts_loc, rafts_radii, num_of_rafts):
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


def crop_image(grayscale_image, raft_center, width):
    """
    crop the area of the raft
    """
    top_row = int(raft_center[1] - width / 2)
    # note that y corresponds to rows, and is directed from top to bottom in scikit-image
    bottom_row = int(raft_center[1] + width / 2)

    left_column = int(raft_center[0] - width / 2)
    right_column = int(raft_center[0] + width / 2)

    raft_image = grayscale_image[top_row:bottom_row, left_column:right_column]
    return raft_image


def tracking_rafts(prev_rafts_locations, detected_centers):
    """
    sort the detected_centers according to the locations of rafts in the previous frame

    the row number of col_ind is raft number in prev_rafts_locations,
    the value in col_ind is the corresponding raft number in the detected_centers
    """
    costMatrix = scipy_distance.cdist(prev_rafts_locations, detected_centers, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(costMatrix)

    return col_ind


def counting_effused_rafts(prev_centers, prev_count, curr_centers, curr_count, boundary_x, max_displacement):
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


def get_rotation_angle(prev_image, curr_image):
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

    vector_along_x_axis_from_center = \
        np.float32([[sizeOfCroppedRaftImage / 2, sizeOfCroppedRaftImage / 2],
                    [sizeOfCroppedRaftImage, sizeOfCroppedRaftImage / 2]]).reshape(-1, 1, 2)
    vector_transformed = cv.perspectiveTransform(vector_along_x_axis_from_center, transformMatrix)

    theta = - np.arctan2(vector_transformed[1, 0, 1] - vector_transformed[0, 0, 1],
                         vector_transformed[1, 0, 0] - vector_transformed[0, 0, 0]) * 180 / np.pi
    # negative sign is to make the sign of the angle the same as in rhino, i.e. counter-clock wise from x-axis is positive

    return theta


def draw_rafts(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
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


def draw_raft_orientations(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicte the orientation of each raft
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    int(rafts_loc[raft_id, 1] - np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]))
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def draw_raft_number(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    """

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 255)  # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1),
                                (rafts_loc[raft_id, 0] - textSize[0] // 2, rafts_loc[raft_id, 1] + textSize[1] // 2),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_effused_raft_count(img_bgr, raft_effused, raft_to_left, raft_to_right, topLeft_X, topLeft_Y, width_X,
                            height_Y):
    """
    draw effused rafts
    """
    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)  # BGR
    font_thickness = 2
    line_color = (0, 0, 255)  # BGR
    line_thickness = 1
    output_img = img_bgr
    output_img = cv.line(output_img, (topLeft_X + width_X // 2, topLeft_Y),
                         (topLeft_X + width_X // 2, topLeft_Y + height_Y), line_color, line_thickness)
    output_img = cv.putText(output_img, 'Effused: ' + str(raft_effused), (topLeftX, topLeftY - 30), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'To left: ' + str(raft_to_left), (topLeftX, topLeftY - 60), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'To right: ' + str(raft_to_right), (topLeftX, topLeftY - 90), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img