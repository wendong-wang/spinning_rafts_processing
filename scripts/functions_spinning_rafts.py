import numpy as np
import cv2 as cv
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as scipy_distance
from scipy.spatial import Voronoi as ScipyVoronoi
from sklearn.metrics import mutual_info_score
# for singular spectrum analysis
import scipy.linalg as linalg


def find_circles_thres(current_frame_gray, num_of_rafts, radii_hough=[17, 19],
                       thres_value=70, sigma_canny=1.0, low_threshold_canny=25, high_threshold_canny=127,
                       min_sep_dist=20, raft_center_threshold=60,
                       top_left_x=390, top_left_y=450, width_x=850, height_y=850):
    """
    find the centers of each raft
    :param current_frame_gray: image in grayscale
    :param num_of_rafts:
    :param radii_hough: the range of Hough radii
    :param thres_value: threshold value
    :param sigma_canny:
    :param low_threshold_canny:
    :param high_threshold_canny:
    :param min_sep_dist:
    :param raft_center_threshold:
    :param top_left_x:
    :param top_left_y:
    :param width_x:
    :param height_y:
    :return: raft_centers, raft_radii, raft_count
    """
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # crop the image
    image_cropped = current_frame_gray[top_left_y: top_left_y + height_y, top_left_x: top_left_x + width_x]

    # threshold the image
    retval, image_thres = cv.threshold(image_cropped, thres_value, 255, 0)

    # find edges
    image_edges = canny(image_thres, sigma=sigma_canny, low_threshold=low_threshold_canny,
                        high_threshold=high_threshold_canny)

    # use Hough transform to find circles
    hough_results = hough_circle(image_edges, np.arange(*radii_hough))
    accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_hough))

    # assuming that the first raft (highest accumulator score) is a good one
    #    raft_centers[0,0] = cx[0]
    #    raft_centers[0,1] = cy[0]
    #    raft_radii[0] = radii[0]
    raft_count = 0  # starting from 1!

    # remove circles that belong to the same raft and circles that happened to be in between rafts
    for accum_score, detected_cx, detected_cy, detected_radius in zip(accums, cx, cy, radii):
        new_raft = 1
        if image_cropped[detected_cy, detected_cx] < raft_center_threshold:
            new_raft = 0
        elif image_cropped[detected_cy - detected_radius // 2: detected_cy + detected_radius // 2,
             detected_cx - detected_radius // 2:detected_cx + detected_radius // 2].mean() \
                < raft_center_threshold:
            new_raft = 0
        #        elif  (detected_cx - width_x/2)**2 +  (detected_cy - height_y/2)**2 > lookup_radius**2:
        #            new_raft = 0
        else:
            cost_matrix = scipy_distance.cdist(np.array([detected_cx, detected_cy], ndmin=2),
                                               raft_centers[:raft_count, :], 'euclidean')
            if np.any(cost_matrix < min_sep_dist):  # raft still exist
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
    raft_centers[:, 0] = raft_centers[:, 0] + top_left_x
    raft_centers[:, 1] = raft_centers[:, 1] + top_left_y

    return raft_centers, raft_radii, raft_count


def find_circles_adaptive(current_frame_gray, num_of_rafts, radii_hough,
                          adaptive_thres_blocksize=9, adaptive_thres_const=-20,
                          min_sep_dist=20, raft_center_threshold=60,
                          top_left_x=390, top_left_y=450, width_x=850, height_y=850):
    """
    find the centers of each raft
    :param current_frame_gray:
    :param num_of_rafts:
    :param radii_hough:
    :param adaptive_thres_blocksize:
    :param adaptive_thres_const:
    :param min_sep_dist:
    :param raft_center_threshold:
    :param top_left_x:
    :param top_left_y:
    :param width_x:
    :param height_y:
    :return: raft_centers, raft_radii, raft_count

    """
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # crop the image
    image_cropped = current_frame_gray[top_left_y: top_left_y + height_y, top_left_x: top_left_x + width_x]

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
                           detected_cx - detected_radius // 2:detected_cx + detected_radius // 2].mean() \
                < raft_center_threshold:
            new_raft = 0
        #        elif  (detected_cx - width_x/2)**2 +  (detected_cy - height_y/2)**2 > lookup_radius**2:
        #            new_raft = 0
        else:
            cost_matrix = scipy_distance.cdist(np.array([detected_cx, detected_cy], ndmin=2),
                                               raft_centers[:raft_count, :], 'euclidean')
            if np.any(cost_matrix < min_sep_dist):  # raft still exist
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
    raft_centers[:, 0] = raft_centers[:, 0] + top_left_x
    raft_centers[:, 1] = raft_centers[:, 1] + top_left_y

    return raft_centers, raft_radii, raft_count


def find_and_sort_circles(image_gray, num_of_rafts, prev_pos, radii_hough, thres_value=30, sigma_Canny=1.0,
                          low_threshold_canny=25, high_threshold_canny=127, max_displ=50):
    """
    For each raft detected in the prev_pos, go through the newly found circles in descending order of scores,
    and the first one within max_displ is the stored as the new position of the raft.

    :param image_gray: gray scale image
    :param num_of_rafts: number of rafts to be located
    :param prev_pos: previous positions of rafts
    :param radii_hough: [starting radius, ending radius], to be unpacked as an argument for hough_circle
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
    hough_results = hough_circle(image_edges, np.arange(*radii_hough))
    accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_hough))

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

    # note the negative sign before the first component, the y component
    # it is to make the orbiting angle in a right-handed coordiante.
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

    # calculate orbiting angle, note the two negative signs in front of both y- and x- components.
    # For y-component, it is for flipping image axis.
    # For x-component, it is make the counting start at x-axis and go clockwise.
    # Note the value of arctan2 is  [-pi, pi]
    orbiting_angles = np.arctan2(-(rafts_loc_sorted[:, 1] - orbiting_center[1]),
                                 -(rafts_loc_sorted[:, 0] - orbiting_center[0])) * 180 / np.pi

    # concatenate and sort
    rafts_loc_radii_dist_angle_layer = \
        np.column_stack((rafts_loc_sorted[:, 0], rafts_loc_sorted[:, 1],
                         rafts_radii_sorted, dist_sorted, orbiting_angles, layer_index))

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
    cost_matrix = scipy_distance.cdist(prev_rafts_locations, detected_centers, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return col_ind


def counting_effused_rafts(prev_centers, prev_count, curr_centers, curr_count, boundary_x, max_displacement):
    """
    test if the raft crosses the boundary of container
    """
    effused_raft_to_left = 0
    effused_raft_to_right = 0
    cost_matrix = scipy_distance.cdist(prev_centers[:prev_count], curr_centers[:curr_count], 'euclidean')
    #  note that row index refers to previous raft number, column index refers to current raft number

    # select the boundary crossing to be in the middle of the cropped image, so only deals with existing rafts
    for raftID in np.arange(prev_count):
        if np.any(cost_matrix[raftID, :] < max_displacement):  # raft still exist
            curr_raft_id = np.nonzero(cost_matrix[raftID, :] < max_displacement)[0][
                0]  # [0][0] is to convert array into scalar
            if (prev_centers[raftID, 0] >= boundary_x) and (curr_centers[curr_raft_id, 0] < boundary_x):
                effused_raft_to_left = effused_raft_to_left + 1
            elif (prev_centers[raftID, 0] < boundary_x) and (curr_centers[curr_raft_id, 0] >= boundary_x):
                effused_raft_to_right = effused_raft_to_right + 1
    return effused_raft_to_left, effused_raft_to_right


def get_rotation_angle(prev_image, curr_image, size_of_cropped_image):
    """
    extract the angle of rotation theta between two frames
    :param curr_image:
    :param prev_image:
    :param size_of_cropped_image:
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
    orb = cv.ORB_create(nfeatures=200)

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
    transform_matrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    if transform_matrix is None:
        transform_matrix, mask = cv.findHomography(src_pts, dst_pts, 0)

    if transform_matrix is None:
        transform_matrix, mask = cv.findHomography(src_pts, dst_pts, 0)

    vector_along_x_axis_from_center = \
        np.float32([[size_of_cropped_image / 2, size_of_cropped_image / 2],
                    [size_of_cropped_image, size_of_cropped_image / 2]]).reshape(-1, 1, 2)
    vector_transformed = cv.perspectiveTransform(vector_along_x_axis_from_center, transform_matrix)

    theta = - np.arctan2(vector_transformed[1, 0, 1] - vector_transformed[0, 0, 1],
                         vector_transformed[1, 0, 0] - vector_transformed[0, 0, 0]) * 180 / np.pi
    # negative sign is to make the sign of the angle to correspond to one in a right-handed coordinate system
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
    draw lines to indicate the orientation of each raft
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
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1),
                                (rafts_loc[raft_id, 0] - text_size[0] // 2, rafts_loc[raft_id, 1] + text_size[1] // 2),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_effused_raft_count(img_bgr, raft_effused, raft_to_left, raft_to_right, topleft_x, topleft_y, width_x,
                            height_y):
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
    output_img = cv.line(output_img, (topleft_x + width_x // 2, topleft_y),
                         (topleft_x + width_x // 2, topleft_y + height_y), line_color, line_thickness)
    output_img = cv.putText(output_img, 'Effused: ' + str(raft_effused), (topleft_x, topleft_y - 30), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'To left: ' + str(raft_to_left), (topleft_x, topleft_y - 60), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'To right: ' + str(raft_to_right), (topleft_x, topleft_y - 90), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


# functions used in the post-processing file
def calculate_centers_of_mass(x_all, y_all):
    """
    calculate the centers of all rafts for each frame
    xAll - x position, (# of frames, # of rafts), unit: pixel
    yAll - y position (# of frames, # of rafts)
    """
    num_of_frames, num_of_rafts = x_all.shape

    x_centers = x_all[:, 0:num_of_rafts].mean(axis=1)
    y_centers = y_all[:, 0:num_of_rafts].mean(axis=1)

    x_relative_to_centers = x_all - x_centers[:, np.newaxis]
    y_relative_to_centers = y_all - y_centers[:, np.newaxis]

    distances_to_centers = np.sqrt(x_relative_to_centers ** 2 + y_relative_to_centers ** 2)

    orbiting_angles = np.arctan2(y_relative_to_centers, x_relative_to_centers) * 180 / np.pi

    return distances_to_centers, orbiting_angles, x_centers, y_centers


def calculate_polar_angle(p1, p2):
    """
    calculate the polar angle of the vector from p1 to p2.
    """
    # note the negative sign before the first component, which is y component
    # the y in scikit-image is flipped.
    # it is to convert the angle into  right-handed coordinate
    # the range is from -pi to pi
    angle = np.arctan2(-(p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / np.pi

    return angle


def adjust_orbiting_angles(orbiting_angles_series, orbiting_angles_diff_threshold=200):
    """
    adjust the orbiting angles to get rid of the jump of 360 when it crosses from -180 to 180, or the reverse
    adjust single point anormaly.
    """

    orbiting_angles_diff = np.diff(orbiting_angles_series)

    index_neg = orbiting_angles_diff < -orbiting_angles_diff_threshold
    index_pos = orbiting_angles_diff > orbiting_angles_diff_threshold

    insertion_indices_neg = np.nonzero(index_neg)
    insertion_indices_pos = np.nonzero(index_pos)

    orbiting_angles_diff_corrected = orbiting_angles_diff.copy()
    orbiting_angles_diff_corrected[insertion_indices_neg[0]] += 360
    orbiting_angles_diff_corrected[insertion_indices_pos[0]] -= 360

    orbiting_angles_corrected = orbiting_angles_series.copy()
    orbiting_angles_corrected[1:] = orbiting_angles_diff_corrected[:]
    orbiting_angles_adjusted = np.cumsum(orbiting_angles_corrected)

    return orbiting_angles_adjusted


def adjust_orbiting_angles2(orbiting_angles_series, orbiting_angles_diff_threshold=200):
    """
    2nd version of ajust_orbiting_angles
    adjust the orbiting angles to get rid of the jump of 360
    when it crosses from -180 to 180, or the reverse
    orbiting_angle_series has the shape (raft num, frame num)
    """

    orbiting_angles_diff = np.diff(orbiting_angles_series, axis=1)

    index_neg = orbiting_angles_diff < -orbiting_angles_diff_threshold
    index_pos = orbiting_angles_diff > orbiting_angles_diff_threshold

    insertion_indices_neg = np.nonzero(index_neg)
    insertion_indices_pos = np.nonzero(index_pos)

    orbiting_angles_diff_corrected = orbiting_angles_diff.copy()
    orbiting_angles_diff_corrected[insertion_indices_neg[0], insertion_indices_neg[1]] += 360
    orbiting_angles_diff_corrected[insertion_indices_pos[0], insertion_indices_pos[1]] -= 360

    orbiting_angles_corrected = orbiting_angles_series.copy()
    orbiting_angles_corrected[:, 1:] = orbiting_angles_diff_corrected[:]
    orbiting_angles_adjusted = np.cumsum(orbiting_angles_corrected, axis=1)

    return orbiting_angles_adjusted


def mutual_info_matrix(time_series, num_of_bins):
    """
    Calculate mutual information for each pair of rafts

    time_series - rows are raft numbers, and columns are times
    numOfBins- numOfBins for calculating histogram
    the result is in unit of bits.
    """
    num_of_rafts, interval_width = time_series.shape
    mi_matrix = np.zeros((num_of_rafts, num_of_rafts))

    for i in range(num_of_rafts):
        for j in range(i + 1):
            i0 = time_series[i, :].copy()
            j0 = time_series[j, :].copy()
            c_xy = np.histogram2d(i0, j0, num_of_bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy) * np.log2(np.e)
            # in unit of bits,  * np.log2(np.e) to convert nats to bits
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def shannon_entropy(c):
    """calculate the Shannon entropy of 1 d data. The unit is bits """

    c_normalized = c / float(np.sum(c))
    c_normalized_nonzero = c_normalized[np.nonzero(c_normalized)]  # gives 1D array
    entropy = -sum(c_normalized_nonzero * np.log2(c_normalized_nonzero))  # unit in bits
    return entropy


def fft_distances(sampling_rate, distances):
    """
    given sampling rate and distances, and output frequency vector and one-sided power spectrum
    sampling_rate: unit Hz
    distances: numpy array, unit micron
    """
    #    sampling_interval = 1/sampling_rate # unit s
    #    times = np.linspace(0,sampling_length*sampling_interval, sampling_length)
    sampling_length = len(distances)  # total number of frames
    fft_dist = np.fft.fft(distances)
    p2 = np.abs(fft_dist / sampling_length)
    p1 = p2[0:int(sampling_length / 2) + 1]
    p1[1:-1] = 2 * p1[1:-1]  # one-sided power spectrum
    frequencies = sampling_rate / sampling_length * np.arange(0, int(sampling_length / 2) + 1)

    return frequencies, p1


def draw_clusters(img_bgr, connectivity_matrix, rafts_loc):
    """
    draw lines between centers of connected rafts
    """
    line_thickness = 2
    line_color = (0, 255, 0)
    output_img = img_bgr
    raft1s, raft2s = np.nonzero(connectivity_matrix)

    for raftA, raftB in zip(raft1s, raft2s):
        output_img = cv.line(output_img, (rafts_loc[raftA, 0], rafts_loc[raftA, 1]),
                             (rafts_loc[raftB, 0], rafts_loc[raftB, 1]), line_color, line_thickness)

    return output_img


def draw_voronoi(img_bgr, rafts_loc):
    """
    draw Voronoi patterns
    """
    points = rafts_loc
    vor = ScipyVoronoi(points)
    output_img = img_bgr
    # drawing Voronoi vertices
    vertex_size = int(3)
    vertex_color = (255, 0, 0)
    for x, y in zip(vor.vertices[:, 0], vor.vertices[:, 1]):
        output_img = cv.circle(output_img, (int(x), int(y)), vertex_size, vertex_color)

    # drawing Voronoi edges
    edge_color = (0, 255, 0)
    edge_thickness = int(2)
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            output_img = cv.line(output_img, (int(vor.vertices[simplex[0], 0]), int(vor.vertices[simplex[0], 1])),
                                 (int(vor.vertices[simplex[1], 0]), int(vor.vertices[simplex[1], 1])), edge_color,
                                 edge_thickness)

    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
            output_img = cv.line(output_img, (int(vor.vertices[i, 0]), int(vor.vertices[i, 1])),
                                 (int(far_point[0]), int(far_point[1])), edge_color, edge_thickness)
    return output_img


def draw_at_bottom_left_of_raft_number_float(img_bgr, rafts_loc, neighbor_count_wt, num_of_rafts):
    """
    write a subscript to indicate nearest neighbor count or weighted nearest neighbor count
    """
    font_face = cv.FONT_ITALIC
    font_scale = 0.5
    font_color = (0, 165, 255)  # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, '{:.2}'.format(neighbor_count_wt[raft_id]),
                                (rafts_loc[raft_id, 0] + text_size[0] // 2, rafts_loc[raft_id, 1] + text_size[1]),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_at_bottom_left_of_raft_number_integer(img_bgr, rafts_loc, neighbor_count_wt, num_of_rafts):
    """
    write a subscript to indicate nearest neighbor count or weighted nearest neighbor count
    """
    font_face = cv.FONT_ITALIC
    font_scale = 0.5
    font_color = (0, 165, 255)  # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, '{:}'.format(neighbor_count_wt[raft_id]),
                                (rafts_loc[raft_id, 0] + text_size[0] // 2, rafts_loc[raft_id, 1] + text_size[1]),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_neighbor_counts(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    """
    points = rafts_loc
    vor = ScipyVoronoi(points)
    neighbor_counts = np.zeros(num_of_rafts, dtype=int)
    for raft_id in range(num_of_rafts):
        neighbor_counts[raft_id] = np.count_nonzero(vor.ridge_points.ravel() == raft_id)

    font_face = cv.FONT_ITALIC
    font_scale = 0.5
    font_color = (0, 165, 255)  # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(neighbor_counts[raft_id]),
                                (rafts_loc[raft_id, 0] + text_size[0] // 2, rafts_loc[raft_id, 1] + text_size[1]),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def polygon_area(x, y):
    """
    calculate the area of a polygon given the x and y coordinates of vertices
    ref: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def ssa_decompose(y, dim):
    """
    from Vimal
    Singular Spectrum Analysis decomposition for a time series
    :param y: time series (array)
    :param dim: the embedding dimension
    :return: (pc, s, v) where
             pc is the matrix with the principal components of y
             s is the vector of the singular values of y given dim
             v is the matrix of the singular vectors of y given dim
    """
    n = len(y)
    t = n - (dim - 1)

    yy = linalg.hankel(y, np.zeros(dim))
    yy = yy[:-dim + 1, :] / np.sqrt(t)

    # here we use gesvd driver (as in Matlab)
    _, s, v = linalg.svd(yy, full_matrices=False, lapack_driver='gesvd')

    # find principal components
    vt = np.matrix(v).T
    pc = np.matrix(yy) * vt

    return np.asarray(pc), s, np.asarray(vt)


def ssa_reconstruct(pc, v, k):
    """
    from Vimal
    Series reconstruction for given SSA decomposition using vector of components
    :param pc: matrix with the principal components from SSA
    :param v: matrix of the singular vectors from SSA
    :param k: vector with the indices of the components to be reconstructed
    :return: the reconstructed time series
    """
    if np.isscalar(k):
        k = [k]

    if pc.ndim != 2:
        raise ValueError('pc must be a 2-dimensional matrix')

    if v.ndim != 2:
        raise ValueError('v must be a 2-dimensional matrix')

    t, dim = pc.shape
    n_points = t + (dim - 1)

    if any(filter(lambda x: dim < x or x < 0, k)):
        raise ValueError('k must be vector of indexes from range 0..%d' % dim)

    pc_comp = np.asarray(np.matrix(pc[:, k]) * np.matrix(v[:, k]).T)

    xr = np.zeros(n_points)
    times = np.zeros(n_points)

    # reconstruction loop
    for i in range(dim):
        xr[i: t + i] = xr[i: t + i] + pc_comp[:, i]
        times[i: t + i] = times[i: t + i] + 1

    xr = (xr / times) * np.sqrt(t)
    return xr


def ssa_full(time_series, embedding_dim=20, reconstruct_components=np.arange(10)):
    """
    combine SSA decomposition and reconstruction together
    """

    pc, s, v = ssa_decompose(time_series, embedding_dim)
    time_series_reconstructed = ssa_reconstruct(pc, v, reconstruct_components)

    return time_series_reconstructed
