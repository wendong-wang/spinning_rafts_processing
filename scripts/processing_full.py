# -*- coding: utf-8 -*-
"""
processing script for spinning rafts systems
"""
import numpy as np
import cv2 as cv
import os
import glob
import shelve
import progressbar
import scripts.functions_spinning_rafts as fsr


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
radiusIntervalHough = [14, 18]  # [71, 77] for 2.5x, [21, 25] for 0.8x,
# [14, 18] for 0.57x for 5x using coaxial illumination,
# [45, 55] for 5x using ring light illumination, [10, 20] for multiple rafts 1 or 2x.
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
# not used find_circles_adaptive, but in the find_circles_thres and FindAndSortCircles
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

diffusionData = 0  # 1 - this is a diffusion data that only tracks one single particle.
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
    date, raftGeo, thinFilmProp, magneticFieldProp, commentsMain = \
        fsr.parse_main_folder_name(mainFolders[mainFolderID])

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
            numOfRafts, batchNum, spinSpeed, spinUnit, magnification, commentsSub = \
                fsr.parse_subfolder_name(videoFileList[expID])
            outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps'
            if os.path.isfile(outputDataFileName + '.dat'):
                errorMessage = '{0}.dat exists'.format(outputDataFileName)
                print(errorMessage)
                continue
            cap = cv.VideoCapture(videoFileList[expID])
            numOfFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        else:
            # parse the subfolder name, read file list, get total number of frames
            numOfRafts, batchNum, spinSpeed, spinUnit, magnification, commentsSub = \
                fsr.parse_subfolder_name(subfolders[expID])
            outputDataFileName = \
                date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + \
                str(magnification) + 'x_' + commentsSub
            if os.path.isfile(outputDataFileName + '.dat'):
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
        raftEffused = np.zeros(numOfFrames, dtype=int)
        raftToLeft = np.zeros(numOfFrames, dtype=int)
        raftToRight = np.zeros(numOfFrames, dtype=int)
        effusedRaftCount = 0
        raftMovingToLeftCount = 0
        raftMovingToRightCount = 0
        raftLocationsInRegion = np.zeros((maxNumOfRaftsInRegion, numOfFrames, 2), dtype=int)
        # (raftNum, frameNum, x(columns)&y(rows)
        raftRadiiInRegion = np.zeros((maxNumOfRaftsInRegion, numOfFrames), dtype=int)

        # read and process the first frame
        currentFrameNum = 0
        if isVideo == 1:
            retval, currentFrameBGR = cap.read()
            currentFrameGray = cv.cvtColor(currentFrameBGR, cv.COLOR_BGR2GRAY)
        else:
            currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
            # currentFrameGray = cv.imread(tiffFileList[currentFrameNum],0)
            currentFrameGray = currentFrameBGR[:, :, 1]
            # use only green channel. We found green channel has the highest contrast.

        # find circles in the first frame
        centers, radii, prevCount = \
            fsr.find_circles_adaptive(currentFrameGray, numOfRafts, radii_hough=radiusIntervalHough,
                                      adaptive_thres_blocksize=adaptiveThresBlocksize,
                                      adaptive_thres_const=adaptiveThresConst, min_sep_dist=minSepDist,
                                      raft_center_threshold=raftCenterThreshold, top_left_x=topLeftX,
                                      top_left_y=topLeftY, width_x=widthX, height_y=heightY)

        if regionalSearch == 1:
            centersInRegion, radiiInRegion, countInRegion = \
                fsr.find_circles_adaptive(currentFrameGray, maxNumOfRaftsInRegion, radii_hough=radiusIntervalHough,
                                          adaptive_thres_blocksize=adaptiveThresBlocksize,
                                          adaptive_thres_const=adaptiveThresConst,
                                          min_sep_dist=minSepDist, raft_center_threshold=raftCenterThreshold,
                                          top_left_x=regionTopLeftX, top_left_y=regionTopLeftY,
                                          width_x=regionWidth, height_y=regionHeight)
            raftLocationsInRegion[:countInRegion, 0, :] = centersInRegion[:countInRegion, :]
            raftRadiiInRegion[:countInRegion, 0] = radiiInRegion[:countInRegion]

        # detect by contours
        #        centers, radii = detect_by_contours(currentFrameGray)
        #        numOfContours, _ = centers.shape
        #        if numOfContours < numOfRafts:
        #            continue

        # sorting
        centersSorted, radiiSorted, distSorted, orbitingAnglesSorted, layerIndexSorted = \
            fsr.numbering_rafts(centers, radii, numOfRafts)

        # transfer data of the first frame to key data set
        raftLocations[:, currentFrameNum, :] = centersSorted
        raftRadii[:, currentFrameNum] = radiiSorted
        raftOrbitingCenters[currentFrameNum, :] = np.mean(centers, axis=0)
        raftOrbitingDistances[:, currentFrameNum] = distSorted
        raftOrbitingAngles[:, currentFrameNum] = orbitingAnglesSorted
        raftOrbitingLayerIndices[:, currentFrameNum] = layerIndexSorted
        if processRotation == 1:
            for raftID in np.arange(numOfRafts):
                firstImages[raftID, :, :] = \
                    fsr.crop_image(currentFrameGray, raftLocations[raftID, currentFrameNum, :], sizeOfCroppedRaftImage)
                raftOrientations[raftID, currentFrameNum] = raftInitialOrientation
                # this could change later when we have a external standard to define what degree is.

        # output images
        currentFrameDraw = currentFrameBGR.copy()
        currentFrameDraw = \
            fsr.draw_rafts(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                           raftRadii[:, currentFrameNum], numOfRafts)
        currentFrameDraw = fsr.draw_raft_number(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
        if effusionData == 1:
            currentFrameDraw = fsr.draw_effused_raft_count(currentFrameDraw, raftEffused[currentFrameNum],
                                                           raftToLeft[currentFrameNum], raftToRight[currentFrameNum],
                                                           topLeftX, topLeftY, widthX, heightY)
        if processRotation == 1:
            currentFrameDraw = \
                fsr.draw_raft_orientations(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                           raftOrientations[:, currentFrameNum],
                                           raftRadii[:, currentFrameNum], numOfRafts)
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
                currentFrameGray = currentFrameBGR[:, :, 1]
                # use only green channel. We found green channel has the highest contrast.

            if diffusionData == 1:
                # top left corner of the next search box: x-coordinate
                if raftLocations[0, currentFrameNum - 1, 0] - diffBoxRadius >= topLeftX:
                    diffBoxTopLeftX = raftLocations[0, currentFrameNum - 1, 0] - diffBoxRadius
                else:
                    diffBoxTopLeftX = topLeftX
                # top left corner of the next search box: y-coordinate
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

                centers, radii, currCount = fsr.find_circles_adaptive(currentFrameGray, numOfRafts,
                                                                      radii_hough=radiusIntervalHough,
                                                                      adaptive_thres_blocksize=adaptiveThresBlocksize,
                                                                      adaptive_thres_const=adaptiveThresConst,
                                                                      min_sep_dist=minSepDist,
                                                                      raft_center_threshold=raftCenterThreshold,
                                                                      top_left_x=diffBoxTopLeftX,
                                                                      top_left_y=diffBoxTopLeftY,
                                                                      width_x=diffBoxWidthX, height_y=diffBoxHeightY)

            else:
                # find circles by Hough transform
                centers, radii, currCount = fsr.find_circles_adaptive(currentFrameGray, numOfRafts,
                                                                      radii_hough=radiusIntervalHough,
                                                                      adaptive_thres_blocksize=adaptiveThresBlocksize,
                                                                      adaptive_thres_const=adaptiveThresConst,
                                                                      min_sep_dist=minSepDist,
                                                                      raft_center_threshold=raftCenterThreshold,
                                                                      top_left_x=topLeftX, top_left_y=topLeftY,
                                                                      width_x=widthX, height_y=heightY)

            if regionalSearch == 1:
                centersInRegion, radiiInRegion, countInRegion = \
                    fsr.find_circles_adaptive(currentFrameGray, maxNumOfRaftsInRegion, radii_hough=radiusIntervalHough,
                                              adaptive_thres_blocksize=adaptiveThresBlocksize,
                                              adaptive_thres_const=adaptiveThresConst,
                                              min_sep_dist=minSepDist, raft_center_threshold=raftCenterThreshold,
                                              top_left_x=regionTopLeftX, top_left_y=regionTopLeftY,
                                              width_x=regionWidth, height_y=regionHeight)
                raftLocationsInRegion[:countInRegion, currentFrameNum, :] = centersInRegion[:countInRegion, :]
                raftRadiiInRegion[:countInRegion, currentFrameNum] = radiiInRegion[:countInRegion]

            # find cirlces by detect contours
            #            centers, radii = detect_by_contours(currentFrameGray)
            #            numOfContours, _ = centers.shape
            #            if numOfContours < numOfRafts:
            #                continue

            # tracking rafts according to the proximity to the previous frame, and then save to key data set, 
            if effusionData == 1:
                targetID = np.arange(numOfRafts)
                raftMovingToLeft, raftMovingToRight = \
                    fsr.counting_effused_rafts(raftLocations[:, currentFrameNum - 1, :], prevCount, centers, currCount,
                                               effusionBoundaryX, maxDisplacement)
                raftMovingToLeftCount = raftMovingToLeftCount + raftMovingToLeft
                raftMovingToRightCount = raftMovingToRightCount + raftMovingToRight
                effusedRaftCount = effusedRaftCount + raftMovingToRight - raftMovingToLeft
                raftEffused[currentFrameNum] = effusedRaftCount
                raftToLeft[currentFrameNum] = raftMovingToLeftCount
                raftToRight[currentFrameNum] = raftMovingToRightCount
                prevCount = currCount
            else:
                targetID = fsr.tracking_rafts(raftLocations[:, currentFrameNum - 1, :], centers)

                # find and track rafts together.
                centers, radii, detectedNum = \
                    fsr.find_and_sort_circles(currentFrameGray, numOfRafts,
                                              prev_pos=raftLocations[:, currentFrameNum-1, :],
                                              radii_hough=radiusIntervalHough, thres_value=thresholdingValue,
                                              sigma_Canny=sigmaCanny, low_threshold_canny=lowThresholdCanny,
                                              high_threshold_canny=highThresholdCanny, max_displ=maxDisplacement)

            # filling key dataset after using FindCircles or DetectByCountours
            raftOrbitingCenters[currentFrameNum, :] = np.mean(centers, axis=0)
            raftOrbitingLayerIndices[:, currentFrameNum] = 1
            for raftID in np.arange(numOfRafts):
                raftLocations[raftID, currentFrameNum, :] = centers[targetID[raftID], :]
                raftRadii[raftID, currentFrameNum] = radii[targetID[raftID]]
                raftOrbitingDistances[raftID, currentFrameNum] = fsr.calculate_distance(
                    raftOrbitingCenters[currentFrameNum, :], raftLocations[raftID, currentFrameNum, :])
                raftOrbitingAngles[raftID, currentFrameNum] = fsr.calculate_orbiting_angle(
                    raftOrbitingCenters[currentFrameNum, :], raftLocations[raftID, currentFrameNum, :])

            # filling key dataset after using FindAndSortCircles
            # raftOrbitingCenters[currentFrameNum, :] = np.mean(centers, axis=0)
            # raftOrbitingLayerIndices[:, currentFrameNum] = 1
            # for raftID in np.arange(numOfRafts):
            #     raftLocations[raftID, currentFrameNum,:] = centers[raftID, :]
            #     raftRadii[raftID, currentFrameNum] = radii[raftID]
            #     raftOrbitingDistances[raftID, currentFrameNum] = \
            #         fsr.calculate_distance(raftOrbitingCenters[currentFrameNum, :],
            #                                raftLocations[raftID,currentFrameNum, :])
            #     raftOrbitingAngles[raftID, currentFrameNum] = \
            #         fsr.calculate_orbiting_angle(raftOrbitingCenters[currentFrameNum, :],
            #                                      raftLocations[raftID, currentFrameNum, :])

            # now deal with rotation
            if processRotation == 1:
                for raftID in np.arange(numOfRafts):
                    currImages[raftID, :, :] = fsr.crop_image(currentFrameGray,
                                                              raftLocations[raftID, currentFrameNum, :],
                                                              sizeOfCroppedRaftImage)
                    rotationAngle = fsr.get_rotation_angle(firstImages[raftID, :, :], currImages[raftID, :, :], 15)
                    raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, 0] + rotationAngle
                    while raftOrientations[raftID, currentFrameNum] < 0:
                        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] + 360
                    while raftOrientations[raftID, currentFrameNum] > 360:
                        raftOrientations[raftID, currentFrameNum] = raftOrientations[raftID, currentFrameNum] - 360

            # output images
            currentFrameDraw = currentFrameBGR.copy()
            currentFrameDraw = fsr.draw_rafts(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                              raftRadii[:, currentFrameNum], numOfRafts)
            currentFrameDraw = fsr.draw_raft_number(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
            if effusionData == 1:
                currentFrameDraw = fsr.draw_effused_raft_count(currentFrameDraw, raftEffused[currentFrameNum],
                                                               raftToLeft[currentFrameNum],
                                                               raftToRight[currentFrameNum],
                                                               topLeftX, topLeftY, widthX, heightY)
            if processRotation == 1:
                currentFrameDraw = fsr.draw_raft_orientations(currentFrameDraw, raftLocations[:, currentFrameNum, :],
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
