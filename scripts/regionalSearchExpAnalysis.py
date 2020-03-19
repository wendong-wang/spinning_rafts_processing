# -*- coding: utf-8 -*-
"""
Sections:
- import libraries and define functions
- loading all the data in a specific main folder into mainDataList
- load data corresponding to a specific experiment (subfolder or video) into variables
- load variables from postprocessed file corresponding to the specific experiment above
- region search data treatment, mostly using raftLocationsInRegion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2 as cv

import scipy.io
from scipy.io import loadmat
from sklearn.metrics import mutual_info_score
from scipy.spatial import distance as scipy_distance
from scipy.spatial import Voronoi as ScipyVoronoi

import progressbar
import os
import glob
import shelve

import scripts.functions_spinning_rafts as fsr


rootFolderNameFromWindows = r'D:\\VideoProcessingFolder'  # r'E:\Data_Camera_Basler_acA800-510uc_coilSystem'
# rootFolderNameFromWindows = '/media/gardi/Seagate Backup Plus Drive/Data_Camera_Basler_acA800-510uc_coilSystem'
# rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc'
# rootFolderNameFromWindows = '/media/gardi/Elements/Data_basler'
# rootFolderNameFromWindows = r'E:\Data_Camera_Basler-acA2500-60uc'
# rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc/2018-10-09_o-D300-sym4-amp2-arcAngle30-Batch21Sep2018_Co500Au60_14mT_tiling_to be analyzed/processed'
# rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc/2018-10-09_o-D300-sym4-amp2-arcAngle30-Batch21Sep2018_Co500Au60_14mT_tiling_to be analyzed/processed/processed'
# rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc/2018-10-09_o-D300-sym4-amp2-arcAngle30-Batch21Sep2018_Co500Au60_14mT_tiling_to be analyzed'
# rootFolderNameFromWindows =  '/media/gardi/Elements/Data_basler'
# rootFolderNameFromWindows = '/media/gardi/MPI-11/Data_basler'
# rootFolderNameFromWindows = '/media/gardi/Elements/Data_PhantomMiroLab140'
# rootFolderNameFromWindows = '/home/gardi/Rafts/Experiments Data/Data_PhantomMiroLab140'
# rootFolderNameFromWindows = '/media/gardi/MPI-Data9/Data_Basler-ace2500-60uc_coilsystem'
os.chdir(rootFolderNameFromWindows)
rootFolderTreeGen = os.walk(rootFolderNameFromWindows)
_, mainFolders, _ = next(rootFolderTreeGen)


# %% loading all the data in a specific main folder into mainDataList
# at the moment, it handles one main folder at a time.

# for mainFolderID in np.arange(0,1):
#    os.chdir(mainFolders[mainFolderID])


mainFolderID = 4
os.chdir(mainFolders[mainFolderID])

dataFileList = glob.glob('*.dat')
dataFileList.sort()
dataFileListExcludingPostProcessed = dataFileList.copy()
numberOfPostprocessedFiles = 0

mainDataList = []
variableListsForAllMainData = []

for dataID in range(len(dataFileList)):
    dataFileToLoad = dataFileList[dataID].partition('.dat')[0]

    if 'postprocessed' in dataFileToLoad:
        # the list length changes as items are deleted
        del dataFileListExcludingPostProcessed[dataID - numberOfPostprocessedFiles]
        numberOfPostprocessedFiles = numberOfPostprocessedFiles + 1
        continue

    tempShelf = shelve.open(dataFileToLoad)
    variableListOfOneMainDataFile = list(tempShelf.keys())

    expDict = {}
    for key in tempShelf:
        try:
            expDict[key] = tempShelf[key]
        except TypeError:
            pass

    tempShelf.close()
    mainDataList.append(expDict)
    variableListsForAllMainData.append(variableListOfOneMainDataFile)

#    # go one level up to the root folder
#    os.chdir('..')

# %% load data corresponding to a specific experiment (subfolder or video) into variables

dataID = 0
# explicitly load variables from data file
date = mainDataList[dataID]['date']
batchNum = mainDataList[dataID]['batchNum']
spinSpeed = mainDataList[dataID]['spinSpeed']
numOfRafts = mainDataList[dataID]['numOfRafts']
numOfFrames = mainDataList[dataID]['numOfFrames']
raftRadii = mainDataList[dataID]['raftRadii']
raftLocations = mainDataList[dataID]['raftLocations']
raftOrbitingCenters = mainDataList[dataID]['raftOrbitingCenters']
raftOrbitingDistances = mainDataList[dataID]['raftOrbitingDistances']
raftOrbitingAngles = mainDataList[dataID]['raftOrbitingAngles']
raftOrbitingLayerIndices = mainDataList[dataID]['raftOrbitingLayerIndices']
magnification = mainDataList[dataID]['magnification']
commentsSub = mainDataList[dataID]['commentsSub']
currentFrameGray = mainDataList[dataID]['currentFrameGray']
raftEffused = mainDataList[dataID]['raftEffused']
subfolderName = mainDataList[dataID]['subfolders'][mainDataList[dataID]['expID']]

variableListFromProcessedFile = list(mainDataList[dataID].keys())

# load the rest if necessary
for key, value in mainDataList[dataID].items():  # loop through key-value pairs of python dictionary
    if not (key in globals()):
        globals()[key] = value

outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(
    magnification) + 'x_' + commentsSub

# %% load all variables from postprocessed file corresponding to the specific experiment above

analysisType = 5  # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi

shelveDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(
    magnification) + 'x_' + 'postprocessed' + str(analysisType)

shelveDataFileExist = glob.glob(shelveDataFileName + '.dat')

if shelveDataFileExist:
    print(shelveDataFileName + ' exists, load additional variables. ')
    tempShelf = shelve.open(shelveDataFileName)
    variableListFromPostProcessedFile = list(tempShelf.keys())

    for key in tempShelf:  # just loop through all the keys in the dictionary
        globals()[key] = tempShelf[key]

    tempShelf.close()
    print('loading complete.')

elif len(shelveDataFileExist) == 0:
    print(shelveDataFileName + ' does not exist')

# %% region search data treatment for effusion experiments only, mostly using raftLocationsInRegion

startingFrameNum = 0
# check raftsEffused, and pick the frame number when the first raft is effused (assume effusing state)
maxDisplacement = 36  # check mainDataList[dataID]['maxDisplacement'] to see if it exists.
radius = 15.5  # raftRadiiInRegion[np.nonzero(raftRadiiInRegion)].mean() # pixel
scaleBar = 300 / radius / 2  # micron per pixel
frameRate = 30  # frame per second

if 'raftLocationsInRegion' in mainDataList[dataID]:
    # dataID is assigned in the section where the data from processed file is loaded
    # these are redundant, but they prevent the automatically detected errors
    # for undeclared variables in this and the following section
    regionTopLeftX = mainDataList[dataID]['regionTopLeftX']
    regionTopLeftY = mainDataList[dataID]['regionTopLeftY']
    regionWidth = mainDataList[dataID]['regionWidth']
    regionHeight = mainDataList[dataID]['regionHeight']
    maxNumOfRaftsInRegion = mainDataList[dataID]['maxNumOfRaftsInRegion']
    raftLocationsInRegion = mainDataList[dataID]['raftLocationsInRegion']
    raftRadiiInRegion = mainDataList[dataID]['raftRadiiInRegion']
else:
    print('this main data file does not have regional search data')

dfRegionSearch = pd.DataFrame(columns=['passingRaftCount', 'enteringFrameNum', 'exitingFrameNum',
                                       'rowIndices', 'positionsXY',
                                       'avgYsInPixel', 'avgSpeedInXInPixel', 'avgSpeedInYInPixel',
                                       'avgYsInMicron', 'avgSpeedInXInMicronPerSec', 'avgSpeedInYInMicronPerSec'])

passingRaftCount = 0
numOfRaftsInPrevFrame = np.count_nonzero(raftLocationsInRegion[:, startingFrameNum, 0])
for prevRaftNum in range(0, numOfRaftsInPrevFrame):
    passingRaftCount = passingRaftCount + 1
    dfRegionSearch.loc[passingRaftCount, 'passingRaftCount'] = passingRaftCount
    dfRegionSearch.loc[passingRaftCount, 'enteringFrameNum'] = startingFrameNum
    dfRegionSearch.loc[passingRaftCount, 'exitingFrameNum'] = startingFrameNum
    dfRegionSearch.loc[passingRaftCount, 'rowIndices'] = [prevRaftNum]  # make it a list
    dfRegionSearch.loc[passingRaftCount, 'positionsXY'] = [raftLocationsInRegion[prevRaftNum, startingFrameNum, :]]

for currentFrameNum in progressbar.progressbar(range(startingFrameNum + 1, numOfFrames)):
    numOfRaftsInCurrFrame = np.count_nonzero(raftLocationsInRegion[:, currentFrameNum, 0])
    # in raftPairwiseDistances, rows - prevRaftNum; columns - currRaftNum
    raftPairwiseDistances = scipy_distance.cdist(raftLocationsInRegion[:numOfRaftsInPrevFrame, currentFrameNum - 1, :],
                                                 raftLocationsInRegion[:numOfRaftsInCurrFrame, currentFrameNum, :],
                                                 'euclidean')

    # loop over all currRaftNum. It necessitates looking for the corresponding raft in the previous frame
    if numOfRaftsInCurrFrame > 0 and numOfRaftsInPrevFrame > 0:
        # otherwise raftPairwiseDistances[:,currRaftNum].min() gives error
        for currRaftNum in range(0, numOfRaftsInCurrFrame):
            if raftPairwiseDistances[:, currRaftNum].min() < maxDisplacement:
                # this is an old raft, get its prevRaftNum from raftPairwiseDistances
                prevRowNum = np.nonzero(raftPairwiseDistances[:, currRaftNum] ==
                                        raftPairwiseDistances[:, currRaftNum].min())[0][0]
                # [0][0] just to remove array
                # use rowNumsSeries to loop over raftIndex and
                # rowNumsList to look for corresponding raft in the previous frame
                # note that the raftIndex in rowNumsSeries is the same as in dfRegionSearch
                rowNumsSeries = dfRegionSearch[dfRegionSearch.exitingFrameNum == currentFrameNum - 1]['rowIndices']
                for raftIndex, rowNumsList in rowNumsSeries.iteritems():
                    # for the correct raft, the last entry of the rowNumsList should be the prevRowNum
                    if rowNumsList[-1] == prevRowNum:
                        dfRegionSearch.loc[raftIndex, 'rowIndices'].append(currRaftNum)
                        dfRegionSearch.loc[raftIndex, 'positionsXY'].append(
                            raftLocationsInRegion[currRaftNum, currentFrameNum, :])
                        dfRegionSearch.loc[raftIndex, 'exitingFrameNum'] = currentFrameNum
            else:
                # this is a new raft, add it into the dfRegionSearch
                passingRaftCount = passingRaftCount + 1
                dfRegionSearch.loc[passingRaftCount, 'passingRaftCount'] = passingRaftCount
                dfRegionSearch.loc[passingRaftCount, 'enteringFrameNum'] = currentFrameNum
                dfRegionSearch.loc[passingRaftCount, 'exitingFrameNum'] = currentFrameNum
                dfRegionSearch.loc[passingRaftCount, 'rowIndices'] = [currRaftNum]  # make it a list
                dfRegionSearch.loc[passingRaftCount, 'positionsXY'] = [
                    raftLocationsInRegion[currRaftNum, currentFrameNum, :]]

    # reset numOfRaftsInPrevFrame
    numOfRaftsInPrevFrame = numOfRaftsInCurrFrame

# loop over all raftIndex to fill the dfRegionSearch
positionsXYListSeries = dfRegionSearch.positionsXY

for raftIndex, positionsXYList in positionsXYListSeries.iteritems():
    positionXYArray = np.array(positionsXYList)
    if positionXYArray[:, 0].size > 1:  # rafts that show up for at least two frame
        avgYsInPixel = positionXYArray[:, 1].mean()
        avgSpeedInXInPixel = (positionXYArray[0, 0] - positionXYArray[-1, 0]) / (
                    positionXYArray[:, 0].size - 1)  # unit is pixel per frame
        avgSpeedInYInPixel = (positionXYArray[0, 1] - positionXYArray[-1, 1]) / (positionXYArray[:, 1].size - 1)

        dfRegionSearch.loc[raftIndex, 'avgYsInPixel'] = avgYsInPixel
        dfRegionSearch.loc[raftIndex, 'avgYsInMicron'] = avgYsInPixel * scaleBar
        dfRegionSearch.loc[raftIndex, 'avgSpeedInXInPixel'] = avgSpeedInXInPixel
        dfRegionSearch.loc[raftIndex, 'avgSpeedInXInMicronPerSec'] = avgSpeedInXInPixel * scaleBar * frameRate
        dfRegionSearch.loc[raftIndex, 'avgSpeedInYInPixel'] = avgSpeedInYInPixel
        dfRegionSearch.loc[raftIndex, 'avgSpeedInYInMicronPerSec'] = avgSpeedInYInPixel * scaleBar * frameRate

avgYsInPixelSeries = dfRegionSearch.avgYsInPixel
avgYsInPixelArray = np.array(avgYsInPixelSeries[avgYsInPixelSeries == avgYsInPixelSeries].tolist())
# using the fact that np.nan != np.nan to remove nan;
# https://stackoverflow.com/questions/20235401/remove-nan-from-pandas-series
avgSpeedInXInPixelSeries = dfRegionSearch.avgSpeedInXInPixel
avgSpeedInXInPixelArray = np.array(
    avgSpeedInXInPixelSeries[avgSpeedInXInPixelSeries == avgSpeedInXInPixelSeries].tolist())
avgSpeedInYInPixelSeries = dfRegionSearch.avgSpeedInYInPixel
avgSpeedInYInPixelArray = np.array(
    avgSpeedInYInPixelSeries[avgSpeedInYInPixelSeries == avgSpeedInYInPixelSeries].tolist())

# average speeds in X direction vs average Y positions
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(avgYsInPixelArray, avgSpeedInXInPixelArray, 'o')
ax.legend(loc='best')
# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, clusters[raftNum, 1, :].max()])
ax.set_xlabel('averge of Y positions', size=20)
ax.set_ylabel('average speed in x direction', size=20)
ax.set_title('average speeds in X direction vs average Y positions', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# average speeds in y direction vs average Y positions
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(avgYsInPixelArray, avgSpeedInYInPixelArray, 'o')
ax.legend(loc='best')
# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, clusters[raftNum, 1, :].max()])
ax.set_xlabel('averge of Y positions', size=20)
ax.set_ylabel('average speed in y direction', size=20)
ax.set_title('average speeds in y direction vs average Y positions', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

csvColNamesRegionSearch = ['passingRaftCount', 'enteringFrameNum', 'exitingFrameNum',
                           'avgYsInPixel', 'avgSpeedInXInPixel', 'avgSpeedInYInPixel',
                           'avgYsInMicron', 'avgSpeedInXInMicronPerSec', 'avgSpeedInYInMicronPerSec']

outputDataFileNameRegionalCSV = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + \
                                'rps_' + str(magnification) + 'x_' + commentsSub + '_RegionalSearch.csv'

dfRegionSearch.to_csv(outputDataFileNameRegionalCSV, index=False, columns=csvColNamesRegionSearch)

# %% drawing for regional search - for effusion experiments only

if os.path.isdir(subfolderName):
    os.chdir(subfolderName)
else:
    print(subfolderName + ' subfolder' + ' does not exist in the current folder.')

tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()

outputImage = 0
outputVideo = 1

currentFrameBGR = cv.imread(tiffFileList[0])
outputFrameRate = 5.0

if outputVideo == 1:
    outputVideoName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(
        magnification) + 'x_RegionalSearch.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)

for currentFrameNum in progressbar.progressbar(range(len(tiffFileList))):  # range(len(tiffFileList))
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    numOfRaftsInCurrFrame = np.count_nonzero(raftLocationsInRegion[:, currentFrameNum, 0])
    currentFrameDraw = fsr.draw_rafts(currentFrameDraw, raftLocationsInRegion[:, currentFrameNum, :],
                                      raftRadiiInRegion[:, currentFrameNum], numOfRaftsInCurrFrame)
    currentFrameDraw = cv.rectangle(currentFrameDraw, (regionTopLeftX, regionTopLeftY),
                                    (regionTopLeftX + regionWidth, regionTopLeftY + regionHeight), (0, 0, 255), 2)

    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
            spinSpeed) + 'rps_RegionalSearch_' + str(currentFrameNum + 1).zfill(4) + '.jpg'
        cv.imwrite(outputImageName, currentFrameDraw)
    if outputVideo == 1:
        videoOut.write(currentFrameDraw)

if outputVideo == 1:
    videoOut.release()

plt.figure()
plt.imshow(currentFrameDraw[:, :, ::-1])