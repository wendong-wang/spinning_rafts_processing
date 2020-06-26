# -*- coding: utf-8 -*-
"""
Sections:
- import libraries and define functions
- loading all the data in a specific main folder into mainDataList
- load data corresponding to a specific experiment (subfolder or video) into variables
- load variables from postprocessed file corresponding to the specific experiment above
- some simple plots just to look at the data for one specific experiment
- cluster analysis
- some plots to look at pairwise data and cluster information.
- drawing clusters and saving into movies
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


mainFolderID = 0
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

# load the rest of the variables if necessary
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

# %% some simple plots just to look at the data for one specific experiment

# plot the center of mass
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(raftOrbitingCenters[:, 0], currentFrameGray.shape[1] - raftOrbitingCenters[:, 1])
fig.show()

# plot the center of mass, x and y coordinate separately
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(raftOrbitingCenters[:, 0], label='x')
ax.plot(raftOrbitingCenters[:, 1], label='y')
ax.legend()
fig.show()

# plot orbiting distances vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0, 1, numOfRafts))

for i in range(0, numOfRafts):
    ax.plot(np.arange(numOfFrames), raftOrbitingDistances[i, :], c=colors[i], label='{}'.format(i))

ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('Time (frame)', size=20)
ax.set_ylabel('distance to center of mass', size=20)
ax.set_title('distance to center of mass, {} Rafts'.format(numOfRafts), size=20)
ax.tick_params(axis='both', labelsize=18, width=2, length=10)
ax.legend()
fig.show()

# dfRaftOrbitingDistances = pd.DataFrame(np.transpose(raftOrbitingDistances))
# dfRaftOrbitingDistances.to_csv(outputDataFileName + '_distances.csv')

# plot orbiting Angles vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0, 1, numOfRafts))

for i in range(0, numOfRafts):
    ax.plot(np.arange(numOfFrames), raftOrbitingAngles[i, :], '-', c=colors[i], label='{}'.format(i))

ax.set_xlim([0, numOfFrames])
ax.set_ylim([raftOrbitingAngles.min(), raftOrbitingAngles.max()])
ax.set_xlabel('Frames(Time)', size=20)
ax.set_ylabel('raft orbiting angles', size=20)
ax.set_title('Raft orbiting angles, {} Rafts'.format(numOfRafts), size=20)
ax.tick_params(axis='both', labelsize=18, width=2, length=10)
ax.legend()
fig.show()

# dfRaftOrbitingAngles= pd.DataFrame(np.transpose(raftOrbitingAngles))
# dfRaftOrbitingAngles.to_csv(outputDataFileName + '_angles.csv')

# plt.close('all')

# %% cluster analysis
radius = raftRadii.mean()  # pixel  check raftRadii.mean()
scaleBar = 300 / radius / 2  # micron per pixel

raftPairwiseDistances = np.zeros((numOfRafts, numOfRafts, numOfFrames))
raftPairwiseEdgeEdgeDistancesSmallest = np.zeros((numOfRafts, numOfFrames))
raftPairwiseDistancesInRadius = np.zeros((numOfRafts, numOfRafts, numOfFrames))
raftPairwiseConnectivity = np.zeros((numOfRafts, numOfRafts, numOfFrames))

# using scipy distance module
t1 = time.perf_counter()
for frameNum in np.arange(numOfFrames):
    raftPairwiseDistances[:, :, frameNum] = scipy_distance.cdist(raftLocations[:, frameNum, :],
                                                                 raftLocations[:, frameNum, :], 'euclidean')
    # smallest nonzero eedistances is assigned to one raft as the pairwise distance,
    # connected rafts will be set to 0 later
    raftPairwiseEdgeEdgeDistancesSmallest[:, frameNum] = np.partition(raftPairwiseDistances[:, :, frameNum], 1, axis=1)[
                                                         :, 1] - radius * 2

t2 = time.perf_counter()
timeTotal = t2 - t1  # in seconds
print(timeTotal)

raftPairwiseDistancesInRadius = raftPairwiseDistances / radius

# plot the histogram of pairwise distance in radius to look at the selection
# of radius value for thresholding connectivity
frameNumToLookAt = 0
raftPairwiseDistancesInRadius_oneFrame = raftPairwiseDistancesInRadius[:, :, frameNumToLookAt]
binsForPairwiseDisttances = np.arange(0, 5, 0.1)
count, edges = np.histogram(raftPairwiseDistancesInRadius_oneFrame, bins=binsForPairwiseDisttances)

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.bar(edges[:-1], count, align='edge', width=0.05)
ax.set_xlabel('pairwise distances', {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of pairwise distances of frame {}'.format(frameNumToLookAt), {'size': 15})
ax.legend(['pairwise distances'])
fig.show()

# re-adjust connectivity thresholding if necessary
# Caution: this way of determing clusters produces errors, mostly false positive.
connectivityThreshold = 2.3  # unit: radius
# re-thresholding the connectivity matrix.
# Note that the diagonal self-distance is zero, and needs to be taken care of seperately
raftPairwiseConnectivity = np.logical_and((raftPairwiseDistancesInRadius < connectivityThreshold),
                                          (raftPairwiseDistancesInRadius > 0)) * 1

# to correct false positive, if the rafts are not connected in the next frame,
# then it is not connected in the present frame
for currentFrameNum in range(numOfFrames - 1):
    raftAs, raftBs = np.nonzero(raftPairwiseConnectivity[:, :, currentFrameNum])
    for raftA, raftB in zip(raftAs, raftBs):
        if raftPairwiseConnectivity[raftA, raftB, currentFrameNum + 1] == 0:
            raftPairwiseConnectivity[raftA, raftB, currentFrameNum] = 0

# information about clusters in all frames. For reach frame, the array has two columns,
# 1st col: cluster number, 2nd col: cluster size (excluding loners)
clusters = np.zeros((numOfRafts, 2, numOfFrames))
# clusterSizeCounts stores the number of clusters of each size for all frames.
# the first index is used as the size of the cluster
clusterSizeCounts = np.zeros((numOfRafts + 1, numOfFrames))

# fill in clusters matrix
t1 = time.perf_counter()
for frameNum in np.arange(numOfFrames):
    clusterNum = 1
    raftAs, raftBs = np.nonzero(raftPairwiseConnectivity[:, :, frameNum])
    # determine the cluster number and store the cluster number in the first column
    for raftA, raftB in zip(raftAs, raftBs):
        # to see if A and B are already registered in the raftsInClusters
        raftsInClusters = np.nonzero(clusters[:, 0, frameNum])
        A = any(raftA in raft for raft in raftsInClusters)
        B = any(raftB in raft for raft in raftsInClusters)
        # if both are new, then it is a new cluster
        if (A == False) and (B == False):
            clusters[raftA, 0, frameNum] = clusterNum
            clusters[raftB, 0, frameNum] = clusterNum
            clusterNum += 1
        # if one of them is new, then it is an old cluster
        if (A == True) and (B == False):
            clusters[raftB, 0, frameNum] = clusters[raftA, 0, frameNum]
        if (A == False) and (B == True):
            clusters[raftA, 0, frameNum] = clusters[raftB, 0, frameNum]
        # if neither is new and if their cluster numbers differ,
        # then change the larger cluster number to the smaller one
        # note that this could lead to a cluster number being jumped over
        if (A == True) and (B == True) and (clusters[raftA, 0, frameNum] != clusters[raftB, 0, frameNum]):
            clusterNumLarge = max(clusters[raftA, 0, frameNum], clusters[raftB, 0, frameNum])
            clusterNumSmall = min(clusters[raftA, 0, frameNum], clusters[raftB, 0, frameNum])
            clusters[clusters[:, 0, frameNum] == clusterNumLarge, 0, frameNum] = clusterNumSmall
            # Count the number of rafts in each cluster and store the cluster size in the second column
    numOfClusters = clusters[:, 0, frameNum].max()
    if numOfClusters > 0:
        for clusterNum in np.arange(1, numOfClusters + 1):
            clusterSize = len(clusters[clusters[:, 0, frameNum] == clusterNum, 0, frameNum])
            clusters[clusters[:, 0, frameNum] == clusterNum, 1, frameNum] = clusterSize
    raftPairwiseEdgeEdgeDistancesSmallest[np.nonzero(clusters[:, 0, frameNum]), frameNum] = 0
t2 = time.perf_counter()
timeTotal = t2 - t1  # in seconds
print(timeTotal)

# fill in clusterSizeCounts matrix
t1 = time.perf_counter()
for frameNum in np.arange(numOfFrames):
    largestClusterSize = clusters[:, 1, frameNum].max()
    # count loners
    numOfLoners = len(clusters[clusters[:, 1, frameNum] == 0, 1, frameNum])
    clusterSizeCounts[1, frameNum] = numOfLoners
    # for the rest, the number of occurrence of cluster size in the 2nd column is the cluster size
    # times the number of clusters of that size
    for clusterSize in np.arange(2, largestClusterSize + 1):
        numOfClusters = len(clusters[clusters[:, 1, frameNum] == clusterSize, 1, frameNum]) / clusterSize
        clusterSizeCounts[int(clusterSize), frameNum] = numOfClusters

t2 = time.perf_counter()
timeTotal = t2 - t1  # in seconds
print(timeTotal)

# some averaging
dummyArray = np.arange((numOfRafts + 1) * numOfFrames).reshape((numOfFrames, -1)).T
dummyArray = np.mod(dummyArray, (numOfRafts + 1))  # rows are cluster sizes, and columns are frame numbers
clusterSizeAvgIncludingLoners = np.average(dummyArray, axis=0, weights=clusterSizeCounts)
clusterSizeAvgIncludingLonersAllFrames = clusterSizeAvgIncludingLoners.mean()
print('clusterSizeAvgIncludingLonersAllFrames = {:.4}'.format(clusterSizeAvgIncludingLonersAllFrames))

clusterSizeCountsExcludingLoners = clusterSizeCounts.copy()
clusterSizeCountsExcludingLoners[1, :] = 0

clusterSizeAvgExcludingLoners, sumOfWeights = np.ma.average(dummyArray, axis=0,
                                                            weights=clusterSizeCountsExcludingLoners, returned=True)
clusterSizeAvgExcludingLonersAllFrames = clusterSizeAvgExcludingLoners.mean()
print('clusterSizeAvgExcludingLonersAllFrames = {:.4} '.format(clusterSizeAvgExcludingLonersAllFrames))

raftPairwiseEdgeEdgeDistancesSmallestMean = raftPairwiseEdgeEdgeDistancesSmallest.mean() * scaleBar
raftPairwiseEdgeEdgeDistancesSmallestStd = raftPairwiseEdgeEdgeDistancesSmallest.std() * scaleBar
numOfLonersAvgAllFrames = clusterSizeCounts[1, :].mean()
print('raftPairwiseEdgeEdgeDistancesSmallestMean = {:.3} micron'.format(raftPairwiseEdgeEdgeDistancesSmallestMean))
print('raftPairwiseEdgeEdgeDistancesSmallestStd = {:.3} micron'.format(raftPairwiseEdgeEdgeDistancesSmallestStd))
print('average number of loners = {:.3}'.format(numOfLonersAvgAllFrames))

# %% some plots to look at pairwise data and cluster information.

# plot pairwise distance to a specific raft vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.jet(np.linspace(0, 1, numOfRafts))

raft1Num = 0

for raft2Num in range(0, numOfRafts):
    ax.plot(np.arange(numOfFrames), raftPairwiseDistancesInRadius[raft1Num, raft2Num, :], c=colors[raft2Num],
            label='{}'.format(raft2Num))
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, raftPairwiseDistancesInRadius[raft1Num, :, :].max()])
ax.set_xlabel('Frames(Time)', size=20)
ax.set_ylabel('distance to raft {}'.format(raft1Num), size=20)
ax.set_title('distance to raft {}, {} Rafts'.format(raft1Num, numOfRafts), size=20)
ax.tick_params(axis='both', labelsize=18, width=2, length=10)
fig.show()

# plot the size of the cluster one specific raft belongs to vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

raftNum = 2

ax.plot(np.arange(numOfFrames), clusters[raftNum, 1, :])
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusters[raftNum, 1, :].max()])
ax.set_xlabel('Frames(Time)', size=20)
ax.set_ylabel('cluster size', size=20)
ax.set_title('the size of the cluster that include raft {}'.format(raftNum), size=20)
ax.tick_params(axis='both', labelsize=18, width=2, length=10)
fig.show()

# plot the number of clusters  vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
colors = plt.cm.jet(np.linspace(0, 1, numOfRafts))

ax.plot(np.arange(numOfFrames), np.count_nonzero(clusterSizeCounts, axis=0), label='num of clusters')
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusters[:, 0, :].max() + 0.5])
ax.set_xlabel('Frames(Time)', size=20)
ax.set_ylabel('cluster number', size=20)
ax.set_title('cluster number', size=20)
ax.tick_params(axis='both', labelsize=18, width=2, length=10)
fig.show()

# plot the number of clusters with 2, 3, 4, ...  rafts vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

rows, _ = np.nonzero(clusterSizeCounts)
maxRaftsInACluster = rows.max()

colors = plt.cm.jet(np.linspace(0, 1, maxRaftsInACluster + 1))

for numOfRaftInACluster in range(1, maxRaftsInACluster + 1):
    ax.plot(np.arange(numOfFrames), clusterSizeCounts[numOfRaftInACluster, :], c=colors[numOfRaftInACluster],
            label='{}'.format(numOfRaftInACluster))

ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusterSizeCounts.max() + 0.5])
ax.set_xlabel('Time(Frames)', size=20)
ax.set_ylabel('cluster count'.format(raft1Num), size=20)
ax.set_title(' the counts of clusters of various sizes over time', size=20)
ax.tick_params(axis='both', labelsize=18, width=2, length=10)
fig.show()

# plot average cluster sizes vs frame number

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
colors = plt.cm.jet(np.linspace(0, 1, numOfRafts))

ax.plot(np.arange(numOfFrames), clusterSizeAvgIncludingLoners, label='average cluster size including loners')
ax.plot(np.arange(numOfFrames), clusterSizeAvgExcludingLoners, label='average cluster size excluding loners')
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusterSizeAvgExcludingLoners.max() + 0.5])
ax.set_xlabel('Times(Frames)', size=20)
ax.set_ylabel('average size of clusters', size=20)
ax.set_title('average size of clusters for {}'.format(outputDataFileName), size=20)
ax.tick_params(axis='both', labelsize=18, width=2, length=10)
fig.show()
# fig.savefig(outputDataFileName+'_' + 'averageSizeOfClusters.png',dpi=300)

# plt.close('all')

# %% drawing clusters and saving into movies
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
        magnification) + 'x_clustersMarked.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)

for currentFrameNum in progressbar.progressbar(range(10)):
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    currentFrameDraw = fsr.draw_rafts(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                      raftRadii[:, currentFrameNum], numOfRafts)
    currentFrameDraw = fsr.draw_raft_number(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
    currentFrameDraw = fsr.draw_clusters(currentFrameDraw, raftPairwiseConnectivity[:, :, currentFrameNum],
                                         raftLocations[:, currentFrameNum, :])
    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
            spinSpeed) + 'rps_cluster_' + str(currentFrameNum + 1).zfill(4) + '.jpg'
        cv.imwrite(outputImageName, currentFrameDraw)
    if outputVideo == 1:
        videoOut.write(currentFrameDraw)

if outputVideo == 1:
    videoOut.release()

# plt.imshow(currentFrameDraw[:,:,::-1])

