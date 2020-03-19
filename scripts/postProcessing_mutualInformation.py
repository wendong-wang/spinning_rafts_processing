# -*- coding: utf-8 -*-
"""
Sections:
- import libraries and define functions
- loading all the data in a specific main folder into mainDataList
- load data corresponding to a specific experiment (subfolder or video) into variables
- load variables from postprocessed file corresponding to the specific experiment above
- mutual information analysis
- plots for mutual information calculations
- Analysis with cross-correlation
- testing permuatation entropy
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

# %% mutual information analysis

# the durartion for which the frames are sampled to calculate one MI
widthOfInterval = 400  # unit: number of frames,

numOfBins = 20

# The gap between two successive MI calculation.
# Try keep (numOfFrames - widthOfInterval)//samplingGap an integer
samplingGap = 200  # unit: number of frames

numOfSamples = (numOfFrames - widthOfInterval) // samplingGap + 1
sampleFrameNums = np.arange(widthOfInterval, numOfFrames, samplingGap)

# pretreatment of position data
raftOrbitingAnglesAdjusted = fsr.adjust_orbiting_angles2(raftOrbitingAngles, orbiting_angles_diff_threshold=200)
raftVelocityR = np.gradient(raftOrbitingDistances, axis=1)
raftVelocityTheta = np.gradient(raftOrbitingAnglesAdjusted, axis=1)
raftVelocityNormPolar = np.sqrt(
    raftVelocityR * raftVelocityR + np.square(raftOrbitingDistances * np.radians(raftVelocityTheta)))
raftVelocityX = np.gradient(raftLocations[:, :, 0], axis=1)
raftVelocityY = np.gradient(raftLocations[:, :, 1], axis=1)
raftVelocityNormXY = np.sqrt(np.square(raftVelocityX) + np.square(raftVelocityY))

# initiate key data sets
# 0 - orbiting distances, 1 - orbiting angles, 2 - coordinate x, 3 - coordinate y
# 4 - velocity R, 5 - velocity theta, 6 - velocity norm in polar coordinate
# 7 - velocity x, 8 - velocity y, 9 - velocity norm in xy
mutualInfoAllSamplesAllRafts = np.zeros((numOfRafts, numOfRafts, numOfSamples, 10))
# mutual information averaged over all rafts for each sample
mutualInfoAllSamplesAvgOverAllRafts = np.zeros((numOfSamples, 10))
mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly = np.zeros((numOfSamples, 10))
mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI = np.zeros((numOfSamples, 10))
# averaged over all rafts and all samples.
mutualInfoAvg = np.zeros(10)
mutualInfoAvgSelfMIOnly = np.zeros(10)
mutualInfoAvgExcludingSelfMI = np.zeros(10)

# mutual information calculation
t1 = time.perf_counter()

for i, endOfInterval in enumerate(sampleFrameNums):
    distancesMatrix = raftOrbitingDistances[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 0] = fsr.mutual_info_matrix(distancesMatrix, numOfBins)

    angleMatrix = raftOrbitingAnglesAdjusted[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 1] = fsr.mutual_info_matrix(angleMatrix, numOfBins)

    coordinateXMatrix = raftLocations[:, endOfInterval - widthOfInterval:endOfInterval, 0]
    mutualInfoAllSamplesAllRafts[:, :, i, 2] = fsr.mutual_info_matrix(coordinateXMatrix, numOfBins)

    coordinateYMatrix = raftLocations[:, endOfInterval - widthOfInterval:endOfInterval, 1]
    mutualInfoAllSamplesAllRafts[:, :, i, 3] = fsr.mutual_info_matrix(coordinateYMatrix, numOfBins)

    velocityRMatrix = raftVelocityR[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 4] = fsr.mutual_info_matrix(velocityRMatrix, numOfBins)

    velocityThetaMatrix = raftVelocityTheta[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 5] = fsr.mutual_info_matrix(velocityThetaMatrix, numOfBins)

    velocityNormPolarMatrix = raftVelocityNormPolar[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 6] = fsr.mutual_info_matrix(velocityNormPolarMatrix, numOfBins)

    velocityXMatrix = raftVelocityX[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 7] = fsr.mutual_info_matrix(velocityXMatrix, numOfBins)

    velocityYMatrix = raftVelocityY[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 8] = fsr.mutual_info_matrix(velocityYMatrix, numOfBins)

    velocityNormXYMatrix = raftVelocityNormXY[:, endOfInterval - widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:, :, i, 9] = fsr.mutual_info_matrix(velocityNormXYMatrix, numOfBins)

mutualInfoAllSamplesAvgOverAllRafts = mutualInfoAllSamplesAllRafts.mean((0, 1))
mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly = np.trace(mutualInfoAllSamplesAllRafts, axis1=0, axis2=1) / numOfRafts
mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI = \
    (mutualInfoAllSamplesAvgOverAllRafts * numOfRafts - mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly) / (numOfRafts
                                                                                                          - 1)

mutualInfoAvg = mutualInfoAllSamplesAvgOverAllRafts.mean(axis=0)
mutualInfoAvgSelfMIOnly = mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly.mean(axis=0)
mutualInfoAvgExcludingSelfMI = mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI.mean(axis=0)

t2 = time.perf_counter()
timeTotal = t2 - t1  # in seconds
print(timeTotal)

# %% plots for mutual information calculations

#   histogram and entropy of one raft
raftNum = 5
frameNum = 800
widthOfInterval = 400
numOfBins = 20

# 0 - orbiting distances, 1 - orbiting angles, 2 - coordinate x, 3 - coordinate y
# 4 - velocity R, 5 - velocity theta, 6 - velocity norm in polar coordinate
# 7 - velocity x, 8 - velocity y, 9 - velocity norm in xy
dataSelection = 9

if dataSelection == 0:
    timeSeries = raftOrbitingDistances[raftNum, frameNum - widthOfInterval:frameNum]
elif dataSelection == 1:
    timeSeries = raftOrbitingAnglesAdjusted[raftNum, frameNum - widthOfInterval:frameNum]
elif dataSelection == 2:
    timeSeries = raftLocations[raftNum, frameNum - widthOfInterval:frameNum, 0]
elif dataSelection == 3:
    timeSeries = raftLocations[raftNum, frameNum - widthOfInterval:frameNum, 1]
elif dataSelection == 4:
    timeSeries = raftVelocityR[raftNum, frameNum - widthOfInterval:frameNum]
elif dataSelection == 5:
    timeSeries = raftVelocityTheta[raftNum, frameNum - widthOfInterval:frameNum]
elif dataSelection == 6:
    timeSeries = raftVelocityNormPolar[raftNum, frameNum - widthOfInterval:frameNum]
elif dataSelection == 7:
    timeSeries = raftVelocityX[raftNum, frameNum - widthOfInterval:frameNum]
elif dataSelection == 8:
    timeSeries = raftVelocityY[raftNum, frameNum - widthOfInterval:frameNum]
elif dataSelection == 9:
    timeSeries = raftVelocityNormXY[raftNum, frameNum - widthOfInterval:frameNum]

count, edges = np.histogram(timeSeries, numOfBins)
entropy = fsr.shannon_entropy(count)

fig, ax = plt.subplots(2, 1, figsize=(10, 15))
ax[0].bar(edges[:-1], count, align='edge', width=(edges.max() - edges.min()) / numOfBins / 2)
ax[0].set_xlabel('time series id = {}'.format(dataSelection), {'size': 15})
ax[0].set_ylabel('count', {'size': 15})
ax[0].set_title(
    'histogram of time series id = {} for raft {} at frame {}, entropy: {:.3} bits,  {} bins'.format(
        dataSelection, raftNum, frameNum, entropy, numOfBins), {'size': 15})
ax[0].legend(['raft {}'.format(raftNum)])
ax[1].plot(timeSeries, np.arange(widthOfInterval), '-o')
ax[1].set_xlabel('time series id = {}'.format(dataSelection), {'size': 15})
ax[1].set_ylabel('frame number', {'size': 15})
ax[1].set_title('trajectory of raft', {'size': 15})
ax[1].legend(['raft {}'.format(raftNum)])
fig.show()

#  histogram and mutual information of two rafts: orbitingDistances
raft1Num = 3
raft2Num = 4
frameNum = 400
widthOfInterval = 400
numOfBins = 20

# 0 - orbiting distances, 1 - orbiting angles, 2 - coordinate x, 3 - coordinate y
# 4 - velocity R, 5 - velocity theta, 6 - velocity norm in polar coordinate
# 7 - velocity x, 8 - velocity y, 9 - velocity norm in xy
dataSelection = 9

if dataSelection == 0:
    timeSeries1 = raftOrbitingDistances[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftOrbitingDistances[raft2Num, frameNum - widthOfInterval:frameNum]
elif dataSelection == 1:
    timeSeries1 = raftOrbitingAnglesAdjusted[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftOrbitingAnglesAdjusted[raft2Num, frameNum - widthOfInterval:frameNum]
elif dataSelection == 2:
    timeSeries1 = raftLocations[raft1Num, frameNum - widthOfInterval:frameNum, 0]
    timeSeries2 = raftLocations[raft2Num, frameNum - widthOfInterval:frameNum, 0]
elif dataSelection == 3:
    timeSeries1 = raftLocations[raft1Num, frameNum - widthOfInterval:frameNum, 1]
    timeSeries2 = raftLocations[raft2Num, frameNum - widthOfInterval:frameNum, 1]
elif dataSelection == 4:
    timeSeries1 = raftVelocityR[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftVelocityR[raft2Num, frameNum - widthOfInterval:frameNum]
elif dataSelection == 5:
    timeSeries1 = raftVelocityTheta[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftVelocityTheta[raft2Num, frameNum - widthOfInterval:frameNum]
elif dataSelection == 6:
    timeSeries1 = raftVelocityNormPolar[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftVelocityNormPolar[raft2Num, frameNum - widthOfInterval:frameNum]
elif dataSelection == 7:
    timeSeries1 = raftVelocityX[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftVelocityX[raft2Num, frameNum - widthOfInterval:frameNum]
elif dataSelection == 8:
    timeSeries1 = raftVelocityY[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftVelocityY[raft2Num, frameNum - widthOfInterval:frameNum]
elif dataSelection == 9:
    timeSeries1 = raftVelocityNormXY[raft1Num, frameNum - widthOfInterval:frameNum]
    timeSeries2 = raftVelocityNormXY[raft2Num, frameNum - widthOfInterval:frameNum]

count1, _ = np.histogram(timeSeries1, numOfBins)
entropy1 = fsr.shannon_entropy(count1)

count2, _ = np.histogram(timeSeries2, numOfBins)
entropy2 = fsr.shannon_entropy(count2)

count, xEdges, yEdges = np.histogram2d(timeSeries1, timeSeries2, numOfBins)
jointEntropy = fsr.shannon_entropy(count)
mutualInformation = mutual_info_score(None, None, contingency=count) * np.log2(np.e)
# in unit of nats, * np.log2(np.e) to convert it to bits

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
X, Y = np.meshgrid(xEdges, yEdges)
im = ax.pcolormesh(X, Y, count, cmap='inferno')
ax.set_title('2D histogram of raft {} and {} at frame {},\n '
             'with individual entropies {:.3} and {:.3} bits, \n'
             'joint entropy: {:.3} bits, mutual information: {:.3} bits,\n'
             'using {} bins'.format(raft1Num, raft2Num, frameNum, entropy1, entropy2, jointEntropy, mutualInformation,
                                    numOfBins))
cb = fig.colorbar(im)
fig.show()

#  plotting  averaged mutual information over time

# 0 - orbiting distances, 1 - orbiting angles, 2 - coordinate x, 3 - coordinate y
# 4 - velocity R, 5 - velocity theta, 6 - velocity norm in polar coordinate
# 7 - velocity x, 8 - velocity y, 9 - velocity norm in xy
dataSelection = 9

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(sampleFrameNums, mutualInfoAllSamplesAvgOverAllRafts[:, dataSelection], '-o',
        label='mutualInfo averaged of all rafts data ID = {}'.format(dataSelection))
ax.plot(sampleFrameNums, mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly[:, dataSelection], '-o',
        label='mutualInfo self MI only averaged of all rafts data ID = {}'.format(dataSelection))
ax.plot(sampleFrameNums, mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI[:, dataSelection], '-o',
        label='mutualInfo excluding self averaged of all rafts data ID = {}'.format(dataSelection))
ax.legend()
ax.set_xlabel('frame numbers', {'size': 15})
ax.set_ylabel('mutual information in bits', {'size': 15})
ax.set_xticks(np.arange(0, sampleFrameNums[-1], 100))
fig.show()

# plotting mutual information matrix for all rafts at a specific frame number
sampleIndex = 5
frameNum = sampleFrameNums[sampleIndex]

# 0 - orbiting distances, 1 - orbiting angles, 2 - coordinate x, 3 - coordinate y
# 4 - velocity R, 5 - velocity theta, 6 - velocity norm in polar coordinate
# 7 - velocity x, 8 - velocity y, 9 - velocity norm in xy
dataSelection = 0

mutualInformationMatrix = mutualInfoAllSamplesAllRafts[:, :, sampleIndex, dataSelection].copy()

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
X, Y = np.meshgrid(np.arange(numOfRafts + 1), np.arange(numOfRafts + 1))  # +1 for the right edge of the last raft.
im = ax.pcolormesh(X, Y, mutualInformationMatrix, cmap='inferno')
ax.set_xticks(np.arange(1, numOfRafts + 1, 10))
ax.set_yticks(np.arange(1, numOfRafts + 1, 10))
ax.set_title('mutual information matrix for data ID = {} at frame number {}'.format(dataSelection, frameNum))
cb = fig.colorbar(im)
fig.show()

# plotting mutual information matrix for all rafts at a specific frame number, excluding self MI
sampleIndex = 5
frameNum = sampleFrameNums[sampleIndex]

dataSelection = 9

mutualInformationMatrix = mutualInfoAllSamplesAllRafts[:, :, sampleIndex, dataSelection].copy()

np.fill_diagonal(mutualInformationMatrix, 0)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
X, Y = np.meshgrid(np.arange(numOfRafts + 1), np.arange(numOfRafts + 1))  # +1 for the right edge of the last raft.
im = ax.pcolormesh(X, Y, mutualInformationMatrix, cmap='inferno')
ax.set_xticks(np.arange(1, numOfRafts + 1, 10))
ax.set_yticks(np.arange(1, numOfRafts + 1, 10))
ax.set_title('mutual information matrix for data ID = {} at frame number {}'.format(dataSelection, frameNum))
cb = fig.colorbar(im)
fig.show()

# plotting the mutual information between one raft and the rest rafts over time, line plot
raft1Num = 90
colors = plt.cm.jet(np.linspace(0, 1, numOfRafts))

dataSelection = 1

mutualInformationSeries = mutualInfoAllSamplesAllRafts[:, :, :, dataSelection].copy()

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
for raft2Num in range(0, numOfRafts):
    if raft1Num != raft2Num:
        ax.plot(sampleFrameNums, mutualInformationSeries[raft1Num, raft2Num, :], '-o', color=colors[raft2Num],
                label='{}'.format(raft2Num))

ax.legend(loc='best')
ax.set_xlabel('frame numbers', {'size': 15})
ax.set_ylabel('mutual information between raft {} and another raft in bits'.format(raft1Num), {'size': 15})
ax.set_xlim([0, sampleFrameNums.max()])
ax.set_ylim([0, mutualInformationSeries.max() + 0.5])
ax.set_xticks(np.arange(0, sampleFrameNums[-1], 100))
ax.set_title(
    'mutual information of data ID = {} between raft {} and another raft in bits'.format(dataSelection, raft1Num))
fig.show()

# plotting the mutual information of one raft with the rest over time, color maps
raftNum = 89
colors = plt.cm.jet(np.linspace(0, 1, numOfRafts))

dataSelection = 0

mutualInformationSeries = mutualInfoAllSamplesAllRafts[:, :, :, dataSelection].copy()

oneRaftMIMatrix = mutualInformationSeries[raftNum, :, :].copy()

oneRaftMIMatrixExcludingSelfMI = oneRaftMIMatrix.copy()
oneRaftMIMatrixExcludingSelfMI[raftNum, :] = 0

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
X, Y = np.meshgrid(np.arange(numOfSamples + 1), np.arange(numOfRafts + 1))  # +1 for the right edge of the last raft.
im = ax.pcolormesh(X, Y, oneRaftMIMatrixExcludingSelfMI, cmap='inferno')
ax.set_xticks(np.arange(1, numOfSamples + 1, 1))
ax.set_yticks(np.arange(1, numOfRafts + 1, 10))
ax.set_title('mutual information of data ID = {} of raft {} over time'.format(dataSelection, raftNum))
cb = fig.colorbar(im)
fig.show()

# fig.savefig(outputDataFileName+'_raftNum{}_'.format(raftNum) + 'MIoverTime.png',dpi=300)


# plt.close('all')


# %%   Analysis with cross-correlation
raft1Num = 89
raft2Num = 90

frameNum = 200
widthOfInterval = 100

traj1 = raftOrbitingDistances[raft1Num, frameNum - widthOfInterval:frameNum + widthOfInterval]
traj2 = raftOrbitingDistances[raft2Num, frameNum - widthOfInterval:frameNum]

fig1, ax1 = plt.subplots()
ax1.plot(np.arange(frameNum - widthOfInterval, frameNum + widthOfInterval), traj1)
ax1.plot(np.arange(frameNum - widthOfInterval, frameNum), traj2)
fig1.show()

fluctuation1 = traj1 - np.mean(traj1)
fluctuation2 = traj2 - np.mean(traj2)

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(frameNum - widthOfInterval, frameNum + widthOfInterval), fluctuation1)
ax2.plot(np.arange(frameNum - widthOfInterval, frameNum), fluctuation2)
fig2.show()

rollingCorr = np.correlate(fluctuation1, fluctuation2, 'valid')
fig3, ax3 = plt.subplots()
ax3.plot(rollingCorr)
fig3.show()

frameRate = 200
f, powerSpectrum = fsr.fft_distances(frameRate, rollingCorr)

fig4, ax4 = plt.subplots()
ax4.plot(f, powerSpectrum)
fig4.show()

plt.close('all')

# %% testing permuatation entropy

import numpy as np
import itertools


def permutation_entropy(time_series, m, delay):
    """Calculate the Permutation Entropy
    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay
    Returns:
        Vector containing Permutation Entropy
    Reference:
        [1] Massimiliano Zanin et al. Permutation Entropy and Its Main Biomedical and Econophysics Applications:
            A Review. http://www.mdpi.com/1099-4300/14/8/1553/pdf
        [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural complexity
            measure for time series. http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf
        [3] http://www.mathworks.com/matlabcentral/fileexchange/37289-permutation-entropy/content/pec.m
    """
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    c = [0] * len(permutations)

    for i in range(n - delay * (m - 1)):
        # sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
        for j in range(len(permutations)):
            if abs(permutations[j] - sorted_index_array).any() == 0:
                c[j] += 1

    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe


time_series = raftOrbitingDistances[0, :].copy()
m = 3  # order = 2
delay = 1

n = len(time_series)

permutations = np.array(list(itertools.permutations(range(m))))

c = [0] * len(permutations)

for i in range(n - delay * (m - 1)):
    # sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
    sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
    for j in range(len(permutations)):
        if abs(permutations[j] - sorted_index_array).any() == 0:
            c[j] += 1

c = [element for element in c if element != 0]
p = np.divide(np.array(c), float(sum(c)))
pe = -sum(p * np.log(p))

pe = permutation_entropy(time_series, 3, 1)

