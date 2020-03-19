# -*- coding: utf-8 -*-
"""
Sections:
- import libraries and define functions
- loading all the data in a specific main folder into mainDataList
- load data corresponding to a specific experiment (subfolder or video) into variables
- load variables from postprocessed file corresponding to the specific experiment above
- velocity and MSD analysis, velocity as a function of radial distance from COM
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


# %% velocity and MSD analysis, velocity as a function of radial distance from COM
# SSA parameters
embeddingDimension = 20
reconstructionComponents = np.arange(5)

raftLocationsX = raftLocations[:, :, 0]
raftLocationsY = raftLocations[:, :, 1]

raftVelocityX = np.gradient(raftLocationsX, axis=1)  # unit pixel/frame
raftVelocityY = np.gradient(raftLocationsY, axis=1)  # unit pixel/frame
raftVelocityNorm = np.sqrt(raftVelocityX ** 2 + raftVelocityY ** 2)

raftVelocityXFiltered = np.zeros_like(raftVelocityX)
raftVelocityYFiltered = np.zeros_like(raftVelocityY)
for raftID in np.arange(numOfRafts):
    raftVelocityXFiltered[raftID, :] = fsr.ssa_full(raftVelocityX[raftID, :],
                                                    embeddingDimension, reconstructionComponents)
    raftVelocityYFiltered[raftID, :] = fsr.ssa_full(raftVelocityY[raftID, :],
                                                    embeddingDimension, reconstructionComponents)

raftVelocityNormFiltered = np.sqrt(raftVelocityXFiltered ** 2 + raftVelocityYFiltered ** 2)

raftKineticEnergies = raftVelocityNormFiltered ** 2
raftKineticEnergiesSumAllRafts = raftKineticEnergies.sum(axis=0)

# get the radial and tangential vectors
raftOrbitingCentersXBroadcasted = np.broadcast_to(raftOrbitingCenters[:, 0], raftLocationsX.shape)
raftOrbitingCentersYBroadcasted = np.broadcast_to(raftOrbitingCenters[:, 1], raftLocationsY.shape)
raftRadialVectorX = raftLocationsX - raftOrbitingCentersXBroadcasted
raftRadialVectorY = raftLocationsY - raftOrbitingCentersYBroadcasted
raftRadialVectorXUnitized = raftRadialVectorX / np.sqrt(raftRadialVectorX ** 2 + raftRadialVectorY ** 2)
raftRadialVectorYUnitized = raftRadialVectorY / np.sqrt(raftRadialVectorX ** 2 + raftRadialVectorY ** 2)
raftTangentialVectorXUnitized = -raftRadialVectorYUnitized
# negative sign is assigned such that the tangential velocity is positive
raftTangentialVectorYUnitized = raftRadialVectorXUnitized
# get the radial and tangential velocities
raftRadialVelocity = raftVelocityXFiltered * raftRadialVectorXUnitized + \
                     raftVelocityYFiltered * raftRadialVectorYUnitized
raftTangentialVelocity = raftVelocityXFiltered * raftTangentialVectorXUnitized + \
                         raftVelocityYFiltered * raftTangentialVectorYUnitized

particleMSD = np.zeros((numOfRafts, numOfFrames))
particleMSDstd = np.zeros((numOfRafts, numOfFrames))
particleRMSD = np.zeros((numOfRafts, numOfFrames))

for raftID in np.arange(numOfRafts):
    corrX = raftLocations[raftID, :, 0]
    corrY = raftLocations[raftID, :, 1]

    # construct Toeplitz matrix, 1st column is the corrdinateX and Y, top right half all zeros
    corrXToeplitz = scipy.linalg.toeplitz(corrX, np.zeros(numOfFrames))
    corrYToeplitz = scipy.linalg.toeplitz(corrY, np.zeros(numOfFrames))

    # broad cast the column of coordinate x and y to the size of Toeplitz matrix
    corrXBroadcasted = np.transpose(np.broadcast_to(corrX, corrXToeplitz.shape))
    corrYBroadcasted = np.transpose(np.broadcast_to(corrY, corrYToeplitz.shape))

    # substrate Toeplitz matrix from broadcasted matrix,
    # for each column, the rows on and below the diagonal are the displacement in x and y coordinates
    # step size is the column index.
    corrXdiffMatrixSquared = (corrXBroadcasted - corrXToeplitz) ** 2
    corrYdiffMatrixSquared = (corrYBroadcasted - corrYToeplitz) ** 2
    particleSquareDisplacement = corrXdiffMatrixSquared + corrYdiffMatrixSquared

    # calculate mean square displacement
    for stepSize in np.arange(numOfFrames):
        particleMSD[raftID, stepSize] = np.average(particleSquareDisplacement[stepSize:, stepSize])
        particleMSDstd[raftID, stepSize] = np.std(particleSquareDisplacement[stepSize:, stepSize])

    particleRMSD[raftID, :] = np.sqrt(particleMSD[raftID, :])

# plot to see XY_MSD
raftToLookAt = 50

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
ax[0].plot(raftLocations[raftToLookAt, :, 0], currentFrameGray.shape[1] - raftLocations[raftToLookAt, :, 1])
ax[0].set_xlabel('Position x (pixel)')
ax[0].set_ylabel('Position y (pixel)')
ax[0].set_xlim([0, currentFrameGray.shape[0]])
ax[0].set_ylim([0, currentFrameGray.shape[1]])
ax[0].set_title('raft ID = {}'.format(raftToLookAt))
ax[1].errorbar(np.arange(numOfFrames), particleMSD[raftToLookAt, :], yerr=particleMSDstd[raftToLookAt, :],
               errorevery=20)
ax[1].set_xlabel('step size')
ax[1].set_ylabel('Mean Square Displacement')
fig.show()

# plotting raft kinetic energy sums
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(np.arange(numOfFrames), raftKineticEnergiesSumAllRafts, '-o')
# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('frame #', size=20)
ax.set_ylabel('kinetic energy sum over all rafts', size=20)
ax.set_title('kinetic energy sum over all rafts, assume mass = 2', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
# ax.legend()
fig.show()

# comparing before and after SSA: check embedding and reconstruction parameters
embeddingDimension = 20
reconstructionComponents = np.arange(5)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
colors = plt.cm.viridis(np.linspace(0, 1, numOfRafts))
for i in range(0, numOfRafts, 90):
    ax.plot(np.arange(0, numOfFrames), raftVelocityX[i, :], label='before SSA {}'.format(i))
    ax.plot(np.arange(0, numOfFrames), fsr.ssa_full(raftVelocityX[i, :], embeddingDimension, reconstructionComponents),
            label='after SSA {}'.format(i))
# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('Time (frame)', size=20)
ax.set_ylabel('raft velocity in x', size=20)
ax.set_title('raft velocity in x', size=20)
ax.legend()
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# plotting tangential velocity  vs orbiting distances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(raftOrbitingDistances.flatten(), raftTangentialVelocity.flatten(), 'o')
# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('orbiting distance', size=20)
ax.set_ylabel('tangential velocity', size=20)
ax.set_title('tangential velocity vs orbiting distance', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plotting radial velocity  vs orbiting distances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftOrbitingDistances.flatten(), raftRadialVelocity.flatten(), 'o')

# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('orbiting distance', size=20)
ax.set_ylabel('radial velocity', size=20)
ax.set_title('radial velocity vs orbiting distance', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plot to check the direction of the tangential vector
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftRadialVectorXUnitized[0, 0], raftRadialVectorYUnitized[0, 0], 'o', label='radial')
ax.plot(raftTangentialVectorXUnitized[0, 0], raftTangentialVectorYUnitized[0, 0], 'o', label='tangential')
ax.plot(raftVelocityX[0, 0], raftVelocityY[0, 0], 'o', label='velocity')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_xlabel(' x', size=20)
ax.set_ylabel('y', size=20)
ax.set_title('test of tangential vector direction', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plot raft velocity x vs velocity y
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftVelocityXFiltered.flatten(), raftVelocityYFiltered.flatten(), 'o')

# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('raft velocity in x', size=20)
ax.set_ylabel('raft velocity in y', size=20)
ax.set_title('raft velocity in x', size=20)
ax.legend()
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# plotting radial vector x and y
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftRadialVectorXUnitized.flatten(), raftRadialVectorYUnitized.flatten(), 'o')

# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('radial vector x', size=20)
ax.set_ylabel('radial vector y', size=20)
ax.set_title('radial vectors', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plotting velocity norm vs orbiting distances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftOrbitingDistances.flatten(), raftVelocityNormFiltered.flatten(), 'o')

# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('orbiting distance', size=20)
ax.set_ylabel('orbiting velocity norm', size=20)
ax.set_title('orbiting velocity vs orbiting distance', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plot velocity in x direction
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0, 1, numOfRafts))

for i in range(0, numOfRafts, 10):
    ax.plot(np.arange(0, numOfFrames), raftVelocityXFiltered[i, :], c=colors[i], label='{}'.format(i))

# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('Time (frame)', size=20)
ax.set_ylabel('raft velocity in x', size=20)
ax.set_title('raft velocity in x', size=20)
ax.legend()
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# plot the velocity in y direction
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0, 1, numOfRafts))

for i in range(0, numOfRafts, 10):
    ax.plot(np.arange(0, numOfFrames), raftVelocityYFiltered[i, :], c=colors[i], label='{}'.format(i))

# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('Time (frame)', size=20)
ax.set_ylabel('raft velocity in y', size=20)
ax.set_title('raft velocity in y', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plot the velocity norm
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0, 1, numOfRafts))

for i in range(0, numOfRafts, 10):
    ax.plot(np.arange(0, numOfFrames), raftVelocityNormFiltered[i, :], c=colors[i], label='{}'.format(i))

# ax.set_xlim([0, numOfFrames])
# ax.set_ylim([0, raftOrbitingDistances.max()])
ax.set_xlabel('Time (frame)', size=20)
ax.set_ylabel('raft velocity norm', size=20)
ax.set_title('raft velocity norm', size=20)
# ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()