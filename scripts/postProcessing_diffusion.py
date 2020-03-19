# -*- coding: utf-8 -*-
"""
Sections:
- import libraries and define functions
- loading all the data in a specific main folder into mainDataList
- load data corresponding to a specific experiment (subfolder or video) into variables
- load variables from postprocessed file corresponding to the specific experiment above
- diffusion data treatment; mainly uses data from raftLocations
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

# %% diffusion data treatment; mainly uses data from raftLocations

for dataID in np.arange(18):
    numOfFrames = mainDataList[dataID]['numOfFrames']
    raftLocations = mainDataList[dataID]['raftLocations']
    currentFrameGray = mainDataList[dataID]['currentFrameGray']
    subfolderName = mainDataList[dataID]['subfolders'][mainDataList[dataID]['expID']]

    corrX = raftLocations[0, :, 0]
    corrY = raftLocations[0, :, 1]

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

    particleMSD = np.zeros(numOfFrames)
    particleMSDstd = np.zeros(numOfFrames)
    # calculate mean square displacement
    for stepSize in np.arange(numOfFrames):
        particleMSD[stepSize] = np.average(particleSquareDisplacement[stepSize:, stepSize])
        particleMSDstd[stepSize] = np.std(particleSquareDisplacement[stepSize:, stepSize])

    particleRMSD = np.sqrt(particleMSD)

    diffusionDataFrame = pd.DataFrame(columns=['StepSize', 'particleMSD', 'particleMSDstd',
                                               'particleRMSD', 'frameNum', 'corrX', 'corrY'])
    diffusionDataFrame['StepSize'] = np.arange(numOfFrames)
    diffusionDataFrame['particleMSD'] = particleMSD
    diffusionDataFrame['particleMSDstd'] = particleMSDstd
    diffusionDataFrame['particleRMSD'] = particleRMSD
    diffusionDataFrame['frameNum'] = np.arange(numOfFrames)
    diffusionDataFrame['corrX'] = corrX
    diffusionDataFrame['corrY'] = corrY

    #    diffusionDataFrame.to_csv(subfolderName + '_diffusion.csv', index = False)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    ax[0].plot(corrX, currentFrameGray.shape[1] - corrY)
    ax[0].set_xlabel('Position x (pixel)')
    ax[0].set_ylabel('Position y (pixel)')
    ax[0].set_xlim([0, currentFrameGray.shape[0]])
    ax[0].set_ylim([0, currentFrameGray.shape[1]])
    ax[1].errorbar(np.arange(numOfFrames), particleMSD, yerr=particleMSDstd, errorevery=20)
    ax[1].set_xlabel('step size')
    ax[1].set_ylabel('Mean Square Displacement')
    fig.savefig(subfolderName + '_XY_MSD.png')

# plt.show()
plt.close('all')