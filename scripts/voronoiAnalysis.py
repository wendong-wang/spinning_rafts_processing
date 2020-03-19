# -*- coding: utf-8 -*-
"""
Sections:
- import libraries and define functions
- loading all the data in a specific main folder into mainDataList
- load data corresponding to a specific experiment (subfolder or video) into variables
- load variables from postprocessed file corresponding to the specific experiment above
- Voronoi analysis
- plots for Voronoi analysis
- drawing Voronoi diagrams and saving into movies
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

# load the rest of variables if necessary.
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
# %% Voronoi analysis

if os.path.isdir(subfolderName):
    os.chdir(subfolderName)
else:
    print(subfolderName + ' subfolder' + ' does not exist in the current folder.')

tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()

dfNeighbors = pd.DataFrame(columns=['frameNum', 'raftID', 'localDensity',
                                    'hexaticOrderParameter', 'pentaticOrderParameter',
                                    'tetraticOrderParameter', 'neighborCount',
                                    'neighborCountWeighted',
                                    'neighborIDs', 'neighborDistances',
                                    'neighborDistanceAvg',
                                    'neighborDistanceWeightedAvg',
                                    'ridgeIndices', 'ridgeVertexPairsOfOneRaft',
                                    'ridgeLengths', 'ridgeLengthsScaled',
                                    'ridgeLengthsScaledNormalizedBySum',
                                    'ridgeLengthsScaledNormalizedByMax'])

dfNeighborsAllFrames = pd.DataFrame(columns=['frameNum', 'raftID', 'localDensity',
                                             'hexaticOrderParameter', 'pentaticOrderParameter',
                                             'tetraticOrderParameter', 'neighborCount',
                                             'neighborCountWeighted',
                                             'neighborIDs', 'neighborDistances',
                                             'neighborDistanceAvg',
                                             'neighborDistanceWeightedAvg',
                                             'ridgeIndices', 'ridgeVertexPairsOfOneRaft',
                                             'ridgeLengths', 'ridgeLengthsScaled',
                                             'ridgeLengthsScaledNormalizedBySum',
                                             'ridgeLengthsScaledNormalizedByMax'])

# code copied from cluster analysis for calculating raft pairwise distances
raftPairwiseDistances = np.zeros((numOfRafts, numOfRafts, numOfFrames))
raftPairwiseEdgeEdgeDistancesSmallest = np.zeros((numOfRafts, numOfFrames))
raftPairwiseDistancesInRadius = np.zeros((numOfRafts, numOfRafts, numOfFrames))
radius = raftRadii.mean()  # pixel  check raftRadii.mean()
for frameNum in np.arange(numOfFrames):
    raftPairwiseDistances[:, :, frameNum] = scipy_distance.cdist(raftLocations[:, frameNum, :],
                                                                 raftLocations[:, frameNum, :], 'euclidean')
    # smallest nonzero eedistances is assigned to one raft as the pairwise distance,
    raftPairwiseEdgeEdgeDistancesSmallest[:, frameNum] = np.partition(raftPairwiseDistances[:, :, frameNum], 1, axis=1)[
                                                         :, 1] - radius * 2
raftPairwiseDistancesInRadius = raftPairwiseDistances / radius


entropyByNeighborCount = np.zeros(numOfFrames)
entropyByNeighborCountWeighted = np.zeros(numOfFrames)
entropyByNeighborDistances = np.zeros(numOfFrames)
entropyByLocalDensities = np.zeros(numOfFrames)

binEdgesNeighborCountWeighted = np.arange(1, 7, 1).tolist()
binEdgesNeighborDistances = np.arange(2, 10, 0.5).tolist() + [100]
binEdgesLocalDensities = np.arange(0, 1, 0.05).tolist()

deltaR = 1
sizeOfArenaInRadius = 10000 / 150  # 1cm square arena, 150 um raft radius
radialRangeArray = np.arange(2, 100, deltaR)

hexaticOrderParameterAvgs = np.zeros(numOfFrames, dtype=np.csingle)
hexaticOrderParameterAvgNorms = np.zeros(numOfFrames)
hexaticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype=np.csingle)
hexaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
hexaticOrderParameterModuliiStds = np.zeros(numOfFrames)

pentaticOrderParameterAvgs = np.zeros(numOfFrames, dtype=np.csingle)
pentaticOrderParameterAvgNorms = np.zeros(numOfFrames)
pentaticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype=np.csingle)
pentaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
pentaticOrderParameterModuliiStds = np.zeros(numOfFrames)

tetraticOrderParameterAvgs = np.zeros(numOfFrames, dtype=np.csingle)
tetraticOrderParameterAvgNorms = np.zeros(numOfFrames)
tetraticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype=np.csingle)
tetraticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
tetraticOrderParameterModuliiStds = np.zeros(numOfFrames)

radialDistributionFunction = np.zeros((numOfFrames, len(radialRangeArray)))  # pair correlation function: g(r)
spatialCorrHexaOrderPara = np.zeros((numOfFrames, len(radialRangeArray)))
# spatial correlation of hexatic order paramter: g6(r)
spatialCorrPentaOrderPara = np.zeros((numOfFrames, len(radialRangeArray)))
# spatial correlation of pentatic order paramter: g5(r)
spatialCorrTetraOrderPara = np.zeros((numOfFrames, len(radialRangeArray)))
# spatial correlation of tetratic order paramter: g4(r)

spatialCorrHexaBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray)))
# spatial correlation of bond orientation parameter: g6(r)/g(r)
spatialCorrPentaBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray)))
# spatial correlation of bond orientation parameter: g5(r)/g(r)
spatialCorrTetraBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray)))
# spatial correlation of bond orientation parameter: g4(r)/g(r)

drawingNeighborCountWeighted = 1  # 0- no drawing, 1- drawing neighborCount, 2 - drawing neighborCountWeighted

drawingRaftOrderParameterModulii = 6  # 4 - tetratic order, 5 - pentatic order, and 6 - hexatic order

outputImage = 1
outputVideo = 0

if outputVideo == 1:
    outputFrameRate = 5.0
    currentFrameBGR = cv.imread(tiffFileList[0])
    outputVideoName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(
        magnification) + 'x_Voronoi' + str(drawingNeighborCountWeighted) + '.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)

for currentFrameNum in progressbar.progressbar(range(numOfFrames)):
    # currentFrameNum = 0
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    currentFrameDraw = fsr.draw_rafts(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                      raftRadii[:, currentFrameNum], numOfRafts)
    currentFrameDraw = fsr.draw_raft_number(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
    currentFrameDraw = fsr.draw_voronoi(currentFrameDraw, raftLocations[:, currentFrameNum, :])
    # plt.imshow(currentFrameDraw[:,:,::-1])

    vor = ScipyVoronoi(raftLocations[:, currentFrameNum, :])
    allVertices = vor.vertices
    neighborPairs = vor.ridge_points
    # row# is the index of a ridge, columns are the two point# that correspond to the ridge
    ridgeVertexPairs = np.asarray(vor.ridge_vertices)
    # row# is the index of a ridge, columns are two vertex# of the ridge
    raftPairwiseDistancesMatrix = raftPairwiseDistancesInRadius[:, :, currentFrameNum]

    for raftID in np.arange(numOfRafts):
        ridgeIndices0 = np.nonzero(neighborPairs[:, 0] == raftID)
        ridgeIndices1 = np.nonzero(neighborPairs[:, 1] == raftID)
        ridgeIndices = np.concatenate((ridgeIndices0, ridgeIndices1), axis=None)
        # index is for the index of neighborPairs or ridgeVertexPairs list
        neighborPairsOfOneRaft = neighborPairs[ridgeIndices, :]
        neighborsOfOneRaft = np.concatenate((neighborPairsOfOneRaft[neighborPairsOfOneRaft[:, 0] == raftID, 1],
                                             neighborPairsOfOneRaft[neighborPairsOfOneRaft[:, 1] == raftID, 0]))
        ridgeVertexPairsOfOneRaft = ridgeVertexPairs[ridgeIndices, :]
        neighborDistances = raftPairwiseDistancesMatrix[raftID, neighborsOfOneRaft]
        neighborDistanceAvg = neighborDistances.mean()

        # order parameters and the spatial correlation functions of the order parameters
        raftLocation = raftLocations[raftID, currentFrameNum, :]
        neighborLocations = raftLocations[neighborsOfOneRaft, currentFrameNum, :]

        # note the negative sign, it is to make the angle Rhino-like
        neighborAnglesInRad = np.arctan2(-(neighborLocations[:, 1] - raftLocation[1]),
                                         (neighborLocations[:, 0] - raftLocation[0]))
        neighborAnglesInDeg = neighborAnglesInRad * 180 / np.pi

        raftHexaticOrderParameter = np.cos(neighborAnglesInRad * 6).mean() + np.sin(neighborAnglesInRad * 6).mean() * 1j
        raftPentaticOrderParameter = np.cos(neighborAnglesInRad * 5).mean() + np.sin(
            neighborAnglesInRad * 5).mean() * 1j
        raftTetraticOrderParameter = np.cos(neighborAnglesInRad * 4).mean() + np.sin(
            neighborAnglesInRad * 4).mean() * 1j

        # calculate local density of each voronoi cell
        if np.all(ridgeVertexPairsOfOneRaft >= 0):
            vertexIDsOfOneRaft = np.unique(ridgeVertexPairsOfOneRaft)
            verticesOfOneRaft = allVertices[vertexIDsOfOneRaft]
            raftXY = raftLocations[raftID, currentFrameNum, :]

            # polar angles in plt.plot
            polarAngles = np.arctan2((verticesOfOneRaft[:, 1] - raftXY[1]),
                                     (verticesOfOneRaft[:, 0] - raftXY[0])) * 180 / np.pi

            verticesOfOneRaftSorted = verticesOfOneRaft[polarAngles.argsort()]

            voronoiCellArea = fsr.polygon_area(verticesOfOneRaftSorted[:, 0], verticesOfOneRaftSorted[:, 1])

            localDensity = radius * radius * np.pi / voronoiCellArea
        else:
            localDensity = 0

        # initialize variables related to ridge lengths
        ridgeLengths = np.zeros(len(neighborsOfOneRaft))
        ridgeLengthsScaled = np.zeros(len(neighborsOfOneRaft))
        ridgeLengthsScaledNormalizedBySum = np.zeros(len(neighborsOfOneRaft))
        ridgeLengthsScaledNormalizedByMax = np.zeros(len(neighborsOfOneRaft))

        # go through all ridges to calculate or assign ridge length
        for ridgeIndexOfOneRaft, neighborID in enumerate(neighborsOfOneRaft):
            neighborDistance = fsr.calculate_distance(raftLocations[raftID, currentFrameNum, :],
                                                      raftLocations[neighborID, currentFrameNum, :])
            if np.all(ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft] >= 0):
                vertex1ID = ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft][0]
                vertex2ID = ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft][1]
                vertex1 = allVertices[vertex1ID]
                vertex2 = allVertices[vertex2ID]
                ridgeLengths[ridgeIndexOfOneRaft] = fsr.calculate_distance(vertex1, vertex2)
                # for ridges that has one vertex outside the image (negative corrdinate)
                # set ridge length to the be the diameter of the raft
                if np.all(vertex1 >= 0) and np.all(vertex2 >= 0):
                    ridgeLengthsScaled[ridgeIndexOfOneRaft] = ridgeLengths[ridgeIndexOfOneRaft] * raftRadii[
                        neighborID, currentFrameNum] * 2 / neighborDistance
                else:
                    ridgeLengthsScaled[ridgeIndexOfOneRaft] = \
                        raftRadii[neighborID, currentFrameNum] ** 2 * 4 / neighborDistance
            else:
                # for ridges that has one vertex in the infinity ridge vertex#< 0 (= -1)
                # set ridge length to the be the diameter of the raft
                ridgeLengths[ridgeIndexOfOneRaft] = raftRadii[neighborID, currentFrameNum] * 2
                ridgeLengthsScaled[ridgeIndexOfOneRaft] = raftRadii[
                                                              neighborID, currentFrameNum] ** 2 * 4 / neighborDistance

        ridgeLengthsScaledNormalizedBySum = ridgeLengthsScaled / ridgeLengthsScaled.sum()
        ridgeLengthsScaledNormalizedByMax = ridgeLengthsScaled / ridgeLengthsScaled.max()
        neighborCountWeighted = ridgeLengthsScaledNormalizedByMax.sum()
        # assuming the neighbor having the longest ridge (scaled) counts one.
        neighborDistanceWeightedAvg = np.average(neighborDistances, weights=ridgeLengthsScaledNormalizedBySum)

        dfNeighbors.loc[raftID, 'frameNum'] = currentFrameNum
        dfNeighbors.loc[raftID, 'raftID'] = raftID
        dfNeighbors.loc[raftID, 'hexaticOrderParameter'] = raftHexaticOrderParameter
        dfNeighbors.loc[raftID, 'pentaticOrderParameter'] = raftPentaticOrderParameter
        dfNeighbors.loc[raftID, 'tetraticOrderParameter'] = raftTetraticOrderParameter
        dfNeighbors.loc[raftID, 'localDensity'] = localDensity
        dfNeighbors.loc[raftID, 'neighborCount'] = len(neighborsOfOneRaft)
        dfNeighbors.loc[raftID, 'neighborCountWeighted'] = neighborCountWeighted
        dfNeighbors.loc[raftID, 'neighborIDs'] = neighborsOfOneRaft
        dfNeighbors.loc[raftID, 'neighborDistances'] = neighborDistances
        dfNeighbors.loc[raftID, 'neighborDistanceAvg'] = neighborDistanceAvg
        dfNeighbors.loc[raftID, 'neighborDistanceWeightedAvg'] = neighborDistanceWeightedAvg
        dfNeighbors.loc[raftID, 'ridgeIndices'] = ridgeIndices
        dfNeighbors.loc[raftID, 'ridgeVertexPairsOfOneRaft'] = ridgeVertexPairsOfOneRaft
        dfNeighbors.loc[raftID, 'ridgeLengths'] = ridgeLengths
        dfNeighbors.loc[raftID, 'ridgeLengthsScaled'] = ridgeLengthsScaled
        dfNeighbors.loc[raftID, 'ridgeLengthsScaledNormalizedBySum'] = ridgeLengthsScaledNormalizedBySum
        dfNeighbors.loc[raftID, 'ridgeLengthsScaledNormalizedByMax'] = ridgeLengthsScaledNormalizedByMax

    hexaticOrderParameterList = dfNeighbors['hexaticOrderParameter'].tolist()
    pentaticOrderParameterList = dfNeighbors['pentaticOrderParameter'].tolist()
    tetraticOrderParameterList = dfNeighbors['tetraticOrderParameter'].tolist()
    neighborCountSeries = dfNeighbors['neighborCount']
    neighborCountWeightedList = dfNeighbors['neighborCountWeighted'].tolist()
    neighborDistancesList = np.concatenate(dfNeighbors['neighborDistances'].tolist())
    localDensitiesList = dfNeighbors['localDensity'].tolist()

    hexaticOrderParameterArray = np.array(hexaticOrderParameterList)
    hexaticOrderParameterAvgs[currentFrameNum] = hexaticOrderParameterArray.mean()
    hexaticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(
        hexaticOrderParameterAvgs[currentFrameNum].real ** 2 + hexaticOrderParameterAvgs[currentFrameNum].imag ** 2)
    hexaticOrderParameterMeanSquaredDeviations[currentFrameNum] = (
                (hexaticOrderParameterArray - hexaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
    hexaticOrderParameterMolulii = np.absolute(hexaticOrderParameterArray)
    hexaticOrderParameterModuliiAvgs[currentFrameNum] = hexaticOrderParameterMolulii.mean()
    hexaticOrderParameterModuliiStds[currentFrameNum] = hexaticOrderParameterMolulii.std()

    pentaticOrderParameterArray = np.array(pentaticOrderParameterList)
    pentaticOrderParameterAvgs[currentFrameNum] = pentaticOrderParameterArray.mean()
    pentaticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(
        pentaticOrderParameterAvgs[currentFrameNum].real ** 2 + pentaticOrderParameterAvgs[currentFrameNum].imag ** 2)
    pentaticOrderParameterMeanSquaredDeviations[currentFrameNum] = (
                (pentaticOrderParameterArray - pentaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
    pentaticOrderParameterModulii = np.absolute(pentaticOrderParameterArray)
    pentaticOrderParameterModuliiAvgs[currentFrameNum] = pentaticOrderParameterModulii.mean()
    pentaticOrderParameterModuliiStds[currentFrameNum] = pentaticOrderParameterModulii.std()

    tetraticOrderParameterArray = np.array(tetraticOrderParameterList)
    tetraticOrderParameterAvgs[currentFrameNum] = tetraticOrderParameterArray.mean()
    tetraticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(
        tetraticOrderParameterAvgs[currentFrameNum].real ** 2 + tetraticOrderParameterAvgs[currentFrameNum].imag ** 2)
    tetraticOrderParameterMeanSquaredDeviations[currentFrameNum] = (
                (tetraticOrderParameterArray - tetraticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
    tetraticOrderParameterModulii = np.absolute(tetraticOrderParameterArray)
    tetraticOrderParameterModuliiAvgs[currentFrameNum] = tetraticOrderParameterModulii.mean()
    tetraticOrderParameterModuliiStds[currentFrameNum] = tetraticOrderParameterModulii.std()

    # g(r), g6(r), g5(r), and g4(r) for this frame
    for radialIndex, radialIntervalStart in enumerate(radialRangeArray):
        radialIntervalEnd = radialIntervalStart + deltaR
        # g(r)
        js, ks = np.logical_and(raftPairwiseDistancesMatrix >= radialIntervalStart,
                                raftPairwiseDistancesMatrix < radialIntervalEnd).nonzero()
        count = len(js)
        density = numOfRafts / sizeOfArenaInRadius ** 2
        radialDistributionFunction[currentFrameNum, radialIndex] = count / (
                    2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
        # g6(r), g5(r), g4(r)
        sumOfProductsOfPsi6 = (hexaticOrderParameterArray[js] * np.conjugate(hexaticOrderParameterArray[ks])).sum().real
        spatialCorrHexaOrderPara[currentFrameNum, radialIndex] = \
            sumOfProductsOfPsi6 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
        sumOfProductsOfPsi5 = \
            (pentaticOrderParameterArray[js] * np.conjugate(pentaticOrderParameterArray[ks])).sum().real
        spatialCorrPentaOrderPara[currentFrameNum, radialIndex] = \
            sumOfProductsOfPsi5 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
        sumOfProductsOfPsi4 = \
            (tetraticOrderParameterArray[js] * np.conjugate(tetraticOrderParameterArray[ks])).sum().real
        spatialCorrTetraOrderPara[currentFrameNum, radialIndex] = \
            sumOfProductsOfPsi4 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))

        # g6(r)/g(r); g5(r)/g(r); g4(r)/g(r)
        if radialDistributionFunction[currentFrameNum, radialIndex] != 0:
            spatialCorrHexaBondOrientationOrder[currentFrameNum, radialIndex] = \
                spatialCorrHexaOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[
                    currentFrameNum, radialIndex]
            spatialCorrPentaBondOrientationOrder[currentFrameNum, radialIndex] = \
                spatialCorrPentaOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[
                    currentFrameNum, radialIndex]
            spatialCorrTetraBondOrientationOrder[currentFrameNum, radialIndex] = \
                spatialCorrTetraOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[
                    currentFrameNum, radialIndex]

    count1 = np.asarray(neighborCountSeries.value_counts())
    entropyByNeighborCount[currentFrameNum] = fsr.shannon_entropy(count1)

    count2, _ = np.histogram(np.asarray(neighborCountWeightedList), binEdgesNeighborCountWeighted)
    entropyByNeighborCountWeighted[currentFrameNum] = fsr.shannon_entropy(count2)

    count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
    entropyByNeighborDistances[currentFrameNum] = fsr.shannon_entropy(count3)

    count4, _ = np.histogram(np.asarray(localDensitiesList), binEdgesLocalDensities)
    entropyByLocalDensities[currentFrameNum] = fsr.shannon_entropy(count4)

    neighborCountWeightedList = dfNeighbors['neighborCountWeighted'].tolist()
    neighborCountList = dfNeighbors['neighborCount'].tolist()

    if drawingRaftOrderParameterModulii == 6:
        currentFrameDrawOrderPara = fsr.draw_at_bottom_left_of_raft_number_float(
            currentFrameDraw.copy(), raftLocations[:, currentFrameNum, :], hexaticOrderParameterMolulii, numOfRafts)
    elif drawingRaftOrderParameterModulii == 5:
        currentFrameDrawOrderPara = fsr.draw_at_bottom_left_of_raft_number_float(
            currentFrameDraw.copy(), raftLocations[:, currentFrameNum, :], pentaticOrderParameterModulii, numOfRafts)
    elif drawingRaftOrderParameterModulii == 4:
        currentFrameDrawOrderPara = fsr.draw_at_bottom_left_of_raft_number_float(
            currentFrameDraw.copy(), raftLocations[:, currentFrameNum, :], tetraticOrderParameterModulii, numOfRafts)

    if drawingNeighborCountWeighted == 1:
        currentFrameDrawNeighborCount = fsr.draw_at_bottom_left_of_raft_number_integer(
            currentFrameDraw.copy(), raftLocations[:, currentFrameNum, :], neighborCountList, numOfRafts)
    elif drawingNeighborCountWeighted == 2:
        currentFrameDrawNeighborCount = fsr.draw_at_bottom_left_of_raft_number_float(
            currentFrameDraw.copy(), raftLocations[:, currentFrameNum, :], neighborCountWeightedList, numOfRafts)

    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
            spinSpeed) + 'rps_Voronoi' + str(drawingNeighborCountWeighted) + '_' + str(currentFrameNum + 1).zfill(
            4) + '.jpg'
        cv.imwrite(outputImageName, currentFrameDrawNeighborCount)
        outputImageNameOrderPara = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
            spinSpeed) + 'rps_OrderPara' + str(drawingRaftOrderParameterModulii) + '_' + str(currentFrameNum + 1).zfill(
            4) + '.jpg'
        cv.imwrite(outputImageNameOrderPara, currentFrameDrawOrderPara)
    if outputVideo == 1:
        videoOut.write(currentFrameDrawNeighborCount)

    dfNeighborsAllFrames = dfNeighborsAllFrames.append(dfNeighbors, ignore_index=True)

if outputVideo == 1:
    videoOut.release()

dfNeighborsAllFrames = dfNeighborsAllFrames.infer_objects()
dfNeighborsAllFramesSorted = dfNeighborsAllFrames.sort_values(['frameNum', 'raftID'], ascending=[1, 1])

# g6(t), g5(t), g4(t): each raft has its own temporal correlation of g6, the unit of deltaT is frame
temporalCorrHexaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype=complex)
temporalCorrPentaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype=complex)
temporalCorrTetraBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype=complex)
temporalCorrHexaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype=complex)
temporalCorrPentaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype=complex)
temporalCorrTetraBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype=complex)

for raftID in np.arange(numOfRafts):
    hexaOrdParaOfOneRaftSeries = dfNeighborsAllFramesSorted.query('raftID == {}'.format(raftID)).hexaticOrderParameter
    pentaOrdParaOfOneRaftSeries = dfNeighborsAllFramesSorted.query('raftID == {}'.format(raftID)).pentaticOrderParameter
    tetraOrdParaOfOneRaftSeries = dfNeighborsAllFramesSorted.query('raftID == {}'.format(raftID)).tetraticOrderParameter

    hexaOrdParaOfOneRaftArray = np.array(hexaOrdParaOfOneRaftSeries.tolist())
    pentaOrdParaOfOneRaftArray = np.array(pentaOrdParaOfOneRaftSeries.tolist())
    tetraOrdParaOfOneRaftArray = np.array(tetraOrdParaOfOneRaftSeries.tolist())
    # construct the Toeplitz matrix, repeat input array twice to avoid the default conjugation
    hexaOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(hexaOrdParaOfOneRaftArray, hexaOrdParaOfOneRaftArray)
    pentaOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(pentaOrdParaOfOneRaftArray, pentaOrdParaOfOneRaftArray)
    tetraOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(tetraOrdParaOfOneRaftArray, tetraOrdParaOfOneRaftArray)

    # construct the conjugated array and braodcasted it to the shape of the Toeplitz matrix
    hexaOrdParaOfOneRaftArrayConjugate = np.conjugate(hexaOrdParaOfOneRaftArray)
    hexaOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(
        np.broadcast_to(hexaOrdParaOfOneRaftArrayConjugate, hexaOrdParaOfOneRaftToeplitzMatrix.shape))
    pentaOrdParaOfOneRaftArrayConjugate = np.conjugate(pentaOrdParaOfOneRaftArray)
    pentaOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(
        np.broadcast_to(pentaOrdParaOfOneRaftArrayConjugate, pentaOrdParaOfOneRaftToeplitzMatrix.shape))
    tetraOrdParaOfOneRaftArrayConjugate = np.conjugate(tetraOrdParaOfOneRaftArray)
    tetraOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(
        np.broadcast_to(tetraOrdParaOfOneRaftArrayConjugate, tetraOrdParaOfOneRaftToeplitzMatrix.shape))

    # multiply the two matrix so that for each column, the rows on and below the diagonal are the products of
    # the conjugate of psi6(t0) and psi6(t0 + tStepSize), the tStepSize is the same the column index.
    hexaOrdParaOfOneRaftBroadcastedTimesToeplitz = \
        hexaOrdParaOfOneRaftArrayConjugateBroadcasted * hexaOrdParaOfOneRaftToeplitzMatrix
    pentaOrdParaOfOneRaftBroadcastedTimesToeplitz = \
        pentaOrdParaOfOneRaftArrayConjugateBroadcasted * pentaOrdParaOfOneRaftToeplitzMatrix
    tetraOrdParaOfOneRaftBroadcastedTimesToeplitz = \
        tetraOrdParaOfOneRaftArrayConjugateBroadcasted * tetraOrdParaOfOneRaftToeplitzMatrix

    for tStepSize in np.arange(numOfFrames):
        temporalCorrHexaBondOrientationOrder[raftID, tStepSize] = np.average(
            hexaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:, tStepSize])
        temporalCorrPentaBondOrientationOrder[raftID, tStepSize] = np.average(
            pentaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:, tStepSize])
        temporalCorrTetraBondOrientationOrder[raftID, tStepSize] = np.average(
            tetraOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:, tStepSize])

temporalCorrHexaBondOrientationOrderAvgAllRafts = temporalCorrHexaBondOrientationOrder.mean(axis=0)
temporalCorrPentaBondOrientationOrderAvgAllRafts = temporalCorrPentaBondOrientationOrder.mean(axis=0)
temporalCorrTetraBondOrientationOrderAvgAllRafts = temporalCorrTetraBondOrientationOrder.mean(axis=0)

# %% plots for Voronoi analysis
frameNumToLook = 0
dfNeighborsOneFrame = dfNeighborsAllFrames[dfNeighborsAllFrames.frameNum == frameNumToLook]

dfNeighborsOneFramehexaOrdPara = dfNeighborsOneFrame['hexaticOrderParameter']
dfNeighborsOneFramePhaseAngle = np.angle(dfNeighborsOneFramehexaOrdPara, deg=True)
dfNeighborsOneFrameModulii = np.absolute(dfNeighborsOneFramehexaOrdPara)
dfNeighborsOneFrameModulii.mean()
dfNeighborsOneFrameCosPhaseAngle = np.cos(dfNeighborsOneFramePhaseAngle)

NeighborCountSeries = dfNeighborsOneFrame['neighborCount']
binEdgesNeighborCount = list(range(NeighborCountSeries.min(), NeighborCountSeries.max() + 2))
count1, _ = np.histogram(np.asarray(NeighborCountSeries), binEdgesNeighborCount)
# count1 = np.asarray(dfNeighborsOneFrame['neighborCount'].value_counts().sort_index())
entropyByNeighborCount1 = fsr.shannon_entropy(count1)
fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.bar(binEdgesNeighborCount[:-1], count1, align='edge', width=0.5)
ax.set_xlabel('neighbor counts', {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor counts, entropy: {:.3} bits'.format(entropyByNeighborCount1), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

neighborCountWeightedSeries = dfNeighborsOneFrame['neighborCountWeighted']
count2, _ = np.histogram(np.asarray(neighborCountWeightedSeries), binEdgesNeighborCountWeighted)
entropyByNeighborCountWeighted2 = fsr.shannon_entropy(count2)
fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.bar(binEdgesNeighborCountWeighted[:-1], count2, align='edge', width=0.5)
ax.set_xlabel('neighbor counts weighted', {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor counts weighted, entropy: {:.3} bits'.format(entropyByNeighborCountWeighted2),
             {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

neighborDistancesList = np.concatenate(dfNeighborsOneFrame['neighborDistances'].tolist())
count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
entropyByNeighborDistances3 = fsr.shannon_entropy(count3)
fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.bar(binEdgesNeighborDistances[:-1], count3, align='edge', width=0.2)
ax.set_xlabel('neighbor distances', {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor distances, entropy: {:.3} bits'.format(entropyByNeighborDistances3), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

localDensitiesList = dfNeighborsOneFrame['localDensity'].tolist()
count4, _ = np.histogram(np.asarray(localDensitiesList), binEdgesLocalDensities)
entropyByLocalDensities4 = fsr.shannon_entropy(count4)
fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.bar(binEdgesLocalDensities[:-1], count4, align='edge', width=0.02)
ax.set_xlabel('local densities', {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of local densities, entropy: {:.3} bits'.format(entropyByLocalDensities4), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(radialRangeArray, radialDistributionFunction[frameNumToLook, :], label='radial distribution function g(r)')
ax.set_xlabel('radial range', {'size': 15})
ax.set_ylabel('radial distribution function g(r)', {'size': 15})
ax.set_title('radial distribution function  g(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(radialRangeArray, spatialCorrHexaOrderPara[frameNumToLook, :],
        label='spatial correlation of hexatic order parameter g6(r)')
ax.set_xlabel('radial range', {'size': 15})
ax.set_ylabel('spatial correlation of hexatic order parameter g6(r)', {'size': 15})
ax.set_title('spatial correlation of hexatic order parameter g6(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(radialRangeArray, spatialCorrHexaBondOrientationOrder[frameNumToLook, :],
        label='spatial correlation of hexa bond orientational order g6(r) / g(r)')
ax.set_xlabel('radial range', {'size': 15})
ax.set_ylabel('spatial correlation of bond orientational order g6(r) / g(r)', {'size': 15})
ax.set_title('spatial correlation of bond orientational order g6(r) / g(r) of frame# {:}'.format(frameNumToLook),
             {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(radialRangeArray, spatialCorrPentaOrderPara[frameNumToLook, :],
        label='spatial correlation of Pentatic order parameter g5(r)')
ax.set_xlabel('radial range', {'size': 15})
ax.set_ylabel('spatial correlation of hexatic order parameter g5(r)', {'size': 15})
ax.set_title('spatial correlation of hexatic order parameter g5(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(radialRangeArray, spatialCorrPentaBondOrientationOrder[frameNumToLook, :],
        label='spatial correlation of penta bond orientational order g5(r) / g(r)')
ax.set_xlabel('radial range', {'size': 15})
ax.set_ylabel('spatial correlation of bond orientational order g5(r) / g(r)', {'size': 15})
ax.set_title('spatial correlation of bond orientational order g5(r) / g(r) of frame# {:}'.format(frameNumToLook),
             {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(radialRangeArray, spatialCorrTetraOrderPara[frameNumToLook, :],
        label='spatial correlation of tetratic order parameter g4(r)')
ax.set_xlabel('radial range', {'size': 15})
ax.set_ylabel('spatial correlation of tetratic order parameter g4(r)', {'size': 15})
ax.set_title('spatial correlation of tetratic order parameter g4(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(radialRangeArray, spatialCorrTetraBondOrientationOrder[frameNumToLook, :],
        label='spatial correlation of tetra bond orientational order g4(r) / g(r)')
ax.set_xlabel('radial range', {'size': 15})
ax.set_ylabel('spatial correlation of tetra bond orientational order g4(r) / g(r)', {'size': 15})
ax.set_title('spatial correlation of tetra bond orientational order g4(r) / g(r) of frame# {:}'.format(frameNumToLook),
             {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(np.arange(numOfFrames), entropyByNeighborCount, label='entropyByNeighborCount')
ax.plot(np.arange(numOfFrames), entropyByNeighborCountWeighted, label='entropyByNeighborCountWeighted')
ax.plot(np.arange(numOfFrames), entropyByNeighborDistances, label='entropyByNeighborDistances')
ax.plot(np.arange(numOfFrames), entropyByLocalDensities, label='entropyByLocalDensities')
ax.set_xlabel('frames', {'size': 15})
ax.set_ylabel('entropies', {'size': 15})
ax.set_title('entropies over frames', {'size': 15})
ax.legend(loc='best')
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(np.arange(numOfFrames), hexaticOrderParameterModuliiAvgs, label='hexatic order parameter modulii average')
ax.plot(np.arange(numOfFrames), pentaticOrderParameterModuliiAvgs, label='pentatic order parameter modulii average')
ax.plot(np.arange(numOfFrames), tetraticOrderParameterModuliiAvgs, label='tetratic order parameter modulii average')
ax.plot(np.arange(numOfFrames), hexaticOrderParameterAvgNorms, label='hexatic order parameter avg norms')
ax.plot(np.arange(numOfFrames), pentaticOrderParameterAvgNorms, label='pentatic order parameter avg norms')
ax.plot(np.arange(numOfFrames), tetraticOrderParameterAvgNorms, label='tetratic order parameter avg norms')
ax.set_xlabel('frames', {'size': 15})
ax.set_ylabel('norm of the average of the order parameters', {'size': 15})
ax.set_title('norm of the average of the order parameters', {'size': 15})
ax.legend(loc='best')
fig.show()

# plot the temporal correlation of one specific raft
raftID = 10

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(np.arange(numOfFrames)[1:], np.real(temporalCorrHexaBondOrientationOrder[raftID, 1:]),
        label='real part of g6(t)')
ax.plot(np.arange(numOfFrames)[1:], np.imag(temporalCorrHexaBondOrientationOrder[raftID, 1:]),
        label='imaginery part of g6(t)')
ax.set_xlabel('temporal step size (frame)', {'size': 15})
ax.set_ylabel('temporal correlation of hexatic order parameter: g6(t)', {'size': 15})
ax.set_title('temporal correlation of hexatic order parameter: g6(t) for raft {}'.format(raftID), {'size': 15})
ax.legend()
fig.show()

# plot the temporal correlation averaged over all rafts
fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.plot(np.arange(numOfFrames)[1:], np.real(temporalCorrHexaBondOrientationOrderAvgAllRafts[1:]),
        label='real part of g6(t) averaged over all rafts')
ax.plot(np.arange(numOfFrames)[1:], np.imag(temporalCorrHexaBondOrientationOrderAvgAllRafts[1:]),
        label='imaginery part of g6(t) averaged over all rafts')
ax.set_xlabel('temporal step size (frame)', {'size': 15})
ax.set_ylabel('averaged temporal correlation of hexatic order parameter: g6(t)', {'size': 15})
ax.set_title('averaged temporal correlation of hexatic order parameter: g6(t) for raft {}'.format(raftID), {'size': 15})
ax.legend()
fig.show()

# %% drawing Voronoi diagrams and saving into movies

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
        magnification) + 'x_Voronoi.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)

for currentFrameNum in progressbar.progressbar(range(len(tiffFileList))):
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    currentFrameDraw = fsr.draw_rafts(currentFrameDraw, raftLocations[:, currentFrameNum, :],
                                      raftRadii[:, currentFrameNum], numOfRafts)
    currentFrameDraw = fsr.draw_raft_number(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
    currentFrameDraw = fsr.draw_voronoi(currentFrameDraw, raftLocations[:, currentFrameNum, :])
    currentFrameDraw = fsr.draw_neighbor_counts(currentFrameDraw, raftLocations[:, currentFrameNum, :], numOfRafts)
    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(
            spinSpeed) + 'rps_Voronoi_' + str(currentFrameNum + 1).zfill(4) + '.jpg'
        cv.imwrite(outputImageName, currentFrameDraw)
    if outputVideo == 1:
        videoOut.write(currentFrameDraw)

if outputVideo == 1:
    videoOut.release()

# plt.imshow(currentFrameBGR[:,:,::-1])
# scipyVoronoiPlot2D(vor)
#
# plt.show()
