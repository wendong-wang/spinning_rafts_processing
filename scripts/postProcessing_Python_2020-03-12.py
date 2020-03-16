# -*- coding: utf-8 -*-
""" 
@author: wwang 

postprocessing script for spinning-rafts system

Sections:
- import libraries and define functions
- loading all the data in a specific main folder into mainDataList
- looping to plot the center of mass and check if the data needs to be re-analyzed, for all subfolders
- post-process data (cluster analysis, Voronoi analysis, and mutual information analysis)
- extract data from all the post-processed files and store in one data frame for plotting
- load data corresponding to a specific experiment (subfolder or video) into variables
- load variables from postprocessed file corresponding to the specific experiment above
- kinetic Energy calculation (Gaurav)
- some simple plots just to look at the data for one specific experiment
- diffusion data treatment; mainly uses data from raftLocations
- region search data treatment, mostly using raftLocationsInRegion
- cluster analysis
- some plots to look at pairwise data and cluster information. 
- drawing clusters and saving into movies
- Voronoi analysis
- plots for Voronoi analysis 
- drawing Voronoi diagrams and saving into movies
- mutual information analysis
- plots for mutual information calculations
- Analysis with cross-correlation
- testing permuatation entropy
- loading a matlab data file

"""
# %% import libraries and define functions

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


# %% looping to plot the center of mass and check if the data needs to be re-analyzed, for all subfolders
for dataID in range(0, len(mainDataList)):
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
    outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(
        magnification) + 'x_' + commentsSub

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    ax.plot(raftOrbitingCenters[:, 0], currentFrameGray.shape[1] - raftOrbitingCenters[:, 1])
    fig.savefig(outputDataFileName + '_COM.png')

plt.close('all')
# %% post-process data (cluster analysis, Voronoi analysis, and mutual information analysis)

analysisType = 5  # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi

listOfNewVariablesForClusterAnalysis = ['raftPairwiseDistances', 'raftPairwiseDistancesInRadius',
                                        'raftPairwiseEdgeEdgeDistancesSmallest', 'scaleBar',
                                        'raftPairwiseConnectivity', 'radius', 'connectivityThreshold',
                                        'clusters', 'clusterSizeCounts', 'dummyArray',
                                        'clusterSizeAvgIncludingLoners', 'clusterSizeAvgIncludingLonersAllFrames',
                                        'clusterSizeCountsExcludingLoners', 'clusterSizeAvgExcludingLoners',
                                        'clusterSizeAvgExcludingLonersAllFrames',
                                        'raftPairwiseEdgeEdgeDistancesSmallestMean',
                                        'raftPairwiseEdgeEdgeDistancesSmallestStd',
                                        'numOfLonersAvgAllFrames']

listOfNewVariablesForVoronoiAnalysis = ['entropyByNeighborCount',
                                        'entropyByNeighborCountWeighted',
                                        'entropyByNeighborDistances',
                                        'entropyByLocalDensities',
                                        'binEdgesNeighborCountWeighted',
                                        'binEdgesNeighborDistances',
                                        'binEdgesLocalDensities',
                                        'neighborDistanceAvgAllRafts',
                                        'neighborDistanceWeightedAvgAllRafts',
                                        'hexaticOrderParameterAvgs',
                                        'hexaticOrderParameterAvgNorms',
                                        'hexaticOrderParameterMeanSquaredDeviations',
                                        'hexaticOrderParameterModuliiAvgs',
                                        'hexaticOrderParameterModuliiStds',
                                        'pentaticOrderParameterAvgs',
                                        'pentaticOrderParameterAvgNorms',
                                        'pentaticOrderParameterMeanSquaredDeviations',
                                        'pentaticOrderParameterModuliiAvgs',
                                        'pentaticOrderParameterModuliiStds',
                                        'tetraticOrderParameterAvgs',
                                        'tetraticOrderParameterAvgNorms',
                                        'tetraticOrderParameterMeanSquaredDeviations',
                                        'tetraticOrderParameterModuliiAvgs',
                                        'tetraticOrderParameterModuliiStds',
                                        'deltaR',
                                        'radialRangeArray',
                                        'radialDistributionFunction',
                                        'spatialCorrHexaOrderPara',
                                        'spatialCorrPentaOrderPara',
                                        'spatialCorrTetraOrderPara',
                                        'spatialCorrHexaBondOrientationOrder',
                                        'spatialCorrPentaBondOrientationOrder',
                                        'spatialCorrTetraBondOrientationOrder',
                                        'spatialCorrPos',
                                        'dfNeighborsAllFrames',
                                        'temporalCorrHexaBondOrientationOrder',
                                        'temporalCorrPentaBondOrientationOrder',
                                        'temporalCorrTetraBondOrientationOrder',
                                        'temporalCorrHexaBondOrientationOrderAvgAllRafts',
                                        'temporalCorrPentaBondOrientationOrderAvgAllRafts',
                                        'temporalCorrTetraBondOrientationOrderAvgAllRafts']

listOfNewVariablesForEntropyAnalysis = ['widthOfInterval', 'numOfBins', 'samplingGap',
                                        'numOfSamples', 'sampleFrameNums',
                                        'raftOrbitingAnglesAdjusted',
                                        'raftVelocityR', 'raftVelocityTheta',
                                        'raftVelocityNormPolar', 'raftVelocityX',
                                        'raftVelocityY', 'raftVelocityNormXY',
                                        'mutualInfoAllSamplesAllRafts',
                                        'mutualInfoAllSamplesAvgOverAllRafts',
                                        'mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly',
                                        'mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI',
                                        'mutualInfoAvg', 'mutualInfoAvgSelfMIOnly',
                                        'mutualInfoAvgExcludingSelfMI']

listOfNewVariablesForVelocityMSDAnalysis = ['embeddingDimension', 'reconstructionComponents',
                                            'raftVelocityXFiltered', 'raftVelocityYFiltered',
                                            'raftVelocityNormFiltered', 'raftKineticEnergies',
                                            'raftKineticEnergiesSumAllRafts',
                                            'raftRadialVectorXUnitized', 'raftRadialVectorYUnitized',
                                            'raftTangentialVectorXUnitized', 'raftTangentialVectorYUnitized',
                                            'raftRadialVelocity', 'raftTangentialVelocity',
                                            'particleMSD', 'particleMSDstd', 'particleRMSD']

# dataID in the mainDataList corresponds to items in dataFileListExcludingPostProcessed
for dataID in range(2, len(dataFileListExcludingPostProcessed)):
    # load variables from mainDataList
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
    raftEffused = mainDataList[dataID]['raftEffused']

    # outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' \
    #                      + str(spinSpeed) + 'rps_' + str(magnification) + 'x_' + commentsSub

    shelveDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(
        magnification) + 'x_' + 'gGx_postprocessed' + str(analysisType)

    shelveDataFileExist = glob.glob(shelveDataFileName + '.dat')  # empty list is false

    if not shelveDataFileExist:
        # cluster analysis
        if analysisType == 1 or analysisType == 2 or analysisType == 4 or analysisType == 5:
            radius = raftRadii.mean()  # pixel  check raftRadii.mean()
            scaleBar = 300 / radius / 2  # micron per pixel

            raftPairwiseDistances = np.zeros((numOfRafts, numOfRafts, numOfFrames))
            raftPairwiseEdgeEdgeDistancesSmallest = np.zeros((numOfRafts, numOfFrames))
            raftPairwiseDistancesInRadius = np.zeros((numOfRafts, numOfRafts, numOfFrames))
            raftPairwiseConnectivity = np.zeros((numOfRafts, numOfRafts, numOfFrames))

            # using scipy distance module
            for frameNum in np.arange(numOfFrames):
                raftPairwiseDistances[:, :, frameNum] = scipy_distance.cdist(raftLocations[:, frameNum, :],
                                                                             raftLocations[:, frameNum, :], 'euclidean')
                # smallest nonzero eedistances is assigned to one raft as the pairwise distance,
                # connected rafts will be set to 0 later
                raftPairwiseEdgeEdgeDistancesSmallest[:, frameNum] = np.partition(raftPairwiseDistances[:, :, frameNum],
                                                                                  1, axis=1)[:, 1] - radius * 2

            raftPairwiseDistancesInRadius = raftPairwiseDistances / radius

            # Caution: this way of determining clusters produces errors, mostly false positive.
            connectivityThreshold = 2.3  # unit: radius

            # Note that the diagonal self-distance is zero, and needs to be taken care of separately
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

            # fill in clusters array
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

            # fill in clusterSizeCounts array        
            for frameNum in np.arange(numOfFrames):
                largestClusterSize = clusters[:, 1, frameNum].max()
                # count loners
                numOfLoners = len(clusters[clusters[:, 1, frameNum] == 0, 1, frameNum])
                clusterSizeCounts[1, frameNum] = numOfLoners
                # for the rest, the number of occurrence of cluster size in the 2nd column is
                # the cluster size times the number of clusters of that size
                for clusterSize in np.arange(2, largestClusterSize + 1):
                    numOfClusters = len(clusters[clusters[:, 1, frameNum] == clusterSize, 1, frameNum]) / clusterSize
                    clusterSizeCounts[int(clusterSize), frameNum] = numOfClusters

            # some averaging
            dummyArray = np.arange((numOfRafts + 1) * numOfFrames).reshape((numOfFrames, -1)).T
            dummyArray = np.mod(dummyArray, (numOfRafts + 1))  # rows are cluster sizes, and columns are frame numbers
            clusterSizeAvgIncludingLoners = np.average(dummyArray, axis=0, weights=clusterSizeCounts)
            clusterSizeAvgIncludingLonersAllFrames = clusterSizeAvgIncludingLoners.mean()

            clusterSizeCountsExcludingLoners = clusterSizeCounts.copy()
            clusterSizeCountsExcludingLoners[1, :] = 0
            clusterSizeAvgExcludingLoners, sumOfWeights = np.ma.average(dummyArray, axis=0,
                                                                        weights=clusterSizeCountsExcludingLoners,
                                                                        returned=True)
            clusterSizeAvgExcludingLonersAllFrames = clusterSizeAvgExcludingLoners.mean()

            raftPairwiseEdgeEdgeDistancesSmallestMean = raftPairwiseEdgeEdgeDistancesSmallest.mean() * scaleBar
            raftPairwiseEdgeEdgeDistancesSmallestStd = raftPairwiseEdgeEdgeDistancesSmallest.std() * scaleBar
            numOfLonersAvgAllFrames = clusterSizeCounts[1, :].mean()

        # voronoi analysis
        if analysisType == 2 or analysisType == 4 or analysisType == 5:
            entropyByNeighborCount = np.zeros(numOfFrames)
            entropyByNeighborCountWeighted = np.zeros(numOfFrames)
            entropyByNeighborDistances = np.zeros(numOfFrames)
            entropyByLocalDensities = np.zeros(numOfFrames)
            neighborDistanceAvgAllRafts = np.zeros(numOfFrames)
            neighborDistanceWeightedAvgAllRafts = np.zeros(numOfFrames)

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

            radialDistributionFunction = np.zeros(
                (numOfFrames, len(radialRangeArray)))  # pair correlation function: g(r)
            spatialCorrHexaOrderPara = np.zeros(
                (numOfFrames, len(radialRangeArray)))  # spatial correlation of hexatic order paramter: g6(r)
            spatialCorrPentaOrderPara = np.zeros(
                (numOfFrames, len(radialRangeArray)))  # spatial correlation of pentatic order paramter: g5(r)
            spatialCorrTetraOrderPara = np.zeros(
                (numOfFrames, len(radialRangeArray)))  # spatial correlation of tetratic order paramter: g4(r)
            spatialCorrHexaBondOrientationOrder = np.zeros(
                (numOfFrames, len(radialRangeArray)))  # spatial correlation of bond orientation parameter: g6(r)/g(r)
            spatialCorrPentaBondOrientationOrder = np.zeros(
                (numOfFrames, len(radialRangeArray)))  # spatial correlation of bond orientation parameter: g5(r)/g(r)
            spatialCorrTetraBondOrientationOrder = np.zeros(
                (numOfFrames, len(radialRangeArray)))  # spatial correlation of bond orientation parameter: g4(r)/g(r)

            spatialCorrPos = np.zeros((numOfFrames, len(radialRangeArray)))  # spatial correlation of positions: gG(r)

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

            for currentFrameNum in progressbar.progressbar(range(numOfFrames)):
                # currentFrameNum = 0
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
                    neighborPairsOfOneRaft = neighborPairs[ridgeIndices, :]
                    neighborsOfOneRaft = np.concatenate((neighborPairsOfOneRaft[
                                                             neighborPairsOfOneRaft[:, 0] == raftID, 1],
                                                         neighborPairsOfOneRaft[
                                                             neighborPairsOfOneRaft[:, 1] == raftID, 0]))
                    ridgeVertexPairsOfOneRaft = ridgeVertexPairs[ridgeIndices, :]
                    neighborDistances = raftPairwiseDistancesMatrix[raftID, neighborsOfOneRaft]
                    neighborDistanceAvg = neighborDistances.mean()

                    # order parameters and their spatial correlation function
                    raftLocation = raftLocations[raftID, currentFrameNum, :]
                    neighborLocations = raftLocations[neighborsOfOneRaft, currentFrameNum, :]

                    # note the negative sign, it is to make the angle in the right-handed coordinates
                    neighborAnglesInRad = np.arctan2(-(neighborLocations[:, 1] - raftLocation[1]),
                                                     (neighborLocations[:, 0] - raftLocation[0]))
                    neighborAnglesInDeg = neighborAnglesInRad * 180 / np.pi

                    raftHexaticOrderParameter = np.cos(neighborAnglesInRad * 6).mean() + np.sin(
                        neighborAnglesInRad * 6).mean() * 1j
                    raftPentaticOrderParameter = np.cos(neighborAnglesInRad * 5).mean() + np.sin(
                        neighborAnglesInRad * 5).mean() * 1j
                    raftTetraticOrderParameter = np.cos(neighborAnglesInRad * 4).mean() + np.sin(
                        neighborAnglesInRad * 4).mean() * 1j

                    # calculate the local density of Voronoi cell
                    if np.all(ridgeVertexPairsOfOneRaft >= 0):
                        vertexIDsOfOneRaft = np.unique(ridgeVertexPairsOfOneRaft)
                        verticesOfOneRaft = allVertices[vertexIDsOfOneRaft]
                        raftXY = raftLocations[raftID, currentFrameNum, :]

                        # polar angles in plt.plot
                        polarAngles = np.arctan2((verticesOfOneRaft[:, 1] - raftXY[1]),
                                                 (verticesOfOneRaft[:, 0] - raftXY[0])) * 180 / np.pi

                        verticesOfOneRaftSorted = verticesOfOneRaft[polarAngles.argsort()]

                        voronoiCellArea = fsr.polygon_area(verticesOfOneRaftSorted[:, 0],
                                                           verticesOfOneRaftSorted[:, 1])

                        localDensity = radius * radius * np.pi / voronoiCellArea
                    else:
                        localDensity = 0

                    # initialize key variables
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
                            # for ridges that has one vertex outside the image (negative coordinate)
                            # set ridge length to the be the diameter of the raft
                            if np.all(vertex1 >= 0) and np.all(vertex2 >= 0):
                                ridgeLengthsScaled[ridgeIndexOfOneRaft] = \
                                    ridgeLengths[ridgeIndexOfOneRaft] * raftRadii[neighborID, currentFrameNum] \
                                    * 2 / neighborDistance
                            else:
                                ridgeLengthsScaled[ridgeIndexOfOneRaft] = raftRadii[neighborID, currentFrameNum] ** 2 \
                                                                          * 4 / neighborDistance
                        else:
                            # for ridges that has one vertex in the infinity ridge vertex#< 0 (= -1)
                            # set ridge length to the be the diameter of the raft
                            ridgeLengths[ridgeIndexOfOneRaft] = raftRadii[neighborID, currentFrameNum] * 2
                            ridgeLengthsScaled[ridgeIndexOfOneRaft] = \
                                raftRadii[neighborID, currentFrameNum] ** 2 * 4 / neighborDistance

                    ridgeLengthsScaledNormalizedBySum = ridgeLengthsScaled / ridgeLengthsScaled.sum()
                    ridgeLengthsScaledNormalizedByMax = ridgeLengthsScaled / ridgeLengthsScaled.max()
                    neighborCountWeighted = ridgeLengthsScaledNormalizedByMax.sum()
                    # assuming the neighbor having the longest ridge counts one.
                    neighborDistanceWeightedAvg = np.average(neighborDistances,
                                                             weights=ridgeLengthsScaledNormalizedBySum)

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
                    hexaticOrderParameterAvgs[currentFrameNum].real ** 2 + hexaticOrderParameterAvgs[
                        currentFrameNum].imag ** 2)
                hexaticOrderParameterMeanSquaredDeviations[currentFrameNum] = (
                            (hexaticOrderParameterArray - hexaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
                hexaticOrderParameterMolulii = np.absolute(hexaticOrderParameterArray)
                hexaticOrderParameterModuliiAvgs[currentFrameNum] = hexaticOrderParameterMolulii.mean()
                hexaticOrderParameterModuliiStds[currentFrameNum] = hexaticOrderParameterMolulii.std()

                pentaticOrderParameterArray = np.array(pentaticOrderParameterList)
                pentaticOrderParameterAvgs[currentFrameNum] = pentaticOrderParameterArray.mean()
                pentaticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(
                    pentaticOrderParameterAvgs[currentFrameNum].real ** 2 + pentaticOrderParameterAvgs[
                        currentFrameNum].imag ** 2)
                pentaticOrderParameterMeanSquaredDeviations[currentFrameNum] = (
                            (pentaticOrderParameterArray - pentaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
                pentaticOrderParameterModulii = np.absolute(pentaticOrderParameterArray)
                pentaticOrderParameterModuliiAvgs[currentFrameNum] = pentaticOrderParameterModulii.mean()
                pentaticOrderParameterModuliiStds[currentFrameNum] = pentaticOrderParameterModulii.std()

                tetraticOrderParameterArray = np.array(tetraticOrderParameterList)
                tetraticOrderParameterAvgs[currentFrameNum] = tetraticOrderParameterArray.mean()
                tetraticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(
                    tetraticOrderParameterAvgs[currentFrameNum].real ** 2 + tetraticOrderParameterAvgs[
                        currentFrameNum].imag ** 2)
                tetraticOrderParameterMeanSquaredDeviations[currentFrameNum] = (
                            (tetraticOrderParameterArray - tetraticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
                tetraticOrderParameterModulii = np.absolute(tetraticOrderParameterArray)
                tetraticOrderParameterModuliiAvgs[currentFrameNum] = tetraticOrderParameterModulii.mean()
                tetraticOrderParameterModuliiStds[currentFrameNum] = tetraticOrderParameterModulii.std()

                angles = np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6
                # + np.angle(hexaticOrderParameterAvgs[currentFrameNum]) #0
                NDistAvg = np.asarray(neighborDistancesList).mean()
                # np.asarray(neighborDistancesList).mean() # in unit of R
                G = 2 * np.pi * np.array((np.cos(angles), np.sin(angles))).T / NDistAvg
                # np.asarray(neighborDistancesList).mean()
                cosPart = np.zeros((len(radialRangeArray), numOfRafts, len(angles)))
                cosPartRaftCount = np.zeros(len(radialRangeArray))
                # tempCount = np.zeros(len(radialRangeArray))

                # g(r) and g6(r), gG(r) for this frame
                for radialIndex, radialIntervalStart in enumerate(radialRangeArray):
                    # radialIntervalStart, radialIndex = 2, 0
                    radialIntervalEnd = radialIntervalStart + deltaR
                    # g(r)
                    js, ks = np.logical_and(raftPairwiseDistancesMatrix >= radialIntervalStart,
                                            raftPairwiseDistancesMatrix < radialIntervalEnd).nonzero()
                    count = len(js)
                    density = numOfRafts / sizeOfArenaInRadius ** 2
                    radialDistributionFunction[currentFrameNum, radialIndex] = count / (
                                2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))

                    # gG(r_)
                    for raft1ID in np.arange(numOfRafts):
                        # raft1ID = 0
                        originXY = raftLocations[raft1ID, currentFrameNum, :] / raftRadii.mean()
                        raftLocationsNew = raftLocations[:, currentFrameNum, :] / raftRadii.mean() - originXY
                        for angleID, angle in enumerate(angles):
                            # angleID, angle = 0, np.angle(hexaticOrderParameterAvgs[currentFrameNum])
                            conditionX = np.logical_and(
                                raftLocationsNew[:, 0] >= radialIntervalStart * np.cos(angle) - NDistAvg / 2,
                                raftLocationsNew[:, 0] < radialIntervalStart * np.cos(angle) + NDistAvg / 2)
                            conditionY = np.logical_and(
                                raftLocationsNew[:, 1] >= radialIntervalStart * np.sin(angle) - NDistAvg / 2,
                                raftLocationsNew[:, 1] < radialIntervalStart * np.sin(angle) + NDistAvg / 2)
                            conditionXY = np.logical_and(conditionX, conditionY)
                            if conditionXY.any():
                                vector12 = raftLocationsNew[conditionXY.nonzero()]
                                cosPart[radialIndex, raft1ID, angleID] = np.cos(
                                    G[angleID, 0] * vector12[:, 0] + G[angleID, 1] * vector12[:, 1]).sum()
                                cosPartRaftCount[radialIndex] = cosPartRaftCount[radialIndex] + np.count_nonzero(
                                    conditionXY)
                    if np.count_nonzero(cosPart[radialIndex, :, :]) > 0:
                        # tempCount[radialIndex] = np.count_nonzero(cosPart[radialIndex, :, :])
                        spatialCorrPos[currentFrameNum, radialIndex] = cosPart[radialIndex, :, :].sum() / \
                                                                       cosPartRaftCount[radialIndex]

                        # g6(r), g5(r), g4(r)
                    sumOfProductsOfPsi6 = (hexaticOrderParameterArray[js] *
                                           np.conjugate(hexaticOrderParameterArray[ks])).sum().real
                    spatialCorrHexaOrderPara[currentFrameNum, radialIndex] = \
                        sumOfProductsOfPsi6 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
                    sumOfProductsOfPsi5 = (pentaticOrderParameterArray[js] *
                                           np.conjugate(pentaticOrderParameterArray[ks])).sum().real
                    spatialCorrPentaOrderPara[currentFrameNum, radialIndex] = \
                        sumOfProductsOfPsi5 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
                    sumOfProductsOfPsi4 = (tetraticOrderParameterArray[js] *
                                           np.conjugate(tetraticOrderParameterArray[ks])).sum().real
                    spatialCorrTetraOrderPara[currentFrameNum, radialIndex] = \
                        sumOfProductsOfPsi4 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
                    # g6(r)/g(r); g5(r)/g(r); g4(r)/g(r)
                    if radialDistributionFunction[currentFrameNum, radialIndex] != 0:
                        spatialCorrHexaBondOrientationOrder[currentFrameNum, radialIndex] = \
                            spatialCorrHexaOrderPara[currentFrameNum, radialIndex] / \
                            radialDistributionFunction[currentFrameNum, radialIndex]
                        spatialCorrPentaBondOrientationOrder[currentFrameNum, radialIndex] = \
                            spatialCorrPentaOrderPara[currentFrameNum, radialIndex] / \
                            radialDistributionFunction[currentFrameNum, radialIndex]
                        spatialCorrTetraBondOrientationOrder[currentFrameNum, radialIndex] = \
                            spatialCorrTetraOrderPara[currentFrameNum, radialIndex] / \
                            radialDistributionFunction[currentFrameNum, radialIndex]

                count1 = np.asarray(neighborCountSeries.value_counts())
                entropyByNeighborCount[currentFrameNum] = fsr.shannon_entropy(count1)

                count2, _ = np.histogram(np.asarray(neighborCountWeightedList), binEdgesNeighborCountWeighted)
                entropyByNeighborCountWeighted[currentFrameNum] = fsr.shannon_entropy(count2)

                count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
                entropyByNeighborDistances[currentFrameNum] = fsr.shannon_entropy(count3)

                count4, _ = np.histogram(np.asarray(localDensitiesList), binEdgesLocalDensities)
                entropyByLocalDensities[currentFrameNum] = fsr.shannon_entropy(count4)

                neighborDistanceAvgAllRafts[currentFrameNum] = dfNeighbors['neighborDistanceAvg'].mean()
                neighborDistanceWeightedAvgAllRafts[currentFrameNum] = dfNeighbors['neighborDistanceWeightedAvg'].mean()

                dfNeighborsAllFrames = dfNeighborsAllFrames.append(dfNeighbors, ignore_index=True)

            dfNeighborsAllFrames = dfNeighborsAllFrames.infer_objects()
            dfNeighborsAllFrames = dfNeighborsAllFrames.sort_values(['frameNum', 'raftID'], ascending=[1, 1])

            # Temporal correlation of g6, g5, and g4, the unit of deltaT is frame
            temporalCorrHexaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype=complex)
            temporalCorrPentaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype=complex)
            temporalCorrTetraBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype=complex)
            temporalCorrHexaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype=complex)
            temporalCorrPentaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype=complex)
            temporalCorrTetraBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype=complex)

            for raftID in np.arange(numOfRafts):
                hexaOrdParaOfOneRaftSeries = dfNeighborsAllFrames.query(
                    'raftID == {}'.format(raftID)).hexaticOrderParameter
                pentaOrdParaOfOneRaftSeries = dfNeighborsAllFrames.query(
                    'raftID == {}'.format(raftID)).pentaticOrderParameter
                tetraOrdParaOfOneRaftSeries = dfNeighborsAllFrames.query(
                    'raftID == {}'.format(raftID)).tetraticOrderParameter

                hexaOrdParaOfOneRaftArray = np.array(hexaOrdParaOfOneRaftSeries.tolist())
                pentaOrdParaOfOneRaftArray = np.array(pentaOrdParaOfOneRaftSeries.tolist())
                tetraOrdParaOfOneRaftArray = np.array(tetraOrdParaOfOneRaftSeries.tolist())
                # construct the Toeplitz matrix, repeat input array twice to avoid the default conjugation 
                hexaOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(hexaOrdParaOfOneRaftArray,
                                                                           hexaOrdParaOfOneRaftArray)
                pentaOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(pentaOrdParaOfOneRaftArray,
                                                                            pentaOrdParaOfOneRaftArray)
                tetraOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(tetraOrdParaOfOneRaftArray,
                                                                            tetraOrdParaOfOneRaftArray)

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

                # multiply the two matrix so that for each column,
                # the rows on and below the diagonal are the products of
                # the conjugate of psi6(t0) and psi6(t0 + tStepSize), the tStepSize is the same the column index. 
                hexaOrdParaOfOneRaftBroadcastedTimesToeplitz = \
                    hexaOrdParaOfOneRaftArrayConjugateBroadcasted * hexaOrdParaOfOneRaftToeplitzMatrix
                pentaOrdParaOfOneRaftBroadcastedTimesToeplitz = \
                    pentaOrdParaOfOneRaftArrayConjugateBroadcasted * pentaOrdParaOfOneRaftToeplitzMatrix
                tetraOrdParaOfOneRaftBroadcastedTimesToeplitz = \
                    tetraOrdParaOfOneRaftArrayConjugateBroadcasted * tetraOrdParaOfOneRaftToeplitzMatrix

                for tStepSize in np.arange(numOfFrames):
                    temporalCorrHexaBondOrientationOrder[raftID, tStepSize] = \
                        np.average(hexaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:, tStepSize])
                    temporalCorrPentaBondOrientationOrder[raftID, tStepSize] = \
                        np.average(pentaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:, tStepSize])
                    temporalCorrTetraBondOrientationOrder[raftID, tStepSize] = \
                        np.average(tetraOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:, tStepSize])

            temporalCorrHexaBondOrientationOrderAvgAllRafts = temporalCorrHexaBondOrientationOrder.mean(axis=0)
            temporalCorrPentaBondOrientationOrderAvgAllRafts = temporalCorrPentaBondOrientationOrder.mean(axis=0)
            temporalCorrTetraBondOrientationOrderAvgAllRafts = temporalCorrTetraBondOrientationOrder.mean(axis=0)

            #  mutual information analysis
        if analysisType == 3 or analysisType == 4:
            # the duration for which the frames are sampled to calculate one MI
            widthOfInterval = 100  # unit: number of frames,

            numOfBins = 16

            # The gap between two successive MI calculation. 
            # Try keep (numOfFrames - widthOfInterval)//samplingGap an integer
            samplingGap = 50  # unit: number of frames

            numOfSamples = (numOfFrames - widthOfInterval) // samplingGap + 1
            sampleFrameNums = np.arange(widthOfInterval, numOfFrames, samplingGap)

            # pretreatment of position data
            raftOrbitingAnglesAdjusted = \
                fsr.adjust_orbiting_angles2(raftOrbitingAngles, orbiting_angles_diff_threshold=200)
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
            mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly = \
                np.trace(mutualInfoAllSamplesAllRafts, axis1=0, axis2=1) / numOfRafts
            mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI = \
                (mutualInfoAllSamplesAvgOverAllRafts * numOfRafts - mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly) / \
                (numOfRafts - 1)

            mutualInfoAvg = mutualInfoAllSamplesAvgOverAllRafts.mean(axis=0)
            mutualInfoAvgSelfMIOnly = mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly.mean(axis=0)
            mutualInfoAvgExcludingSelfMI = mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI.mean(axis=0)

        #  particle velocity and MSD analysis
        if analysisType == 5:
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
                raftVelocityXFiltered[raftID, :] = fsr.ssa_full(raftVelocityX[raftID, :], embeddingDimension,
                                                                reconstructionComponents)
                raftVelocityYFiltered[raftID, :] = fsr.ssa_full(raftVelocityY[raftID, :], embeddingDimension,
                                                                reconstructionComponents)

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

            # MSD analysis
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

        # save postprocessed data file
        tempShelf = shelve.open(shelveDataFileName)
        if analysisType == 1 or analysisType == 2 or analysisType == 4 or analysisType == 5:
            for key in listOfNewVariablesForClusterAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    # print('ERROR shelving: {0}'.format(key))
                    pass

        if analysisType == 2 or analysisType == 4 or analysisType == 5:
            for key in listOfNewVariablesForVoronoiAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    # print('ERROR shelving: {0}'.format(key))
                    pass

        if analysisType == 3 or analysisType == 4:
            for key in listOfNewVariablesForEntropyAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    # print('ERROR shelving: {0}'.format(key))
                    pass
        if analysisType == 5:
            for key in listOfNewVariablesForVelocityMSDAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    # print('ERROR shelving: {0}'.format(key))
                    pass
        tempShelf.close()

# %% extract data from all the post-processed files and store in dataframes for plotting

# dataFileListPostprocessedOnly = glob.glob('*postprocessed.dat')
# dataFileListPostprocessedOnly.sort()

summaryDataFrameColNames = ['batchNum', 'spinSpeed', 'commentsSub',
                            'magnification', 'radiusAvg', 'numOfFrames',
                            'frameWidth', 'frameHeight', 'deltaR',
                            'clusterSizeAvgIncludingLonersAllFrames',
                            'clusterSizeAvgExcludingLonersAllFrames',
                            'raftPairwiseEdgeEdgeDistancesSmallestMean',
                            'raftPairwiseEdgeEdgeDistancesSmallestStd',
                            'numOfLonersAvgAllFrames',
                            'entropyByNeighborCountAvgAllFrames',
                            'entropyByNeighborCountStdAllFrames',
                            'entropyByNeighborCountWeightedAvgAllFrames',
                            'entropyByNeighborCountWeightedStdAllFrames',
                            'entropyByNeighborDistancesAvgAllFrames',
                            'entropyByNeighborDistancesStdAllFrames',
                            'entropyByLocalDensitiesAvgAllFrames',
                            'entropyByLocalDensitiesStdAllFrames',
                            'neighborDistanceAvgAllRaftsAvgAllFrames',
                            'neighborDistanceAvgAllRaftsStdAllFrames',
                            'neighborDistanceWeightedAvgAllRaftsAvgAllFrames',
                            'neighborDistanceWeightedAvgAllRaftsStdAllFrames',
                            'hexaticOrderParameterAvgNormsAvgAllFrames',
                            'hexaticOrderParameterAvgNormsStdAllFrames',
                            'hexaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames',
                            'hexaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames',
                            'pentaticOrderParameterAvgNormsAvgAllFrames',
                            'pentaticOrderParameterAvgNormsStdAllFrames',
                            'pentaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames',
                            'pentaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames',
                            'tetraticOrderParameterAvgNormsAvgAllFrames',
                            'tetraticOrderParameterAvgNormsStdAllFrames',
                            'tetraticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames',
                            'tetraticOrderParameterModuliiAvgsAllRaftsStdsAllFrames',
                            'mutualInfoAvg0', 'mutualInfoAvg1',
                            'mutualInfoAvg2', 'mutualInfoAvg3',
                            'mutualInfoAvg4', 'mutualInfoAvg5',
                            'mutualInfoAvg6', 'mutualInfoAvg7',
                            'mutualInfoAvg8', 'mutualInfoAvg9',
                            'mutualInfoAvgExcludingSelfMI0',
                            'mutualInfoAvgExcludingSelfMI1',
                            'mutualInfoAvgExcludingSelfMI2',
                            'mutualInfoAvgExcludingSelfMI3',
                            'mutualInfoAvgExcludingSelfMI4',
                            'mutualInfoAvgExcludingSelfMI5',
                            'mutualInfoAvgExcludingSelfMI6',
                            'mutualInfoAvgExcludingSelfMI7',
                            'mutualInfoAvgExcludingSelfMI8',
                            'mutualInfoAvgExcludingSelfMI9',
                            'mutualInfoAvgSelfMIOnly0',
                            'mutualInfoAvgSelfMIOnly1',
                            'mutualInfoAvgSelfMIOnly2',
                            'mutualInfoAvgSelfMIOnly3',
                            'mutualInfoAvgSelfMIOnly4',
                            'mutualInfoAvgSelfMIOnly5',
                            'mutualInfoAvgSelfMIOnly6',
                            'mutualInfoAvgSelfMIOnly7',
                            'mutualInfoAvgSelfMIOnly8',
                            'mutualInfoAvgSelfMIOnly9',
                            'raftKineticEnergiesSumAllRaftsAvgAllFrames',
                            'raftKineticEnergiesSumAllRaftsStdAllFrames']

dfSummary = pd.DataFrame(columns=summaryDataFrameColNames)

dfRadialDistributionFunction = pd.DataFrame(columns=['distancesInRadius'])
dfSpatialCorrHexaOrderPara = pd.DataFrame(columns=['distancesInRadius'])
dfSpatialCorrPentaOrderPara = pd.DataFrame(columns=['distancesInRadius'])
dfSpatialCorrTetraOrderPara = pd.DataFrame(columns=['distancesInRadius'])
dfSpatialCorrHexaBondOrientationOrder = pd.DataFrame(columns=['distancesInRadius'])
dfSpatialCorrPentaBondOrientationOrder = pd.DataFrame(columns=['distancesInRadius'])
dfSpatialCorrTetraBondOrientationOrder = pd.DataFrame(columns=['distancesInRadius'])

dfTemporalCorrHexaBondOrientationOrderOfOneRaft = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfTemporalCorrPentaBondOrientationOrderOfOneRaft = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfTemporalCorrTetraBondOrientationOrderOfOneRaft = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = pd.DataFrame(columns=['timeDifferenceInFrames'])

dfHexaBondOrientatiotionOrderModuliiAvgTime = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfHexaBondOrientatiotionOrderAvgNormTime = pd.DataFrame(columns=['timeDifferenceInFrames'])

dfPentaBondOrientatiotionOrderModuliiAvgTime = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfPentaBondOrientatiotionOrderAvgNormTime = pd.DataFrame(columns=['timeDifferenceInFrames'])

dfTetraBondOrientatiotionOrderModuliiAvgTime = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfTetraBondOrientatiotionOrderAvgNormTime = pd.DataFrame(columns=['timeDifferenceInFrames'])

dfNeighborDistTime = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfEntropyNeighborDistTime = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfEntropyNeighborCountTime = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfEntropyLocalDensitiesTime = pd.DataFrame(columns=['timeDifferenceInFrames'])
dfClusterSizeIncludingLonersTime = pd.DataFrame(columns=['timeDifferenceInFrames'])

dfRaftVelocitiesVsOrbitingDistances = pd.DataFrame()

dfRaftXYAndMSD = pd.DataFrame(columns=['timeDifferenceInFrames'])

# for now, just fill the distancesInRadius column with radialRangeArray and
# timeDifferenceInFrames with timeDifferenceArray
deltaR = 1  # check this every time
radialRangeArray = np.arange(2, 100, deltaR)
dfRadialDistributionFunction['distancesInRadius'] = radialRangeArray
dfSpatialCorrHexaOrderPara['distancesInRadius'] = radialRangeArray
dfSpatialCorrPentaOrderPara['distancesInRadius'] = radialRangeArray
dfSpatialCorrTetraOrderPara['distancesInRadius'] = radialRangeArray
dfSpatialCorrHexaBondOrientationOrder['distancesInRadius'] = radialRangeArray
dfSpatialCorrPentaBondOrientationOrder['distancesInRadius'] = radialRangeArray
dfSpatialCorrTetraBondOrientationOrder['distancesInRadius'] = radialRangeArray

radiusInPixel = 23  # check this every time
scaleBar = 150 / radiusInPixel  # unit micron/pixel
numOfFrames = 500  # check this every time
frameRate = 75  # unit fps, check this every time
# timeDifferenceArray = np.arange(numOfFrames)
# dfTemporalCorrHexaBondOrientationOrderOfOneRaft['timeDifferenceInFrames'] = timeDifferenceArray
# dfTemporalCorrPentaBondOrientationOrderOfOneRaft['timeDifferenceInFrames'] = timeDifferenceArray
# dfTemporalCorrTetraBondOrientationOrderOfOneRaft['timeDifferenceInFrames'] = timeDifferenceArray
# dfTemporalCorrHexaBondOrientationOrderAvgAllRafts['timeDifferenceInFrames'] = timeDifferenceArray
# dfTemporalCorrPentaBondOrientationOrderAvgAllRafts['timeDifferenceInFrames'] = timeDifferenceArray
# dfTemporalCorrTetraBondOrientationOrderAvgAllRafts['timeDifferenceInFrames'] = timeDifferenceArray
# dfRaftXYAndMSD['timeDifferenceInFrames'] = timeDifferenceArray


frameNumToLookAt = 0
raftIDToLookAt = 0

analysisType = 5  # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi

for dataID in range(0, len(mainDataList)):
    # load data from main DataList: 
    dfSummary.loc[dataID, 'batchNum'] = mainDataList[dataID]['batchNum']
    dfSummary.loc[dataID, 'spinSpeed'] = mainDataList[dataID]['spinSpeed']
    dfSummary.loc[dataID, 'commentsSub'] = mainDataList[dataID]['commentsSub']
    dfSummary.loc[dataID, 'magnification'] = mainDataList[dataID]['magnification']
    dfSummary.loc[dataID, 'radiusAvg'] = mainDataList[dataID]['raftRadii'].mean()
    dfSummary.loc[dataID, 'numOfFrames'] = mainDataList[dataID]['numOfFrames']
    dfSummary.loc[dataID, 'frameWidth'] = mainDataList[dataID]['currentFrameGray'].shape[0]
    # from frame size, you can guess frame rate.
    dfSummary.loc[dataID, 'frameHeight'] = mainDataList[dataID]['currentFrameGray'].shape[1]
    # radiusInPixel = 23 # check this every time
    scaleBar = 150 / mainDataList[dataID]['raftRadii'].mean()  # unit micron/pixel

    # construct shelveName
    date = mainDataList[dataID]['date']
    numOfRafts = mainDataList[dataID]['numOfRafts']
    batchNum = mainDataList[dataID]['batchNum']
    spinSpeed = mainDataList[dataID]['spinSpeed']
    magnification = mainDataList[dataID]['magnification']
    shelveName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + \
                 str(magnification) + 'x_' + 'postprocessed' + str(analysisType)
    shelveDataFileName = shelveName + '.dat'

    if os.path.isfile(shelveDataFileName):
        tempShelf = shelve.open(shelveName)

        if analysisType == 1 or analysisType == 2 or analysisType == 4 or analysisType == 5:
            dfSummary.loc[dataID, 'clusterSizeAvgIncludingLonersAllFrames'] = tempShelf[
                'clusterSizeAvgIncludingLonersAllFrames']
            dfSummary.loc[dataID, 'clusterSizeAvgExcludingLonersAllFrames'] = tempShelf[
                'clusterSizeAvgExcludingLonersAllFrames']
            dfSummary.loc[dataID, 'raftPairwiseEdgeEdgeDistancesSmallestMean'] = tempShelf[
                'raftPairwiseEdgeEdgeDistancesSmallestMean']
            dfSummary.loc[dataID, 'raftPairwiseEdgeEdgeDistancesSmallestStd'] = tempShelf[
                'raftPairwiseEdgeEdgeDistancesSmallestStd']
            dfSummary.loc[dataID, 'numOfLonersAvgAllFrames'] = tempShelf['numOfLonersAvgAllFrames']

        if analysisType == 2 or analysisType == 4 or analysisType == 5:
            dfSummary.loc[dataID, 'entropyByNeighborCountAvgAllFrames'] = tempShelf['entropyByNeighborCount'].mean()
            dfSummary.loc[dataID, 'entropyByNeighborCountStdAllFrames'] = tempShelf['entropyByNeighborCount'].std()
            dfSummary.loc[dataID, 'entropyByNeighborCountWeightedAvgAllFrames'] = tempShelf[
                'entropyByNeighborCountWeighted'].mean()
            dfSummary.loc[dataID, 'entropyByNeighborCountWeightedStdAllFrames'] = tempShelf[
                'entropyByNeighborCountWeighted'].std()
            dfSummary.loc[dataID, 'entropyByNeighborDistancesAvgAllFrames'] = tempShelf[
                'entropyByNeighborDistances'].mean()
            dfSummary.loc[dataID, 'entropyByNeighborDistancesStdAllFrames'] = tempShelf[
                'entropyByNeighborDistances'].std()
            dfSummary.loc[dataID, 'entropyByLocalDensitiesAvgAllFrames'] = tempShelf['entropyByLocalDensities'].mean()
            dfSummary.loc[dataID, 'entropyByLocalDensitiesStdAllFrames'] = tempShelf['entropyByLocalDensities'].std()
            dfSummary.loc[dataID, 'neighborDistanceAvgAllRaftsAvgAllFrames'] = tempShelf[
                'neighborDistanceAvgAllRafts'].mean()
            dfSummary.loc[dataID, 'neighborDistanceAvgAllRaftsStdAllFrames'] = tempShelf[
                'neighborDistanceAvgAllRafts'].std()
            dfSummary.loc[dataID, 'neighborDistanceWeightedAvgAllRaftsAvgAllFrames'] = tempShelf[
                'neighborDistanceWeightedAvgAllRafts'].mean()
            dfSummary.loc[dataID, 'neighborDistanceWeightedAvgAllRaftsStdAllFrames'] = tempShelf[
                'neighborDistanceWeightedAvgAllRafts'].std()
            dfSummary.loc[dataID, 'deltaR'] = tempShelf['deltaR']
            dfSummary.loc[dataID, 'hexaticOrderParameterAvgNormsAvgAllFrames'] = tempShelf[
                'hexaticOrderParameterAvgNorms'].mean()
            dfSummary.loc[dataID, 'hexaticOrderParameterAvgNormsStdAllFrames'] = tempShelf[
                'hexaticOrderParameterAvgNorms'].std()
            dfSummary.loc[dataID, 'hexaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames'] = tempShelf[
                'hexaticOrderParameterModuliiAvgs'].mean()
            dfSummary.loc[dataID, 'hexaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames'] = tempShelf[
                'hexaticOrderParameterModuliiAvgs'].std()
            dfSummary.loc[dataID, 'pentaticOrderParameterAvgNormsAvgAllFrames'] = tempShelf[
                'pentaticOrderParameterAvgNorms'].mean()
            dfSummary.loc[dataID, 'pentaticOrderParameterAvgNormsStdAllFrames'] = tempShelf[
                'pentaticOrderParameterAvgNorms'].std()
            dfSummary.loc[dataID, 'pentaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames'] = tempShelf[
                'pentaticOrderParameterModuliiAvgs'].mean()
            dfSummary.loc[dataID, 'pentaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames'] = tempShelf[
                'pentaticOrderParameterModuliiAvgs'].std()
            dfSummary.loc[dataID, 'tetraticOrderParameterAvgNormsAvgAllFrames'] = tempShelf[
                'tetraticOrderParameterAvgNorms'].mean()
            dfSummary.loc[dataID, 'tetraticOrderParameterAvgNormsStdAllFrames'] = tempShelf[
                'tetraticOrderParameterAvgNorms'].std()
            dfSummary.loc[dataID, 'tetraticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames'] = tempShelf[
                'tetraticOrderParameterModuliiAvgs'].mean()
            dfSummary.loc[dataID, 'tetraticOrderParameterModuliiAvgsAllRaftsStdsAllFrames'] = tempShelf[
                'tetraticOrderParameterModuliiAvgs'].std()
            columnName = str(batchNum) + '_' + str(spinSpeed).zfill(4)
            dfRadialDistributionFunction[columnName] = tempShelf['radialDistributionFunction'][frameNumToLookAt, :]
            dfSpatialCorrHexaOrderPara[columnName] = tempShelf['spatialCorrHexaOrderPara'][frameNumToLookAt, :]
            dfSpatialCorrPentaOrderPara[columnName] = tempShelf['spatialCorrPentaOrderPara'][frameNumToLookAt, :]
            dfSpatialCorrTetraOrderPara[columnName] = tempShelf['spatialCorrTetraOrderPara'][frameNumToLookAt, :]
            dfSpatialCorrHexaBondOrientationOrder[columnName] = tempShelf['spatialCorrHexaBondOrientationOrder'][
                                                                frameNumToLookAt, :]
            dfSpatialCorrPentaBondOrientationOrder[columnName] = tempShelf['spatialCorrPentaBondOrientationOrder'][
                                                                 frameNumToLookAt, :]
            dfSpatialCorrTetraBondOrientationOrder[columnName] = tempShelf['spatialCorrTetraBondOrientationOrder'][
                                                                 frameNumToLookAt, :]
            dfTemporalCorrHexaBondOrientationOrderOfOneRaft = dfTemporalCorrHexaBondOrientationOrderOfOneRaft.join(
                pd.Series(np.real(tempShelf['temporalCorrHexaBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_real'), how='outer')
            dfTemporalCorrHexaBondOrientationOrderOfOneRaft = dfTemporalCorrHexaBondOrientationOrderOfOneRaft.join(
                pd.Series(np.imag(tempShelf['temporalCorrHexaBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_imag'), how='outer')
            dfTemporalCorrHexaBondOrientationOrderOfOneRaft = dfTemporalCorrHexaBondOrientationOrderOfOneRaft.join(
                pd.Series(np.absolute(tempShelf['temporalCorrHexaBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_abs'), how='outer')
            dfTemporalCorrPentaBondOrientationOrderOfOneRaft = dfTemporalCorrPentaBondOrientationOrderOfOneRaft.join(
                pd.Series(np.real(tempShelf['temporalCorrPentaBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_real'), how='outer')
            dfTemporalCorrPentaBondOrientationOrderOfOneRaft = dfTemporalCorrPentaBondOrientationOrderOfOneRaft.join(
                pd.Series(np.imag(tempShelf['temporalCorrPentaBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_imag'), how='outer')
            dfTemporalCorrPentaBondOrientationOrderOfOneRaft = dfTemporalCorrPentaBondOrientationOrderOfOneRaft.join(
                pd.Series(np.absolute(tempShelf['temporalCorrPentaBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_abs'), how='outer')
            dfTemporalCorrTetraBondOrientationOrderOfOneRaft = dfTemporalCorrTetraBondOrientationOrderOfOneRaft.join(
                pd.Series(np.real(tempShelf['temporalCorrTetraBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_real'), how='outer')
            dfTemporalCorrTetraBondOrientationOrderOfOneRaft = dfTemporalCorrTetraBondOrientationOrderOfOneRaft.join(
                pd.Series(np.imag(tempShelf['temporalCorrTetraBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_imag'), how='outer')
            dfTemporalCorrTetraBondOrientationOrderOfOneRaft = dfTemporalCorrTetraBondOrientationOrderOfOneRaft.join(
                pd.Series(np.absolute(tempShelf['temporalCorrTetraBondOrientationOrder'][raftIDToLookAt, :]),
                          name=columnName + '_abs'), how='outer')
            dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.join(
                pd.Series(np.real(tempShelf['temporalCorrHexaBondOrientationOrderAvgAllRafts']),
                          name=columnName + '_real'), how='outer')
            dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.join(
                pd.Series(np.imag(tempShelf['temporalCorrHexaBondOrientationOrderAvgAllRafts']),
                          name=columnName + '_imag'), how='outer')
            dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.join(
                pd.Series(np.absolute(tempShelf['temporalCorrHexaBondOrientationOrderAvgAllRafts']),
                          name=columnName + '_abs'), how='outer')
            dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = \
                dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.join(
                    pd.Series(np.real(tempShelf['temporalCorrPentaBondOrientationOrderAvgAllRafts']),
                              name=columnName + '_real'), how='outer')
            dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = \
                dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.join(
                    pd.Series(np.imag(tempShelf['temporalCorrPentaBondOrientationOrderAvgAllRafts']),
                              name=columnName + '_imag'), how='outer')
            dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = \
                dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.join(
                    pd.Series(np.absolute(tempShelf['temporalCorrPentaBondOrientationOrderAvgAllRafts']),
                              name=columnName + '_abs'), how='outer')
            dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = \
                dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.join(
                    pd.Series(np.real(tempShelf['temporalCorrTetraBondOrientationOrderAvgAllRafts']),
                              name=columnName + '_real'), how='outer')
            dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = \
                dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.join(
                    pd.Series(np.imag(tempShelf['temporalCorrTetraBondOrientationOrderAvgAllRafts']),
                              name=columnName + '_imag'), how='outer')
            dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = \
                dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.join(
                    pd.Series(np.absolute(tempShelf['temporalCorrTetraBondOrientationOrderAvgAllRafts']),
                              name=columnName + '_abs'), how='outer')
            dfHexaBondOrientatiotionOrderModuliiAvgTime = dfHexaBondOrientatiotionOrderModuliiAvgTime.join(
                pd.Series(tempShelf['hexaticOrderParameterModuliiAvgs'], name=columnName + '_time'), how='outer')
            dfHexaBondOrientatiotionOrderAvgNormTime = dfHexaBondOrientatiotionOrderAvgNormTime.join(
                pd.Series(tempShelf['hexaticOrderParameterAvgNorms'], name=columnName + '_time'), how='outer')
            dfPentaBondOrientatiotionOrderModuliiAvgTime = dfPentaBondOrientatiotionOrderModuliiAvgTime.join(
                pd.Series(tempShelf['pentaticOrderParameterModuliiAvgs'], name=columnName + '_time'), how='outer')
            dfPentaBondOrientatiotionOrderAvgNormTime = dfPentaBondOrientatiotionOrderAvgNormTime.join(
                pd.Series(tempShelf['pentaticOrderParameterAvgNorms'], name=columnName + '_time'), how='outer')
            dfTetraBondOrientatiotionOrderModuliiAvgTime = dfTetraBondOrientatiotionOrderModuliiAvgTime.join(
                pd.Series(tempShelf['tetraticOrderParameterModuliiAvgs'], name=columnName + '_time'), how='outer')
            dfTetraBondOrientatiotionOrderAvgNormTime = dfTetraBondOrientatiotionOrderAvgNormTime.join(
                pd.Series(tempShelf['tetraticOrderParameterAvgNorms'], name=columnName + '_time'), how='outer')
            dfNeighborDistTime = dfNeighborDistTime.join(
                pd.Series(tempShelf['neighborDistanceAvgAllRafts'], name=columnName + '_time'), how='outer')
            dfEntropyNeighborDistTime = dfEntropyNeighborDistTime.join(
                pd.Series(tempShelf['entropyByNeighborDistances'], name=columnName + '_time'), how='outer')
            dfEntropyNeighborCountTime = dfEntropyNeighborCountTime.join(
                pd.Series(tempShelf['entropyByNeighborCount'], name=columnName + '_time'), how='outer')
            dfEntropyLocalDensitiesTime = dfEntropyLocalDensitiesTime.join(
                pd.Series(tempShelf['entropyByLocalDensities'], name=columnName + '_time'), how='outer')
            dfClusterSizeIncludingLonersTime = dfClusterSizeIncludingLonersTime.join(
                pd.Series(tempShelf['clusterSizeAvgIncludingLoners'], name=columnName + '_time'), how='outer')

        if analysisType == 3 or analysisType == 4:
            mutualInfoAvg = tempShelf['mutualInfoAvg']
            mutualInfoAvgExcludingSelfMI = tempShelf['mutualInfoAvgExcludingSelfMI']
            mutualInfoAvgSelfMIOnly = tempShelf['mutualInfoAvgSelfMIOnly']

            for ii in range(10):
                dfSummary.loc[dataID, 'mutualInfoAvg' + str(ii)] = mutualInfoAvg[ii]
                dfSummary.loc[dataID, 'mutualInfoAvgExcludingSelfMI' + str(ii)] = mutualInfoAvgExcludingSelfMI[ii]
                dfSummary.loc[dataID, 'mutualInfoAvgSelfMIOnly' + str(ii)] = mutualInfoAvgSelfMIOnly[ii]

        if analysisType == 5:
            # velocity unit conversion: (pixel/frame) * (frameRate frame/second) * (scaleBar um/pixel) = um/sec
            dfSummary.loc[dataID, 'raftKineticEnergiesSumAllRaftsAvgAllFrames'] = \
                tempShelf['raftKineticEnergiesSumAllRafts'].mean() * (frameRate * scaleBar) ** 2
            dfSummary.loc[dataID, 'raftKineticEnergiesSumAllRaftsStdAllFrames'] = \
                tempShelf['raftKineticEnergiesSumAllRafts'].std() * (frameRate * scaleBar) ** 2
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(
                pd.Series(mainDataList[dataID]['raftOrbitingDistances'].flatten(order='F') * scaleBar,
                          name=columnName + '_orbitingDistances'), how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(
                pd.Series(tempShelf['raftVelocityXFiltered'].flatten(order='F') * (frameRate * scaleBar),
                          name=columnName + '_VelocityX'),
                how='outer')  # rafts in one frame upon rafts in another frame
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(
                pd.Series(tempShelf['raftVelocityYFiltered'].flatten(order='F') * (frameRate * scaleBar),
                          name=columnName + '_VelocityY'), how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(
                pd.Series(tempShelf['raftVelocityNormFiltered'].flatten(order='F') * (frameRate * scaleBar),
                          name=columnName + '_VelocityNorm'), how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(
                pd.Series(tempShelf['raftRadialVelocity'].flatten(order='F') * (frameRate * scaleBar),
                          name=columnName + '_RadialVelocity'), how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(
                pd.Series(tempShelf['raftTangentialVelocity'].flatten(order='F') * (frameRate * scaleBar),
                          name=columnName + '_TangentialVelocity'), how='outer')
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(
                pd.Series(mainDataList[dataID]['raftLocations'][raftIDToLookAt, :, 0] * scaleBar,
                          name=columnName + '_RaftX'), how='outer')
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(pd.Series((mainDataList[dataID]['currentFrameGray'].shape[1] -
                                                            mainDataList[dataID]['raftLocations'][raftIDToLookAt, :,
                                                            1]) * scaleBar, name=columnName + '_RaftY'),
                                                 how='outer')  # make the Y values from bottom to top
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(
                pd.Series(tempShelf['particleMSD'][raftIDToLookAt, :] * scaleBar ** 2, name=columnName + '_MSD'),
                how='outer')
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(
                pd.Series(tempShelf['particleMSDstd'][raftIDToLookAt, :] * scaleBar ** 2, name=columnName + '_MSDstd'),
                how='outer')

        tempShelf.close()
    else:
        print('missing data file: ' + shelveDataFileName)

dfSummaryConverted = dfSummary.infer_objects()
dfSummarySorted = dfSummaryConverted.sort_values(by=['batchNum', 'spinSpeed'], ascending=[True, False])

# csvColNames = ['batchNum','spinSpeed',
#               'clusterSizeAvgIncludingLonersAllFrames', 
#              'clusterSizeAvgExcludingLonersAllFrames',
#              'raftPairwiseEdgeEdgeDistancesSmallestMean',
#              'raftPairwiseEdgeEdgeDistancesSmallestStd',
#              'numOfLonersAvgAllFrames',
#              'entropyByNeighborCountAvgAllFrames',
#              'entropyByNeighborCountStdAllFrames',
#              'entropyByNeighborCountWeightedAvgAllFrames',
#              'entropyByNeighborCountWeightedStdAllFrames',
#              'entropyByNeighborDistancesAvgAllFrames',
#              'entropyByNeighborDistancesStdAllFrames',
#              'entropyByLocalDensitiesAvgAllFrames',
#              'entropyByLocalDensitiesStdAllFrames',
#              'neighborDistanceAvgAllRaftsAllFrames',
#              'neighborDistanceWeightedAvgAllRaftsAllFrames',
#              'hexaticOrderParameterAvgNormsAvgAllFrames',
#              'hexaticOrderParameterAvgNormsStdAllFrames',
#              'hexaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames',
#              'hexaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames',
#              'pentaticOrderParameterAvgNormsAvgAllFrames',
#              'pentaticOrderParameterAvgNormsStdAllFrames',
#              'pentaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames',
#              'pentaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames',
#              'tetraticOrderParameterAvgNormsAvgAllFrames',
#              'tetraticOrderParameterAvgNormsStdAllFrames',
#              'tetraticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames',
#              'tetraticOrderParameterModuliiAvgsAllRaftsStdsAllFrames',
#              'raftKineticEnergiesSumAllRaftsAvgAllFrames',
#              'raftKineticEnergiesSumAllRaftsStdAllFrames']

dataFileName = mainFolders[mainFolderID]
dfSummarySorted.to_csv(dataFileName + '_summary.csv', index=False, columns=summaryDataFrameColNames)

dfRadialDistributionFunction.to_csv(dataFileName + '_gr-frame{}.csv'.format(frameNumToLookAt), index=False)
dfSpatialCorrHexaOrderPara.to_csv(dataFileName + '_g6r-frame{}.csv'.format(frameNumToLookAt), index=False)
dfSpatialCorrPentaOrderPara.to_csv(dataFileName + '_g5r-frame{}.csv'.format(frameNumToLookAt), index=False)
dfSpatialCorrTetraOrderPara.to_csv(dataFileName + '_g4r-frame{}.csv'.format(frameNumToLookAt), index=False)
dfSpatialCorrHexaBondOrientationOrder.to_csv(dataFileName + '_g6r-over-gr-frame{}.csv'.format(frameNumToLookAt),
                                             index=False)
dfSpatialCorrPentaBondOrientationOrder.to_csv(dataFileName + '_g5r-over-gr-frame{}.csv'.format(frameNumToLookAt),
                                              index=False)
dfSpatialCorrTetraBondOrientationOrder.to_csv(dataFileName + '_g4r-over-gr-frame{}.csv'.format(frameNumToLookAt),
                                              index=False)

dfTemporalCorrHexaBondOrientationOrderOfOneRaft.to_csv(dataFileName + '_g6t-raft{}.csv'.format(raftIDToLookAt),
                                                       index=False)
dfTemporalCorrPentaBondOrientationOrderOfOneRaft.to_csv(dataFileName + '_g5t-raft{}.csv'.format(raftIDToLookAt),
                                                        index=False)
dfTemporalCorrTetraBondOrientationOrderOfOneRaft.to_csv(dataFileName + '_g4t-raft{}.csv'.format(raftIDToLookAt),
                                                        index=False)
dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.to_csv(dataFileName + '_g6t-avgAllRafts.csv', index=False)
dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.to_csv(dataFileName + '_g5t-avgAllRafts.csv', index=False)
dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.to_csv(dataFileName + '_g4t-avgAllRafts.csv', index=False)

dfRaftVelocitiesVsOrbitingDistances.to_csv(dataFileName + '_velocitiesVsOrbitingDistances.csv', index=False)
dfRaftXYAndMSD.to_csv(dataFileName + '_raftXYAndMSD-raft{}.csv'.format(raftIDToLookAt), index=False)

dfHexaBondOrientatiotionOrderModuliiAvgTime.to_csv(dataFileName + '_psi6ModuliiAvg-time.csv', index=False)
dfHexaBondOrientatiotionOrderAvgNormTime.to_csv(dataFileName + '_psi6AvgNorm-time.csv', index=False)
dfPentaBondOrientatiotionOrderModuliiAvgTime.to_csv(dataFileName + '_psi5ModuliiAvg-time.csv', index=False)
dfPentaBondOrientatiotionOrderAvgNormTime.to_csv(dataFileName + '_psi5AvgNorm-time.csv', index=False)
dfTetraBondOrientatiotionOrderModuliiAvgTime.to_csv(dataFileName + '_psi4ModuliiAvg-time.csv', index=False)
dfTetraBondOrientatiotionOrderAvgNormTime.to_csv(dataFileName + '_psi4AvgNorm-time.csv', index=False)

dfNeighborDistTime.to_csv(dataFileName + 'neighborDist-time.csv', index=False)
dfEntropyNeighborDistTime.to_csv(dataFileName + '_entropybyNeighborDist-time.csv', index=False)
dfEntropyNeighborCountTime.to_csv(dataFileName + '_entropybyNeighborCount-time.csv', index=False)
dfEntropyLocalDensitiesTime.to_csv(dataFileName + '_entropybyLocalDensities-time.csv', index=False)
dfClusterSizeIncludingLonersTime.to_csv(dataFileName + '_clustersizeincludingloners-time.csv', index=False)

# %% load data corresponding to a specific experiment (subfolder or video) into variables

dataID = 0

variableListFromProcessedFile = list(mainDataList[dataID].keys())

for key, value in mainDataList[dataID].items():  # loop through key-value pairs of python dictionary
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

# %%Kinetic Energy calculation (Gaurav)
raftlocx = raftLocations[:, ::5, 0]
raftlocy = raftLocations[:, ::5, 1]
raftvelx = np.gradient(raftlocx, axis=1)
raftvely = np.gradient(raftlocy, axis=1)
raftVelnormxy = np.square(raftvely) + np.square(raftvelx)
mean_vel = raftVelnormxy.mean(axis=0)
mean_vel_fps = mean_vel.mean()
energy_fps = mean_vel_fps + (raftRadii.mean() ** 2) * (2 * np.pi * spinSpeed * 5) ** 2

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

# %% loading a matlab data file

data = loadmat("2018-02-23_19Rafts_1_2500rpm_20x_Cam_20189_CineF5mat.mat")

numOfRafts = data['numOfRafts'][0][0]
numOfFrames = data['numOfFrames'][0][0]
raftRadii = data['raftRadii']
raftLocations = data['raftLocations']
raftOrbitingCenters = centers = np.mean(raftLocations, axis=0)
raftOrbitingDistances = np.zeros((numOfRafts, numOfFrames))
raftOrbitingAngles = np.zeros((numOfRafts, numOfFrames))

for currentFrameNum in np.arange(numOfFrames):
    for raftID in np.arange(numOfRafts):
        raftOrbitingDistances[raftID, currentFrameNum] = fsr.calculate_distance(
            raftOrbitingCenters[currentFrameNum, :], raftLocations[raftID, currentFrameNum, :])
        raftOrbitingAngles[raftID, currentFrameNum] = fsr.calculate_polar_angle(
            raftOrbitingCenters[currentFrameNum, :], raftLocations[raftID, currentFrameNum, :])

aa = np.array([3, 16, 49, 49, 21, 6, 1])
bb = np.array([498, 116, 116, 8, 4, 22, 2, 2, 4, 8, 8, 0, 2, 0, 2, 24])
Eaa = fsr.shannon_entropy(aa)
Ebb = fsr.shannon_entropy(bb)
