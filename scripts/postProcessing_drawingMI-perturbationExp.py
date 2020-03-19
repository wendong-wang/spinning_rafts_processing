# -*- coding: utf-8 -*-
"""
 - Draw mutual information values as color-dots for perturbation experiments
"""

import os, glob, shelve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



sampleNumForRaftNumChange = 1000 #1 raft captured: 75; 2 rafts captured: 95

numOfRaftChanged = 0 # 0, -1, -2

# folder name for processed and postprocessed data
#folderName = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\batch3_1RaftCaptured\processed_postprocessed_interval50-gap25-bin8'
#folderName = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\batch4_2RaftsCaptured\processed_postprocessed_interval50_gap25_bin8'
folderName = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\batch1_noRaftCaptured\processed_postprocessed_interval50_gap25_bin8'
#folderName = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\batch2_sideMotion\processed_postprocessed_interval50_gap25_bin8'

# folder name for the raw tiff file
#tiffFileFolder = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\54Rafts_3_60rps_0.57x_2ms-500fps_copperWireAsPredator_OneRaftKIckedout'
#tiffFileFolder = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\53Rafts_4_60rps_0.57x_2ms-500fps_copperWireAsPredator_TwoRaftsKIckedout'
tiffFileFolder = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\55Rafts_1_60rps_0.57x_2ms-500fps_copperWireAsPredator_noRaftsKickedout'
#tiffFileFolder = r'E:\Data_PhantomMiroLab140\2019-05-14c_o-D300_14mT_copperWireAsPredator_processed\55Rafts_2_60rps_0.57x_2ms-500fps_copperWireAsPredator_side-motion'

#%% load processed data files


os.chdir(folderName)

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

# select the right data file
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

for key, value in mainDataList[dataID].items(): # loop through key-value pairs of python dictionary
    globals()[key] = value
    
    
#%% load postprocessed data files
    
analysisType = 4 # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi

# the undefined variables for the shelve data file name are loaded from the processed file
shelveDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' \
                     + str(magnification) + 'x_' + 'postprocessed' + str(analysisType)

shelveDataFileExist = glob.glob(shelveDataFileName+'.dat')

if shelveDataFileExist:
    print(shelveDataFileName + ' exists, load additional variables. ' )
    tempShelf = shelve.open(shelveDataFileName)
    variableListFromPostProcessedFile = list(tempShelf.keys())

    mutualInfoAllSamplesAllRafts = tempShelf["mutualInfoAllSamplesAllRafts"]
    numOfSamples = tempShelf['numOfSamples']
    samplingGap = tempShelf['samplingGap']

    for key in tempShelf:
        # just loop through all the keys in the dictionary
        if not (key in globals()):
            globals()[key] = tempShelf[key]
    
    tempShelf.close()
    print('loading complete.')
    
elif len(shelveDataFileExist) == 0:
    print(shelveDataFileName + ' does not exist')

#%% prepare mutual information data for plotting

# the variable mutualInfoAllSamplesAllRafts is loaded from the postprocessed data file
# it has dimentions (numOfRafts, numOfRafts, numOfSamples, 10)
# for the last dimension,

# 0 - neighbor distances smallest; 1 - neighbor distances mean, 2 - neighbor distances max
mutualInfoSelected = mutualInfoAllSamplesAllRafts[:, :, :, 0].copy()

#numOfRafts, numOfRafts, numOfSamples = mutualInfoSelected.shape

raftMIAvgOverPairs = np.empty((numOfSamples, numOfRafts))
# numOfSamples and numOfRafts are read from processed and postprocessed data

for t in range(numOfSamples):
    mutualInfoInOneSample = mutualInfoSelected[:,:,t]
    mutualInfoInOneSampleSummedExcludingSelf = mutualInfoInOneSample.sum(axis=0) - mutualInfoInOneSample.diagonal()
    if t < sampleNumForRaftNumChange :
        raftMIAvgOverPairs[t, :] = mutualInfoInOneSampleSummedExcludingSelf/numOfRafts
    else :
        raftMIAvgOverPairs[t, :] = mutualInfoInOneSampleSummedExcludingSelf/(numOfRafts + numOfRaftChanged)


#%% drawing and output data


plt.ioff()
#plt.ion()
plt.style.use('dark_background')

cmap = plt.cm.get_cmap('inferno')
norm = plt.cm.colors.PowerNorm(0.4, vmin = raftMIAvgOverPairs.min(), vmax = raftMIAvgOverPairs.max())


for i,time_steps in enumerate(range(0,numOfSamples)):
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(18,9))
    ax = axes.flatten()
    ts=int(time_steps * samplingGap) # samplingGap is read from the postprocessed file

    img1_name= os.path.join(tiffFileFolder, '2019-05-14_copperWireAsPredator.'+str(ts + 1).zfill(5)+'.tiff')

    img1=plt.imread(img1_name)

    _, height, _ = img1.shape

    ax[0].imshow(img1[::-1, :, :], origin='lower')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set(aspect='equal', adjustable='box')

    raftl = raftLocations[:,ts,:] # raftLocations is read from the processed file
    if time_steps < sampleNumForRaftNumChange :
        im = ax[1].scatter(raftl[:,0], height - raftl[:,1],s=raftMIAvgOverPairs[time_steps,:]*150,
                           c=raftMIAvgOverPairs[time_steps,:], marker='o', norm=mpl.colors.Normalize(),
                           vmin=0, vmax=1.7, alpha=1.0,)
    else:
        im = ax[1].scatter(raftl[:numOfRaftChanged, 0], height - raftl[:numOfRaftChanged, 1],
                           s=raftMIAvgOverPairs[time_steps,:numOfRaftChanged]*150,
                           c=raftMIAvgOverPairs[time_steps,:numOfRaftChanged], marker='o',
                           norm=mpl.colors.Normalize(), vmin=0, vmax=1.7, alpha=1.0,)

    ax[1].set_xticks([])
    ax[1].set_yticks([])
#    ax[1].set(aspect='equal')

    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.subplots_adjust(wspace = 0.01)

    fig.savefig('2019-05-14_55Rafts_1_60.0rps_'+str(time_steps + 1).zfill(4) + '.png',dpi=100)
    fig.clf()
    plt.close('all')

plt.ion()