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
- clucster analysis
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
#%% import libraries and define functions

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import cv2 as cv
## if mpl.is_interactive() == True, plt.ioff() 
## other useful ones: plt.close('all'); 

from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from scipy.io import loadmat
from sklearn.metrics import mutual_info_score
from scipy.spatial import distance as scipyDistance
from scipy.spatial import Voronoi as scipyVoronoi
from scipy.spatial import voronoi_plot_2d as scipyVoronoiPlot2D

import progressbar
import os, glob
import shelve

# for singular spectrum analysis
from sklearn.decomposition import PCA as sklearnPCA
import scipy.linalg as linalg
import scipy.stats as stats



def calculateCentersOfMass (xAll,yAll, raftNum = 1):
    """calculate the centers of all rafts for each frame
    
    xAll - x position, (# of frames, # of rafts), unit: pixel
    yAll - y position (# of frames, # of rafts)
    """
    numOfFrames, numOfRafts = xAll.shape
    
    xCenters = xAll[:,0:numOfRafts].mean(axis = 1)
    yCenters = yAll[:,0:numOfRafts].mean(axis = 1)

    xRelativeToCenters = xAll - xCenters[:,np.newaxis]
    yRelativeToCenters = yAll - yCenters[:,np.newaxis]

    distancesToCenters = np.sqrt(xRelativeToCenters**2 + yRelativeToCenters**2)

    orbitingAngles = np.arctan2(yRelativeToCenters, xRelativeToCenters) * 180 / np.pi

    return distancesToCenters, orbitingAngles, xCenters, yCenters

def CalculateDistance(p1, p2):
    ''' calculate the distance between p1 and p2
    '''
    
    dist = np.sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)
    
    return dist

def CalculatePolarAngle(p1, p2):
    ''' calculate the polar angle of the vector from p1 to p2. 
    '''
    
    # note the negative sign before the first component, which is y component 
    # the y in scikit-image is flipped. 
    # it is to make the value of the angle appears natural, as in Rhino, with x-axis pointing right, and y-axis pointing up. 
    # the range is from -pi to pi
    angle = np.arctan2(-(p2[1] - p1[1]), (p2[0] - p1[0])) *180 / np.pi
    
    return angle 

def AdjustOrbitingAngles(orbiting_angles_series, orbiting_angles_diff_threshold = 200):
    ''' adjust the orbiting angles to get rid of the jump of 360 when it crosses from -180 to 180, or the reverse
        adjust single point anormaly. 
    '''

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

def AdjustOrbitingAngles2(orbiting_angles_series, orbiting_angles_diff_threshold = 200):
    ''' 2d version of AjustOrbitingAngles
        adjust the orbiting angles to get rid of the jump of 360 
        when it crosses from -180 to 180, or the reverse
        orbiting_angle_series has the shape (raft num, frame num)  
    '''

    orbiting_angles_diff = np.diff(orbiting_angles_series, axis = 1)
    
    index_neg = orbiting_angles_diff < -orbiting_angles_diff_threshold
    index_pos = orbiting_angles_diff > orbiting_angles_diff_threshold
    
    insertion_indices_neg = np.nonzero(index_neg)
    insertion_indices_pos = np.nonzero(index_pos)
    
    
    orbiting_angles_diff_corrected = orbiting_angles_diff.copy()
    orbiting_angles_diff_corrected[insertion_indices_neg[0], insertion_indices_neg[1]] += 360
    orbiting_angles_diff_corrected[insertion_indices_pos[0], insertion_indices_pos[1]] -= 360
    
    orbiting_angles_corrected = orbiting_angles_series.copy()
    orbiting_angles_corrected[:,1:] = orbiting_angles_diff_corrected[:]
    orbiting_angles_adjusted = np.cumsum(orbiting_angles_corrected, axis = 1)
    
    return orbiting_angles_adjusted

def MutualInfoMatrix(time_series, num_of_bins):
    """
    Calculate mutual information for each pair of rafts
    
    time_series - rows are raft numbers, and columns are times
    numOfBins- numOfBins for calculating histogram
    the result is in unit of bits. 
    """
    num_of_rafts, interval_width = time_series.shape
    mutual_info_matrix = np.zeros((num_of_rafts,num_of_rafts))
    
    for i in range(num_of_rafts):
        for j in range(i+1):
            i0 = time_series[i,:].copy()
            j0 = time_series[j,:].copy()
            c_xy = np.histogram2d(i0, j0, num_of_bins)[0]
            mi = mutual_info_score(None, None, contingency=c_xy)* np.log2(np.e) # in unit of bits,  * np.log2(np.e) to convert nats to bits
            mutual_info_matrix[i,j] = mi
            mutual_info_matrix[j,i] = mi
                
    return mutual_info_matrix

def ShannonEntropy(c):
    """calculate the Shannon entropy of 1 d data. The unit is bits """
    
    c_normalized = c / float(np.sum(c))
    c_normalized_nonzero = c_normalized[np.nonzero(c_normalized)] # gives 1D array
    H = -sum(c_normalized_nonzero* np.log2(c_normalized_nonzero))  # unit in bits
    return H

def FFTDistances(sampling_rate, distances):
    ''' given sampling rate and distances, and output frequency vector and one-sided power spectrum
        sampling_rate: unit Hz
        distances: numpy array, unit micron
    '''
#    sampling_interval = 1/sampling_rate # unit s
#    times = np.linspace(0,sampling_length*sampling_interval, sampling_length)
    sampling_length = len(distances) # total number of frames
    fft_distances = np.fft.fft(distances)
    P2 = np.abs(fft_distances/sampling_length)
    P1 = P2[0:int(sampling_length/2)+1]
    P1[1:-1] = 2*P1[1:-1] # one-sided powr spectrum
    frequencies = sampling_rate/sampling_length * np.arange(0,int(sampling_length/2)+1)
    
    return frequencies, P1
 
def DrawRafts(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    ''' draw circles around rafts
    '''
    
    circle_thickness = int(2)
    circle_color = (0,0,255) # openCV: BGR

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id,0], rafts_loc[raft_id,1]), rafts_radii[raft_id], circle_color, circle_thickness)
    
    return output_img
 
def DrawRaftOrientations(img_bgr, rafts_loc, rafts_ori, num_of_rafts):
    ''' draw lines to indicte the orientation of each raft
    '''

    line_thickness = int(2)
    line_color = (255,0,0)
    line_length = 50
        
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id,0], rafts_loc[raft_id,1])
        line_end = (int(rafts_loc[raft_id,0] + np.cos(rafts_ori[raft_id]*np.pi/180)*line_length), int(rafts_loc[raft_id,1] - np.sin(rafts_ori[raft_id]*np.pi/180)*line_length))
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)
    
    return output_img

def DrawRaftNumber(img_bgr, rafts_loc, num_of_rafts):
    ''' draw the raft number at the center of the rafts
    '''
    
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0,255,255) # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id+1),fontFace, font_scale, font_thickness)
        output_img = cv.putText(output_img,str(raft_id+1),(rafts_loc[raft_id,0] - textSize[0]//2, rafts_loc[raft_id,1] + textSize[1]//2), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
        
    return output_img

def DrawClusters(img_bgr, connectivity_matrix, rafts_loc):
    ''' draw lines between centers of connected rafts
    '''
    line_thickness = 2
    line_color = (0,255,0)
    output_img = img_bgr
    raftAs, raftBs = np.nonzero(connectivity_matrix)
    
    for raftA, raftB in zip(raftAs, raftBs):
        output_img = cv.line(output_img, (rafts_loc[raftA,0], rafts_loc[raftA,1]), (rafts_loc[raftB,0], rafts_loc[raftB,1]), line_color, line_thickness)
 
    return output_img

def DrawVoronoi(img_bgr, rafts_loc):
    ''' draw Voronoi patterns
    '''
    points = rafts_loc
    vor = scipyVoronoi(points)
    output_img = img_bgr
    # drawing Voronoi vertices
    vertex_size = int(3)
    vertex_color = (255,0,0)
    for x, y in zip(vor.vertices[:,0], vor.vertices[:,1]):
        output_img = cv.circle(output_img, (int(x), int(y)), vertex_size, vertex_color)
    
    # drawing Voronoi edges
    edge_color = (0, 255, 0)
    edge_thickness = int(2)
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            output_img = cv.line(output_img, (int(vor.vertices[simplex[0], 0]), int(vor.vertices[simplex[0], 1])), (int(vor.vertices[simplex[1], 0]), int(vor.vertices[simplex[1], 1])), edge_color, edge_thickness)

    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
            output_img = cv.line(output_img, (int(vor.vertices[i,0]), int(vor.vertices[i,1])), (int(far_point[0]), int(far_point[1])), edge_color, edge_thickness)
    return output_img

def DrawAtBottomLeftOfRaftNumberFloat(img_bgr, rafts_loc, neighbor_count_wt, num_of_rafts):
    ''' write a subscript to indicate nearest neighbor count or weighted nearest neighbor count
    '''
    fontFace = cv.FONT_ITALIC
    font_scale = 0.5
    font_color = (0,165,255) # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id+1),fontFace, font_scale, font_thickness)
        output_img = cv.putText(output_img,'{:.2}'.format(neighbor_count_wt[raft_id]),(rafts_loc[raft_id,0] + textSize[0]//2, rafts_loc[raft_id,1] + textSize[1]), fontFace, font_scale, font_color, font_thickness, cv.LINE_AA)
        
    return output_img

def DrawAtBottomLeftOfRaftNumberInteger(img_bgr, rafts_loc, neighbor_count_wt, num_of_rafts):
    ''' write a subscript to indicate nearest neighbor count or weighted nearest neighbor count
    '''
    fontFace = cv.FONT_ITALIC
    font_scale = 0.5
    font_color = (0,165,255) # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id+1),fontFace, font_scale, font_thickness)
        output_img = cv.putText(output_img,'{:}'.format(neighbor_count_wt[raft_id]),(rafts_loc[raft_id,0] + textSize[0]//2, rafts_loc[raft_id,1] + textSize[1]), fontFace, font_scale, font_color, font_thickness, cv.LINE_AA)
        
    return output_img

def DrawNeighborCounts(img_bgr, rafts_loc, num_of_rafts):
    ''' draw the raft number at the center of the rafts
    '''
    points = rafts_loc
    vor = scipyVoronoi(points)
    neighborCounts = np.zeros(numOfRafts, dtype=int)
    for raftID in range(numOfRafts):
        neighborCounts[raftID] = np.count_nonzero(vor.ridge_points.ravel() == raftID)
    
    fontFace = cv.FONT_ITALIC
    font_scale = 0.5
    font_color = (0,165,255) # BGR
    font_thickness = 1
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id+1),fontFace, font_scale, font_thickness)
        output_img = cv.putText(output_img,str(neighborCounts[raft_id]),(rafts_loc[raft_id,0] + textSize[0]//2, rafts_loc[raft_id,1] + textSize[1]), fontFace, font_scale, font_color, font_thickness, cv.LINE_AA)
        
    return output_img


def PolygonArea(x,y):
    ''' calculate the area of a polygon given the x and y coordinates of vertices
    ref: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    '''
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def SSADecompose(y, dim):
    """
    from Vimal
    Singular Spectrum Analysis decomposition for a time series
    Example:
    -------
    >>> import numpy as np
    >>>
    >>> x = np.linspace(0, 5, 1000)
    >>> y = 2*x + 2*np.sin(5*x) + 0.5*np.random.randn(1000)
    >>> pc, s, v = ssa(y, 15)
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

def SSAReconstruct(pc, v, k):
    """
    from Vimal
    Series reconstruction for given SSA decomposition using vector of components
    Example:
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> y = 2*x + 2*np.sin(5*x) + 0.5*np.random.randn(1000)
    >>> pc, s, v = ssa(y, 15)
    >>>
    >>> yr = inv_ssa(pc, v, [0,1])
    >>> plt.plot(x, yr)
    :param pc: matrix with the principal components from SSA
    :param v: matrix of the singular vectors from SSA
    :param k: vector with the indices of the components to be reconstructed
    :return: the reconstructed time series
    """
    if np.isscalar(k): k = [k]

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
        xr[i : t + i] = xr[i : t + i] + pc_comp[:, i]
        times[i : t + i] = times[i : t + i] + 1

    xr = (xr / times) * np.sqrt(t)
    return xr

def SSAFull(time_series, embedding_dim = 20, reconstruct_components = np.arange(10)):
    """
    combine SSA decomposition and reconstruction together
    """
    
    pc, s, v = SSADecompose(time_series, embedding_dim)
    time_series_reconstructed = SSAReconstruct(pc, v, reconstruct_components)
    
    return time_series_reconstructed


rootFolderNameFromWindows = r'D:\\VideoProcessingFolder' #r'E:\Data_Camera_Basler_acA800-510uc_coilSystem'
#rootFolderNameFromWindows = '/media/gardi/Seagate Backup Plus Drive/Data_Camera_Basler_acA800-510uc_coilSystem'
#rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc'   
#rootFolderNameFromWindows = '/media/gardi/Elements/Data_basler'
#rootFolderNameFromWindows = r'E:\Data_Camera_Basler-acA2500-60uc'
#rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc/2018-10-09_o-D300-sym4-amp2-arcAngle30-Batch21Sep2018_Co500Au60_14mT_tiling_to be analyzed/processed'
#rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc/2018-10-09_o-D300-sym4-amp2-arcAngle30-Batch21Sep2018_Co500Au60_14mT_tiling_to be analyzed/processed/processed'
#rootFolderNameFromWindows = '/media/gardi/Elements/Data_Camera_Basler-acA2500-60uc/2018-10-09_o-D300-sym4-amp2-arcAngle30-Batch21Sep2018_Co500Au60_14mT_tiling_to be analyzed'
#rootFolderNameFromWindows =  '/media/gardi/Elements/Data_basler'
#rootFolderNameFromWindows = '/media/gardi/MPI-11/Data_basler'
#rootFolderNameFromWindows = '/media/gardi/Elements/Data_PhantomMiroLab140'
#rootFolderNameFromWindows = '/home/gardi/Rafts/Experiments Data/Data_PhantomMiroLab140'
#rootFolderNameFromWindows = '/media/gardi/MPI-Data9/Data_Basler-ace2500-60uc_coilsystem'
os.chdir(rootFolderNameFromWindows)
rootFolderTreeGen = os.walk(rootFolderNameFromWindows)
_, mainFolders, _ = next(rootFolderTreeGen) 


#%% loading all the data in a specific main folder into mainDataList
# at the moment, it handles one main folder at a time. 

#for mainFolderID in np.arange(0,1):
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


#%% looping to plot the center of mass and check if the data needs to be re-analyzed, for all subfolders
for dataID in range(0,len(mainDataList)):
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
    outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_' + commentsSub 
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    ax.plot(raftOrbitingCenters[:,0], currentFrameGray.shape[1] - raftOrbitingCenters[:,1])
    fig.savefig(outputDataFileName + '_COM.png')

plt.close('all')
#%% post-process data (cluster analysis, Voronoi analysis, and mutual information analysis)

analysisType = 2 # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi

listOfNewVariablesForClusterAnalysis = ['raftPairwiseDistances', 'raftPairwiseDistancesInRadius',
                                        'raftPairwiseEdgeEdgeDistancesSmallest','scaleBar',
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
                                        'samplingWindowSizes',
                                        'entropyByNeighborCountInWindows',
                                        'entropyByNeighborCountWeightedInWindows',
                                        'entropyByNeighborDistancesInWindows',
                                        'entropyByLocalDensitiesInWindows',
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
for dataID in range(1,len(dataFileListExcludingPostProcessed)):
    ######################### load variables from mainDataList
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
    
#    outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_' + commentsSub 

    shelveDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_' + 'postprocessed' + str(analysisType)
    
    shelveDataFileExist = glob.glob(shelveDataFileName+'.dat') # empty list is false
    
    if not shelveDataFileExist:
        ############################### cluster analysis
        if analysisType == 1 or analysisType == 2 or analysisType == 4 or analysisType == 5:
            radius = raftRadii.mean() #pixel  check raftRadii.mean()
            scaleBar = 300 / radius /2 # micron per pixel
            
            raftPairwiseDistances = np.zeros((numOfRafts, numOfRafts, numOfFrames))
            raftPairwiseEdgeEdgeDistancesSmallest = np.zeros((numOfRafts, numOfFrames))
            raftPairwiseDistancesInRadius = np.zeros((numOfRafts, numOfRafts, numOfFrames))
            raftPairwiseConnectivity = np.zeros((numOfRafts, numOfRafts, numOfFrames))
            
            # using scipy distance module
            for frameNum in np.arange(numOfFrames):
                raftPairwiseDistances[:,:,frameNum] = scipyDistance.cdist(raftLocations[:,frameNum,:], raftLocations[:,frameNum,:], 'euclidean')
                # smallest nonzero eedistances is assigned to one raft as the pairwise distance, connected rafts will be set to 0 later
                raftPairwiseEdgeEdgeDistancesSmallest[:,frameNum] = np.partition(raftPairwiseDistances[:,:,frameNum], 1, axis = 1)[:,1] - radius *2
    
            raftPairwiseDistancesInRadius = raftPairwiseDistances / radius
            
            # Caution: this way of determing clusters produces errors, mostly false positive. 
            connectivityThreshold = 2.3 # unit: radius
    
            # Note that the diagonal self-distance is zero, and needs to be taken care of seperately
            raftPairwiseConnectivity = np.logical_and((raftPairwiseDistancesInRadius < connectivityThreshold), (raftPairwiseDistancesInRadius > 0)) *1
            
            # to correct false positive, if the rafts are not connected in the next frame, 
            # then it is not connected in the present frame
            for currentFrameNum in range(numOfFrames-1):
                raftAs, raftBs = np.nonzero(raftPairwiseConnectivity[:,:,currentFrameNum])
                for raftA, raftB in zip(raftAs, raftBs):
                    if raftPairwiseConnectivity[raftA, raftB,currentFrameNum+1] == 0:
                        raftPairwiseConnectivity[raftA, raftB,currentFrameNum] = 0
      
        
            # information about clusters in all frames. For reach frame, the array has two columns, 
            # 1st col: cluster number, 2nd col: cluster size (excluding loners)
            clusters = np.zeros((numOfRafts, 2, numOfFrames))
            # clusterSizeCounts stores the number of clusters of each size for all frames. 
            # the first index is used as the size of the cluster
            clusterSizeCounts = np.zeros((numOfRafts+1,numOfFrames)) 
            
            # fill in clusters array
            for frameNum in np.arange(numOfFrames):
                clusterNum = 1
                raftAs, raftBs = np.nonzero(raftPairwiseConnectivity[:,:,frameNum])
                # determine the cluster number and store the cluster number in the first column
                for raftA, raftB in zip(raftAs, raftBs):
                    # to see if A and B are already registered in the raftsInClusters
                    raftsInClusters = np.nonzero(clusters[:,0,frameNum])
                    A = any(raftA in raft for raft  in raftsInClusters)
                    B = any(raftB in raft for raft  in raftsInClusters)
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
                    # if neigher is new and if their cluster numbers differ, then change the larger cluster number to the smaller one
                    # note that this could lead to a cluster number being jumped over
                    if (A == True) and (B == True) and (clusters[raftA, 0, frameNum] != clusters[raftB, 0, frameNum]):
                        clusterNumLarge = max(clusters[raftA,0, frameNum], clusters[raftB,0, frameNum])
                        clusterNumSmall = min(clusters[raftA,0, frameNum], clusters[raftB,0, frameNum])
                        clusters[clusters[:,0, frameNum] == clusterNumLarge,0, frameNum] = clusterNumSmall
                # Count the number of rafts in each cluster and store the cluster size in the second column
                numOfClusters = clusters[:,0,frameNum].max()
                if numOfClusters > 0:
                    for clusterNum in np.arange(1, numOfClusters+1):
                        clusterSize = len(clusters[clusters[:,0,frameNum] == clusterNum,0,frameNum])
                        clusters[clusters[:,0,frameNum] == clusterNum,1,frameNum] = clusterSize
                raftPairwiseEdgeEdgeDistancesSmallest[np.nonzero(clusters[:,0,frameNum]),frameNum] = 0
    
            # fill in clusterSizeCounts array        
            for frameNum in np.arange(numOfFrames):
                largestClusterSize = clusters[:,1,frameNum].max()
                # count loners
                numOfLoners = len(clusters[clusters[:,1,frameNum] == 0,1,frameNum])
                clusterSizeCounts[1,frameNum] = numOfLoners
                # for the rest, the number of occurrence of cluster size in the 2nd column is the cluster size times the number of clusters of that size
                for clusterSize in np.arange(2, largestClusterSize+1):
                    numOfClusters = len(clusters[clusters[:,1,frameNum] == clusterSize,1,frameNum])/clusterSize
                    clusterSizeCounts[int(clusterSize),frameNum] = numOfClusters
            
            # some averageing
            dummyArray = np.arange((numOfRafts + 1) * numOfFrames).reshape((numOfFrames,-1)).T
            dummyArray = np.mod(dummyArray, (numOfRafts + 1)) # rows are cluster sizes, and columns are frame numbers
            clusterSizeAvgIncludingLoners = np.average(dummyArray, axis = 0, weights = clusterSizeCounts)
            clusterSizeAvgIncludingLonersAllFrames = clusterSizeAvgIncludingLoners.mean()
            
            clusterSizeCountsExcludingLoners = clusterSizeCounts.copy()
            clusterSizeCountsExcludingLoners[1,:] = 0
            clusterSizeAvgExcludingLoners, sumOfWeights = np.ma.average(dummyArray, axis = 0, weights = clusterSizeCountsExcludingLoners, returned = True)
            clusterSizeAvgExcludingLonersAllFrames = clusterSizeAvgExcludingLoners.mean()
            
            raftPairwiseEdgeEdgeDistancesSmallestMean = raftPairwiseEdgeEdgeDistancesSmallest.mean() * scaleBar
            raftPairwiseEdgeEdgeDistancesSmallestStd = raftPairwiseEdgeEdgeDistancesSmallest.std() * scaleBar
            numOfLonersAvgAllFrames = clusterSizeCounts[1,:].mean()
        
        ########################### voronoi analysis
        if analysisType == 2 or analysisType == 4 or analysisType == 5:
            deltaR = 1
            sizeOfArenaInRadius = 15000/150 # 1.5cm square arena, 150 um raft radius
            radialRangeArray = np.arange(2, 100, deltaR)
            
            samplingWindowStep = 2
            samplingWindowSizes = np.arange(5, np.floor(sizeOfArenaInRadius/2)*1.5, samplingWindowStep)
            samplingWindowCount = len(samplingWindowSizes)
            
            entropyByNeighborCount = np.zeros(numOfFrames)
            entropyByNeighborCountWeighted = np.zeros(numOfFrames)
            entropyByNeighborDistances = np.zeros(numOfFrames)
            entropyByLocalDensities = np.zeros(numOfFrames)
            neighborDistanceAvgAllRafts = np.zeros(numOfFrames)
            neighborDistanceWeightedAvgAllRafts = np.zeros(numOfFrames)
            
            entropyByNeighborCountInWindows = np.zeros((numOfFrames, samplingWindowCount))
            entropyByNeighborCountWeightedInWindows = np.zeros((numOfFrames, samplingWindowCount))
            entropyByNeighborDistancesInWindows = np.zeros((numOfFrames, samplingWindowCount))
            entropyByLocalDensitiesInWindows = np.zeros((numOfFrames, samplingWindowCount))
            
            binEdgesNeighborCountWeighted = np.arange(1, 7, 1).tolist()
            binEdgesNeighborDistances = np.arange(2,10,0.5).tolist() + [100]
            binEdgesLocalDensities = np.arange(0,1,0.05).tolist()
            
            hexaticOrderParameterAvgs = np.zeros(numOfFrames, dtype = np.csingle)
            hexaticOrderParameterAvgNorms = np.zeros(numOfFrames)
            hexaticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype = np.csingle)
            hexaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
            hexaticOrderParameterModuliiStds = np.zeros(numOfFrames)
            
            pentaticOrderParameterAvgs = np.zeros(numOfFrames, dtype = np.csingle)
            pentaticOrderParameterAvgNorms = np.zeros(numOfFrames)
            pentaticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype = np.csingle)
            pentaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
            pentaticOrderParameterModuliiStds = np.zeros(numOfFrames)
            
            tetraticOrderParameterAvgs = np.zeros(numOfFrames, dtype = np.csingle)
            tetraticOrderParameterAvgNorms = np.zeros(numOfFrames)
            tetraticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype = np.csingle)
            tetraticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
            tetraticOrderParameterModuliiStds = np.zeros(numOfFrames)
            
            radialDistributionFunction = np.zeros((numOfFrames, len(radialRangeArray))) # pair correlation function: g(r)
            spatialCorrHexaOrderPara = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of hexatic order paramter: g6(r)
            spatialCorrPentaOrderPara = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of pentatic order paramter: g5(r)
            spatialCorrTetraOrderPara = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of tetratic order paramter: g4(r)
            spatialCorrHexaBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of bond orientation parameter: g6(r)/g(r)
            spatialCorrPentaBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of bond orientation parameter: g5(r)/g(r)
            spatialCorrTetraBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of bond orientation parameter: g4(r)/g(r)
            
            spatialCorrPos = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of positions: gG(r)
            
            dfNeighbors = pd.DataFrame(columns = ['frameNum', 'raftID', 'raftOrbitingDistInR','localDensity', 
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
            
            dfNeighborsAllFrames = pd.DataFrame(columns = ['frameNum', 'raftID', 'raftOrbitingDistInR', 'localDensity',
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
                vor = scipyVoronoi(raftLocations[:,currentFrameNum,:])
                allVertices = vor.vertices
                neighborPairs = vor.ridge_points # row# is the index of a ridge, columns are the two point# that correspond to the ridge 
                ridgeVertexPairs = np.asarray(vor.ridge_vertices) # row# is the index of a ridge, columns are two vertex# of the ridge
                raftPairwiseDistancesMatrix = raftPairwiseDistancesInRadius[:, :, currentFrameNum]
                

                for raftID in np.arange(numOfRafts):
                    ridgeIndices0 =  np.nonzero(neighborPairs[:,0] == raftID)
                    ridgeIndices1 =  np.nonzero(neighborPairs[:,1] == raftID)
                    ridgeIndices = np.concatenate((ridgeIndices0, ridgeIndices1), axis = None)
                    neighborPairsOfOneRaft = neighborPairs[ridgeIndices,:]
                    neighborsOfOneRaft = np.concatenate((neighborPairsOfOneRaft[neighborPairsOfOneRaft[:,0] == raftID,1], neighborPairsOfOneRaft[neighborPairsOfOneRaft[:,1] == raftID,0]))
                    ridgeVertexPairsOfOneRaft = ridgeVertexPairs[ridgeIndices,:]
                    neighborDistances = raftPairwiseDistancesMatrix[raftID, neighborsOfOneRaft]
                    neighborDistanceAvg = neighborDistances.mean()
                    
                    ## order parameters and their spatial correlation function
                    raftLocation = raftLocations[raftID,currentFrameNum,:]
                    raftOrbitingDistInR = raftOrbitingDistances[raftID, currentFrameNum] / radius # unit: R
                    neighborLocations = raftLocations[neighborsOfOneRaft,currentFrameNum,:]
                    
                    # note the negative sign, it is to make the angle Rhino-like
                    neighborAnglesInRad = np.arctan2(-(neighborLocations[:,1] - raftLocation[1]),(neighborLocations[:,0] - raftLocation[0]))
                    neighborAnglesInDeg = neighborAnglesInRad * 180 / np.pi
                    
                    raftHexaticOrderParameter = np.cos(neighborAnglesInRad*6).mean() + np.sin(neighborAnglesInRad*6).mean()*1j
                    raftPentaticOrderParameter = np.cos(neighborAnglesInRad*5).mean() + np.sin(neighborAnglesInRad*5).mean()*1j
                    raftTetraticOrderParameter = np.cos(neighborAnglesInRad*4).mean() + np.sin(neighborAnglesInRad*4).mean()*1j
                    
                    # calculate the local density of Voronoi cell
                    if np.all(ridgeVertexPairsOfOneRaft >= 0): 
                        vertexIDsOfOneRaft = np.unique(ridgeVertexPairsOfOneRaft)
                        verticesOfOneRaft = allVertices[vertexIDsOfOneRaft]
                        raftXY = raftLocations[raftID, currentFrameNum, :]
                        
                        #polar angles in plt.plot
                        polarAngles = np.arctan2((verticesOfOneRaft[:,1] - raftXY[1]), (verticesOfOneRaft[:,0] - raftXY[0])) * 180 / np.pi
                        
                        verticesOfOneRaftSorted = verticesOfOneRaft[polarAngles.argsort()]
                        
                        voronoiCellArea = PolygonArea(verticesOfOneRaftSorted[:,0], verticesOfOneRaftSorted[:,1])
                        
                        localDensity = radius * radius * np.pi / voronoiCellArea
                    else:
                        localDensity = 0
                
                    #initialize key variables
                    ridgeLengths = np.zeros(len(neighborsOfOneRaft))
                    ridgeLengthsScaled = np.zeros(len(neighborsOfOneRaft))
                    ridgeLengthsScaledNormalizedBySum = np.zeros(len(neighborsOfOneRaft))
                    ridgeLengthsScaledNormalizedByMax = np.zeros(len(neighborsOfOneRaft))
                    
                    #go through all ridges to calculate or assign ridge length
                    for ridgeIndexOfOneRaft, neighborID in enumerate(neighborsOfOneRaft):
                        neighborDistance = CalculateDistance(raftLocations[raftID,currentFrameNum,:], raftLocations[neighborID,currentFrameNum,:])
                        if np.all(ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft] >= 0 ):
                            vertex1ID = ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft][0]
                            vertex2ID = ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft][1]
                            vertex1 = allVertices[vertex1ID]
                            vertex2 = allVertices[vertex2ID]
                            ridgeLengths[ridgeIndexOfOneRaft] = CalculateDistance(vertex1, vertex2)
                            #for ridges that has one vertex outside the image (negative corrdinate)
                            #set ridge length to the be the diameter of the raft
                            if np.all(vertex1 >= 0) and np.all(vertex2 >= 0):
                                ridgeLengthsScaled[ridgeIndexOfOneRaft] = ridgeLengths[ridgeIndexOfOneRaft] * raftRadii[neighborID,currentFrameNum] * 2 / neighborDistance
                            else:
                                ridgeLengthsScaled[ridgeIndexOfOneRaft] = raftRadii[neighborID,currentFrameNum] ** 2 * 4 / neighborDistance
                        else:
                            #for ridges that has one vertex in the infinity ridge vertex#< 0 (= -1) 
                            #set ridge length to the be the diameter of the raft
                            ridgeLengths[ridgeIndexOfOneRaft] = raftRadii[neighborID,currentFrameNum] * 2
                            ridgeLengthsScaled[ridgeIndexOfOneRaft] = raftRadii[neighborID,currentFrameNum] ** 2 * 4 / neighborDistance
                            
                    ridgeLengthsScaledNormalizedBySum = ridgeLengthsScaled / ridgeLengthsScaled.sum()
                    ridgeLengthsScaledNormalizedByMax = ridgeLengthsScaled / ridgeLengthsScaled.max()
                    neighborCountWeighted = ridgeLengthsScaledNormalizedByMax.sum() # assuming the neighbor having the longest ridge counts one. 
                    neighborDistanceWeightedAvg = np.average(neighborDistances, weights = ridgeLengthsScaledNormalizedBySum)
                    
                    
                    dfNeighbors.loc[raftID, 'frameNum'] = currentFrameNum
                    dfNeighbors.loc[raftID, 'raftID'] = raftID
                    dfNeighbors.loc[raftID, 'raftOrbitingDistInR'] = raftOrbitingDistInR
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
                    
                 
#                hexaticOrderParameterList =  dfNeighbors['hexaticOrderParameter'].tolist()
#                pentaticOrderParameterList =  dfNeighbors['pentaticOrderParameter'].tolist()
#                tetraticOrderParameterList =  dfNeighbors['tetraticOrderParameter'].tolist()
#                
#                
#                hexaticOrderParameterArray = np.array(hexaticOrderParameterList)
#                hexaticOrderParameterAvgs[currentFrameNum] = hexaticOrderParameterArray.mean()
#                hexaticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(hexaticOrderParameterAvgs[currentFrameNum].real ** 2 + hexaticOrderParameterAvgs[currentFrameNum].imag ** 2)
#                hexaticOrderParameterMeanSquaredDeviations[currentFrameNum] = ((hexaticOrderParameterArray - hexaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
#                hexaticOrderParameterMolulii = np.absolute(hexaticOrderParameterArray)
#                hexaticOrderParameterModuliiAvgs[currentFrameNum] = hexaticOrderParameterMolulii.mean()
#                hexaticOrderParameterModuliiStds[currentFrameNum] = hexaticOrderParameterMolulii.std()
#                
#                pentaticOrderParameterArray = np.array(pentaticOrderParameterList)
#                pentaticOrderParameterAvgs[currentFrameNum] = pentaticOrderParameterArray.mean()
#                pentaticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(pentaticOrderParameterAvgs[currentFrameNum].real ** 2 + pentaticOrderParameterAvgs[currentFrameNum].imag ** 2)
#                pentaticOrderParameterMeanSquaredDeviations[currentFrameNum] = ((pentaticOrderParameterArray - pentaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
#                pentaticOrderParameterModulii = np.absolute(pentaticOrderParameterArray)
#                pentaticOrderParameterModuliiAvgs[currentFrameNum] = pentaticOrderParameterModulii.mean()
#                pentaticOrderParameterModuliiStds[currentFrameNum] = pentaticOrderParameterModulii.std()
#                
#                tetraticOrderParameterArray = np.array(tetraticOrderParameterList)
#                tetraticOrderParameterAvgs[currentFrameNum] = tetraticOrderParameterArray.mean()
#                tetraticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(tetraticOrderParameterAvgs[currentFrameNum].real ** 2 + tetraticOrderParameterAvgs[currentFrameNum].imag ** 2)
#                tetraticOrderParameterMeanSquaredDeviations[currentFrameNum] = ((tetraticOrderParameterArray - tetraticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
#                tetraticOrderParameterModulii = np.absolute(tetraticOrderParameterArray)
#                tetraticOrderParameterModuliiAvgs[currentFrameNum] = tetraticOrderParameterModulii.mean()
#                tetraticOrderParameterModuliiStds[currentFrameNum] = tetraticOrderParameterModulii.std()
#                
#                angles = np.arange(0,2*np.pi, np.pi/3) + np.pi/6 #+ np.angle(hexaticOrderParameterAvgs[currentFrameNum]) #0
#                neighborDistancesList = np.concatenate(dfNeighbors['neighborDistances'].tolist())
#                NDistAvg = np.asarray(neighborDistancesList).mean() #np.asarray(neighborDistancesList).mean() # in unit of R
#                G = 2 * np.pi * np.array((np.cos(angles), np.sin(angles))).T / NDistAvg # np.asarray(neighborDistancesList).mean()
#                cosPart = np.zeros((len(radialRangeArray), numOfRafts, len(angles)))
#                cosPartRaftCount = np.zeros(len(radialRangeArray))
##                tempCount = np.zeros(len(radialRangeArray))
#
#                ## g(r) and g6(r), gG(r) for this frame
#                for radialIndex, radialIntervalStart in enumerate(radialRangeArray): 
#                    # radialIntervalStart, radialIndex = 2, 0
#                    radialIntervalEnd =  radialIntervalStart + deltaR
#                    #g(r)
#                    js, ks = np.logical_and(raftPairwiseDistancesMatrix>=radialIntervalStart, raftPairwiseDistancesMatrix<radialIntervalEnd).nonzero()
#                    count = len(js)
#                    density = numOfRafts / sizeOfArenaInRadius**2 
#                    radialDistributionFunction[currentFrameNum, radialIndex] =  count / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
#                    
#                    #gG(r_)                  
#                    for raft1ID in np.arange(numOfRafts):
#                        # raft1ID = 0
#                        originXY = raftLocations[raft1ID, currentFrameNum, :] / raftRadii.mean()
#                        raftLocationsNew = raftLocations[:,currentFrameNum,:] / raftRadii.mean() - originXY
#                        for angleID, angle in enumerate(angles):
#                            # angleID, angle = 0, np.angle(hexaticOrderParameterAvgs[currentFrameNum])
#                            conditionX = np.logical_and(raftLocationsNew[:,0] >= radialIntervalStart * np.cos(angle) - NDistAvg/2, raftLocationsNew[:,0] < radialIntervalStart * np.cos(angle) + NDistAvg/2)
#                            conditionY = np.logical_and(raftLocationsNew[:,1] >= radialIntervalStart * np.sin(angle) - NDistAvg/2, raftLocationsNew[:,1] < radialIntervalStart * np.sin(angle) + NDistAvg/2)
#                            conditionXY = np.logical_and(conditionX, conditionY)
#                            if conditionXY.any():          
#                                vector12 = raftLocationsNew[conditionXY.nonzero()]
#                                cosPart[radialIndex, raft1ID, angleID] = np.cos(G[angleID, 0] * vector12[:, 0] + G[angleID, 1] * vector12[:,1]).sum()
#                                cosPartRaftCount[radialIndex] = cosPartRaftCount[radialIndex] + np.count_nonzero(conditionXY)
#                    if np.count_nonzero(cosPart[radialIndex, :, :]) > 0: 
##                        tempCount[radialIndex] = np.count_nonzero(cosPart[radialIndex, :, :])
#                        spatialCorrPos[currentFrameNum, radialIndex] = cosPart[radialIndex, :, :].sum() / cosPartRaftCount[radialIndex]     
#                    
#
#                    # g6(r), g5(r), g4(r)
#                    sumOfProductsOfPsi6 = (hexaticOrderParameterArray[js] * np.conjugate(hexaticOrderParameterArray[ks])).sum().real
#                    spatialCorrHexaOrderPara[currentFrameNum, radialIndex] = sumOfProductsOfPsi6 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
#                    sumOfProductsOfPsi5 = (pentaticOrderParameterArray[js] * np.conjugate(pentaticOrderParameterArray[ks])).sum().real
#                    spatialCorrPentaOrderPara[currentFrameNum, radialIndex] = sumOfProductsOfPsi5 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
#                    sumOfProductsOfPsi4 = (tetraticOrderParameterArray[js] * np.conjugate(tetraticOrderParameterArray[ks])).sum().real
#                    spatialCorrTetraOrderPara[currentFrameNum, radialIndex] = sumOfProductsOfPsi4 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
#                    # g6(r)/g(r); g5(r)/g(r); g4(r)/g(r)
#                    if radialDistributionFunction[currentFrameNum, radialIndex] != 0: 
#                        spatialCorrHexaBondOrientationOrder[currentFrameNum, radialIndex] = spatialCorrHexaOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[currentFrameNum, radialIndex]
#                        spatialCorrPentaBondOrientationOrder[currentFrameNum, radialIndex] = spatialCorrPentaOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[currentFrameNum, radialIndex]
#                        spatialCorrTetraBondOrientationOrder[currentFrameNum, radialIndex] = spatialCorrTetraOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[currentFrameNum, radialIndex]
#             
                #calculate various entropies and entropies within windows of different sizes
                neighborCountSeries = dfNeighbors['neighborCount']
                neighborCountWeightedList = dfNeighbors['neighborCountWeighted'].tolist()
                neighborDistancesList = np.concatenate(dfNeighbors['neighborDistances'].tolist())
                localDensitiesList = dfNeighbors['localDensity'].tolist()
                
                count1 = np.asarray(neighborCountSeries.value_counts())
                entropyByNeighborCount[currentFrameNum] = ShannonEntropy(count1)
                
                count2, _ = np.histogram(np.asarray(neighborCountWeightedList),binEdgesNeighborCountWeighted)
                entropyByNeighborCountWeighted[currentFrameNum] = ShannonEntropy(count2)
                
                count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
                entropyByNeighborDistances[currentFrameNum] = ShannonEntropy(count3)
                
                count4, _ = np.histogram(np.asarray(localDensitiesList), binEdgesLocalDensities)
                entropyByLocalDensities[currentFrameNum] = ShannonEntropy(count4)
                
                for windowID, samplingWindowRadius in enumerate(samplingWindowSizes):
                    dfRaftsInWindow = dfNeighbors[dfNeighbors.raftOrbitingDistInR <= samplingWindowRadius]
                    
                    neighborCountSeries = dfRaftsInWindow['neighborCount']
                    count1 = np.asarray(neighborCountSeries.value_counts())
                    entropyByNeighborCountInWindows[currentFrameNum, windowID] = ShannonEntropy(count1)
                    
                    neighborCountWeightedList = dfRaftsInWindow['neighborCountWeighted'].tolist()
                    count2, _ = np.histogram(np.asarray(neighborCountWeightedList),binEdgesNeighborCountWeighted)
                    entropyByNeighborCountWeightedInWindows[currentFrameNum, windowID] = ShannonEntropy(count2)
                    
                    neighborDistancesList = np.concatenate(dfRaftsInWindow['neighborDistances'].tolist())
                    count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
                    entropyByNeighborDistancesInWindows[currentFrameNum, windowID] = ShannonEntropy(count3)
                    
                    localDensitiesList = dfRaftsInWindow['localDensity'].tolist()
                    count4, _ = np.histogram(np.asarray(localDensitiesList), binEdgesLocalDensities)
                    entropyByLocalDensitiesInWindows[currentFrameNum, windowID] = ShannonEntropy(count4)
                    

                neighborDistanceAvgAllRafts[currentFrameNum] = dfNeighbors['neighborDistanceAvg'].mean()
                neighborDistanceWeightedAvgAllRafts[currentFrameNum] = dfNeighbors['neighborDistanceWeightedAvg'].mean()
                
#                dfNeighborsAllFrames = dfNeighborsAllFrames.append(dfNeighbors,ignore_index=True)
            
#            dfNeighborsAllFrames = dfNeighborsAllFrames.infer_objects()
#            dfNeighborsAllFrames = dfNeighborsAllFrames.sort_values(['frameNum','raftID'], ascending = [1,1])
            
      
            
            
            # Temporal correlation of g6, g5, and g4, the unit of deltaT is frame
            temporalCorrHexaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype = complex) 
            temporalCorrPentaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype = complex) 
            temporalCorrTetraBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype = complex) 
            temporalCorrHexaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype = complex)
            temporalCorrPentaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype = complex)
            temporalCorrTetraBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype = complex)
            
            for raftID in np.arange(numOfRafts): 
                hexaOrdParaOfOneRaftSeries = dfNeighborsAllFrames.query('raftID == {}'.format(raftID)).hexaticOrderParameter
                pentaOrdParaOfOneRaftSeries = dfNeighborsAllFrames.query('raftID == {}'.format(raftID)).pentaticOrderParameter
                tetraOrdParaOfOneRaftSeries = dfNeighborsAllFrames.query('raftID == {}'.format(raftID)).tetraticOrderParameter
                
                hexaOrdParaOfOneRaftArray = np.array(hexaOrdParaOfOneRaftSeries.tolist())
                pentaOrdParaOfOneRaftArray = np.array(pentaOrdParaOfOneRaftSeries.tolist())
                tetraOrdParaOfOneRaftArray = np.array(tetraOrdParaOfOneRaftSeries.tolist())
                # construct the Toeplitz matrix, repeat input array twice to avoid the default conjugation 
                hexaOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(hexaOrdParaOfOneRaftArray, hexaOrdParaOfOneRaftArray)
                pentaOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(pentaOrdParaOfOneRaftArray, pentaOrdParaOfOneRaftArray)
                tetraOrdParaOfOneRaftToeplitzMatrix = scipy.linalg.toeplitz(tetraOrdParaOfOneRaftArray, tetraOrdParaOfOneRaftArray)
                
                # construct the conjugated array and braodcasted it to the shape of the Toeplitz matrix
                hexaOrdParaOfOneRaftArrayConjugate = np.conjugate(hexaOrdParaOfOneRaftArray)
                hexaOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(np.broadcast_to(hexaOrdParaOfOneRaftArrayConjugate, hexaOrdParaOfOneRaftToeplitzMatrix.shape))
                pentaOrdParaOfOneRaftArrayConjugate = np.conjugate(pentaOrdParaOfOneRaftArray)
                pentaOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(np.broadcast_to(pentaOrdParaOfOneRaftArrayConjugate, pentaOrdParaOfOneRaftToeplitzMatrix.shape))
                tetraOrdParaOfOneRaftArrayConjugate = np.conjugate(tetraOrdParaOfOneRaftArray)
                tetraOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(np.broadcast_to(tetraOrdParaOfOneRaftArrayConjugate, tetraOrdParaOfOneRaftToeplitzMatrix.shape))
                
                # multiply the two matrix so that for each column, the rows on and below the diagonal are the products of 
                # the conjugate of psi6(t0) and psi6(t0 + tStepSize), the tStepSize is the same the column index. 
                hexaOrdParaOfOneRaftBroadcastedTimesToeplitz = hexaOrdParaOfOneRaftArrayConjugateBroadcasted * hexaOrdParaOfOneRaftToeplitzMatrix
                pentaOrdParaOfOneRaftBroadcastedTimesToeplitz = pentaOrdParaOfOneRaftArrayConjugateBroadcasted * pentaOrdParaOfOneRaftToeplitzMatrix
                tetraOrdParaOfOneRaftBroadcastedTimesToeplitz = tetraOrdParaOfOneRaftArrayConjugateBroadcasted * tetraOrdParaOfOneRaftToeplitzMatrix
                
                for tStepSize in np.arange(numOfFrames):
                    temporalCorrHexaBondOrientationOrder[raftID, tStepSize] = np.average(hexaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:,tStepSize])
                    temporalCorrPentaBondOrientationOrder[raftID, tStepSize] = np.average(pentaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:,tStepSize])
                    temporalCorrTetraBondOrientationOrder[raftID, tStepSize] = np.average(tetraOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:,tStepSize])
            
            temporalCorrHexaBondOrientationOrderAvgAllRafts = temporalCorrHexaBondOrientationOrder.mean(axis = 0)
            temporalCorrPentaBondOrientationOrderAvgAllRafts = temporalCorrPentaBondOrientationOrder.mean(axis = 0)
            temporalCorrTetraBondOrientationOrderAvgAllRafts = temporalCorrTetraBondOrientationOrder.mean(axis = 0) 
                
        ###########################  mutual information analysis
        if analysisType == 3 or analysisType == 4:
            # the durartion for which the frames are sampled to calculate one MI
            widthOfInterval = 100 # unit: number of frames,
            
            numOfBins = 16
            
            # The gap between two successive MI calculation. 
            # Try keep (numOfFrames - widthOfInterval)//samplingGap an integer
            samplingGap = 50 # unit: number of frames
             
            numOfSamples = (numOfFrames - widthOfInterval)//samplingGap + 1
            sampleFrameNums = np.arange(widthOfInterval,numOfFrames,samplingGap)
            
            # pretreatment of position data
            raftOrbitingAnglesAdjusted = AdjustOrbitingAngles2(raftOrbitingAngles, orbiting_angles_diff_threshold = 200)
            raftVelocityR = np.gradient(raftOrbitingDistances, axis=1)
            raftVelocityTheta = np.gradient(raftOrbitingAnglesAdjusted, axis=1)
            raftVelocityNormPolar = np.sqrt(raftVelocityR * raftVelocityR + np.square(raftOrbitingDistances * np.radians(raftVelocityTheta)))
            raftVelocityX = np.gradient(raftLocations[:,:,0],axis = 1)
            raftVelocityY = np.gradient(raftLocations[:,:,1],axis = 1)
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
            
            
            ### mutual information calculation
            for i, endOfInterval in enumerate(sampleFrameNums):
                distancesMatrix = raftOrbitingDistances[:, endOfInterval-widthOfInterval:endOfInterval] 
                mutualInfoAllSamplesAllRafts[:,:,i,0] = MutualInfoMatrix(distancesMatrix, numOfBins)
                    
                angleMatrix = raftOrbitingAnglesAdjusted[:, endOfInterval-widthOfInterval:endOfInterval]
                mutualInfoAllSamplesAllRafts[:,:,i,1] = MutualInfoMatrix(angleMatrix, numOfBins)
                
                coordinateXMatrix = raftLocations[:, endOfInterval-widthOfInterval:endOfInterval, 0]
                mutualInfoAllSamplesAllRafts[:,:,i,2] = MutualInfoMatrix(coordinateXMatrix, numOfBins)
                
                coordinateYMatrix = raftLocations[:, endOfInterval-widthOfInterval:endOfInterval, 1]
                mutualInfoAllSamplesAllRafts[:,:,i,3] = MutualInfoMatrix(coordinateYMatrix, numOfBins)
                
                velocityRMatrix = raftVelocityR[:, endOfInterval-widthOfInterval:endOfInterval]
                mutualInfoAllSamplesAllRafts[:,:,i,4] = MutualInfoMatrix(velocityRMatrix, numOfBins)
                
                velocityThetaMatrix = raftVelocityTheta[:, endOfInterval-widthOfInterval:endOfInterval]
                mutualInfoAllSamplesAllRafts[:,:,i,5] = MutualInfoMatrix(velocityThetaMatrix, numOfBins)
                
                velocityNormPolarMatrix = raftVelocityNormPolar[:, endOfInterval-widthOfInterval:endOfInterval]
                mutualInfoAllSamplesAllRafts[:,:,i,6] = MutualInfoMatrix(velocityNormPolarMatrix, numOfBins)
                
                velocityXMatrix = raftVelocityX[:, endOfInterval-widthOfInterval:endOfInterval]
                mutualInfoAllSamplesAllRafts[:,:,i,7] = MutualInfoMatrix(velocityXMatrix, numOfBins)
                
                velocityYMatrix = raftVelocityY[:, endOfInterval-widthOfInterval:endOfInterval]
                mutualInfoAllSamplesAllRafts[:,:,i,8] = MutualInfoMatrix(velocityYMatrix, numOfBins)
                
                velocityNormXYMatrix = raftVelocityNormXY[:, endOfInterval-widthOfInterval:endOfInterval]
                mutualInfoAllSamplesAllRafts[:,:,i,9] = MutualInfoMatrix(velocityNormXYMatrix, numOfBins)
            
            
            mutualInfoAllSamplesAvgOverAllRafts =  mutualInfoAllSamplesAllRafts.mean((0,1))
            mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly = np.trace(mutualInfoAllSamplesAllRafts, axis1 = 0, axis2 = 1) / numOfRafts
            mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI = (mutualInfoAllSamplesAvgOverAllRafts * numOfRafts - mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly) / (numOfRafts - 1)
            
            mutualInfoAvg = mutualInfoAllSamplesAvgOverAllRafts.mean(axis = 0)
            mutualInfoAvgSelfMIOnly = mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly.mean(axis = 0)
            mutualInfoAvgExcludingSelfMI = mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI.mean(axis = 0)

        ###########################  particle velocity and MSD analysis
        if analysisType == 5:
            embeddingDimension = 20
            reconstructionComponents = np.arange(5)
            
            raftLocationsX = raftLocations[:,:,0]
            raftLocationsY = raftLocations[:,:,1]
            
            raftVelocityX = np.gradient(raftLocationsX, axis = 1) # unit pixel/frame
            raftVelocityY = np.gradient(raftLocationsY, axis = 1) # unit pixel/frame
            raftVelocityNorm = np.sqrt(raftVelocityX ** 2 + raftVelocityY ** 2)
            
            raftVelocityXFiltered = np.zeros_like(raftVelocityX)
            raftVelocityYFiltered = np.zeros_like(raftVelocityY)
            for raftID in np.arange(numOfRafts):
                raftVelocityXFiltered[raftID,:] = SSAFull(raftVelocityX[raftID,:], embeddingDimension, reconstructionComponents)
                raftVelocityYFiltered[raftID,:] = SSAFull(raftVelocityY[raftID,:], embeddingDimension, reconstructionComponents)
            
            raftVelocityNormFiltered = np.sqrt(raftVelocityXFiltered ** 2 + raftVelocityYFiltered ** 2)
            
            raftKineticEnergies = raftVelocityNormFiltered ** 2
            raftKineticEnergiesSumAllRafts = raftKineticEnergies.sum(axis = 0)
            
            # get the radial and tangential vectors
            raftOrbitingCentersXBroadcasted  = np.broadcast_to(raftOrbitingCenters[:,0], raftLocationsX.shape)
            raftOrbitingCentersYBroadcasted  = np.broadcast_to(raftOrbitingCenters[:,1], raftLocationsY.shape)
            raftRadialVectorX = raftLocationsX - raftOrbitingCentersXBroadcasted
            raftRadialVectorY = raftLocationsY - raftOrbitingCentersYBroadcasted
            raftRadialVectorXUnitized = raftRadialVectorX / np.sqrt(raftRadialVectorX ** 2 + raftRadialVectorY ** 2)
            raftRadialVectorYUnitized = raftRadialVectorY / np.sqrt(raftRadialVectorX ** 2 + raftRadialVectorY ** 2)
            raftTangentialVectorXUnitized = -raftRadialVectorYUnitized # negative sign is assigned such that the tangential velocity is positive
            raftTangentialVectorYUnitized =  raftRadialVectorXUnitized 
            # get the radial and tangential velocities
            raftRadialVelocity = raftVelocityXFiltered * raftRadialVectorXUnitized + raftVelocityYFiltered * raftRadialVectorYUnitized
            raftTangentialVelocity = raftVelocityXFiltered * raftTangentialVectorXUnitized + raftVelocityYFiltered * raftTangentialVectorYUnitized
            
            # MSD analysis
            particleMSD = np.zeros((numOfRafts, numOfFrames))
            particleMSDstd = np.zeros((numOfRafts, numOfFrames))
            particleRMSD = np.zeros((numOfRafts, numOfFrames))
                
            for raftID in np.arange(numOfRafts):
                corrX = raftLocations[raftID,:,0]
                corrY = raftLocations[raftID,:,1]
                
                # construct Toeplitz matrix, 1st column is the corrdinateX and Y, top right half all zeros
                corrXToeplitz = scipy.linalg.toeplitz(corrX, np.zeros(numOfFrames))
                corrYToeplitz = scipy.linalg.toeplitz(corrY, np.zeros(numOfFrames))
                
                # broad cast the column of coordinate x and y to the size of Toeplitz matrix
                corrXBroadcasted = np.transpose(np.broadcast_to(corrX, corrXToeplitz.shape))
                corrYBroadcasted = np.transpose(np.broadcast_to(corrY, corrYToeplitz.shape))
                
                # substrate Toeplitz matrix from broadcasted matrix,  
                # for each column, the rows on and below the diagonal are the displacement in x and y coordinates
                # step size is the column index. 
                corrXdiffMatrixSquared = (corrXBroadcasted - corrXToeplitz)**2
                corrYdiffMatrixSquared = (corrYBroadcasted - corrYToeplitz)**2
                particleSquareDisplacement = corrXdiffMatrixSquared + corrYdiffMatrixSquared
                
                # calculate mean square displacement
                for stepSize in np.arange(numOfFrames):
                    particleMSD[raftID, stepSize] = np.average(particleSquareDisplacement[stepSize:,stepSize])
                    particleMSDstd[raftID, stepSize] = np.std(particleSquareDisplacement[stepSize:,stepSize])
            
                particleRMSD[raftID, :] = np.sqrt(particleMSD[raftID, :])
            
        ###########################   save postprocessed data file
        tempShelf = shelve.open(shelveDataFileName)
        if analysisType == 1 or analysisType == 2 or analysisType == 4 or analysisType == 5:
            for key in listOfNewVariablesForClusterAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    #print('ERROR shelving: {0}'.format(key))
                    pass
        
        if analysisType == 2 or analysisType == 4 or analysisType == 5:
            for key in listOfNewVariablesForVoronoiAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    #print('ERROR shelving: {0}'.format(key))
                    pass

        if analysisType == 3 or analysisType == 4:
            for key in listOfNewVariablesForEntropyAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    #print('ERROR shelving: {0}'.format(key))
                    pass
        if analysisType == 5:
            for key in listOfNewVariablesForVelocityMSDAnalysis:
                try:
                    tempShelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, tempShelf, and imported modules can not be shelved.
                    #
                    #print('ERROR shelving: {0}'.format(key))
                    pass
        tempShelf.close()

#%% extract data from all the post-processed files and store in dataframes for plotting

#dataFileListPostprocessedOnly = glob.glob('*postprocessed.dat')
#dataFileListPostprocessedOnly.sort()

summaryDataFrameColNames = ['batchNum','spinSpeed','commentsSub', 
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

dfSummary = pd.DataFrame(columns = summaryDataFrameColNames)

dfRadialDistributionFunction  = pd.DataFrame(columns = ['distancesInRadius'])
dfSpatialCorrHexaOrderPara  = pd.DataFrame(columns = ['distancesInRadius'])
dfSpatialCorrPentaOrderPara  = pd.DataFrame(columns = ['distancesInRadius'])
dfSpatialCorrTetraOrderPara  = pd.DataFrame(columns = ['distancesInRadius'])
dfSpatialCorrHexaBondOrientationOrder  = pd.DataFrame(columns = ['distancesInRadius'])
dfSpatialCorrPentaBondOrientationOrder  = pd.DataFrame(columns = ['distancesInRadius'])
dfSpatialCorrTetraBondOrientationOrder  = pd.DataFrame(columns = ['distancesInRadius'])

dfTemporalCorrHexaBondOrientationOrderOfOneRaft = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfTemporalCorrPentaBondOrientationOrderOfOneRaft = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfTemporalCorrTetraBondOrientationOrderOfOneRaft = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = pd.DataFrame(columns = ['timeDifferenceInFrames'])

dfHexaBondOrientatiotionOrderModuliiAvgTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfHexaBondOrientatiotionOrderAvgNormTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])

dfPentaBondOrientatiotionOrderModuliiAvgTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfPentaBondOrientatiotionOrderAvgNormTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])

dfTetraBondOrientatiotionOrderModuliiAvgTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfTetraBondOrientatiotionOrderAvgNormTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])


dfNeighborDistTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfEntropyNeighborDistTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfEntropyNeighborCountTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfEntropyLocalDensitiesTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])
dfClusterSizeIncludingLonersTime = pd.DataFrame(columns = ['timeDifferenceInFrames'])




dfRaftVelocitiesVsOrbitingDistances = pd.DataFrame()

dfRaftXYAndMSD = pd.DataFrame(columns = ['timeDifferenceInFrames'])

# for now, just fill the distancesInRadius column with radialRangeArray and timeDifferenceInFrames with timeDifferenceArray
deltaR = 1 # check this every time
radialRangeArray = np.arange(2, 100, deltaR)
dfRadialDistributionFunction['distancesInRadius'] = radialRangeArray
dfSpatialCorrHexaOrderPara['distancesInRadius'] = radialRangeArray
dfSpatialCorrPentaOrderPara['distancesInRadius'] = radialRangeArray
dfSpatialCorrTetraOrderPara['distancesInRadius'] = radialRangeArray
dfSpatialCorrHexaBondOrientationOrder['distancesInRadius'] = radialRangeArray
dfSpatialCorrPentaBondOrientationOrder['distancesInRadius'] = radialRangeArray
dfSpatialCorrTetraBondOrientationOrder['distancesInRadius'] = radialRangeArray

radiusInPixel = 23 # check this every time
scaleBar = 150 / radiusInPixel # unit micron/pixel
numOfFrames = 500 # check this every time 
frameRate = 75 # unit fps, check this every time
#timeDifferenceArray = np.arange(numOfFrames)
#dfTemporalCorrHexaBondOrientationOrderOfOneRaft['timeDifferenceInFrames'] = timeDifferenceArray
#dfTemporalCorrPentaBondOrientationOrderOfOneRaft['timeDifferenceInFrames'] = timeDifferenceArray
#dfTemporalCorrTetraBondOrientationOrderOfOneRaft['timeDifferenceInFrames'] = timeDifferenceArray
#dfTemporalCorrHexaBondOrientationOrderAvgAllRafts['timeDifferenceInFrames'] = timeDifferenceArray
#dfTemporalCorrPentaBondOrientationOrderAvgAllRafts['timeDifferenceInFrames'] = timeDifferenceArray
#dfTemporalCorrTetraBondOrientationOrderAvgAllRafts['timeDifferenceInFrames'] = timeDifferenceArray
#dfRaftXYAndMSD['timeDifferenceInFrames'] = timeDifferenceArray



frameNumToLookAt = 0
raftIDToLookAt = 0

analysisType = 5 # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi

for dataID in range(0,len(mainDataList)):
    # load data from main DataList: 
    dfSummary.loc[dataID, 'batchNum'] = mainDataList[dataID]['batchNum']
    dfSummary.loc[dataID, 'spinSpeed'] = mainDataList[dataID]['spinSpeed']
    dfSummary.loc[dataID, 'commentsSub'] = mainDataList[dataID]['commentsSub']
    dfSummary.loc[dataID, 'magnification'] = mainDataList[dataID]['magnification']
    dfSummary.loc[dataID, 'radiusAvg'] = mainDataList[dataID]['raftRadii'].mean()
    dfSummary.loc[dataID, 'numOfFrames'] = mainDataList[dataID]['numOfFrames']
    dfSummary.loc[dataID, 'frameWidth'] = mainDataList[dataID]['currentFrameGray'].shape[0] # from frame size, you can guess frame rate. 
    dfSummary.loc[dataID, 'frameHeight'] = mainDataList[dataID]['currentFrameGray'].shape[1]
#    radiusInPixel = 23 # check this every time
    scaleBar = 150 / mainDataList[dataID]['raftRadii'].mean() # unit micron/pixel
    
    #construct shelveName
    date = mainDataList[dataID]['date']
    numOfRafts = mainDataList[dataID]['numOfRafts']
    batchNum = mainDataList[dataID]['batchNum']
    spinSpeed = mainDataList[dataID]['spinSpeed']
    magnification = mainDataList[dataID]['magnification']
    shelveName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_' + 'postprocessed' + str(analysisType)
    shelveDataFileName = shelveName + '.dat'
    
    if os.path.isfile(shelveDataFileName):
        tempShelf = shelve.open(shelveName)
        
        if analysisType == 1 or analysisType == 2 or analysisType == 4 or analysisType == 5:
            dfSummary.loc[dataID, 'clusterSizeAvgIncludingLonersAllFrames'] = tempShelf['clusterSizeAvgIncludingLonersAllFrames']
            dfSummary.loc[dataID, 'clusterSizeAvgExcludingLonersAllFrames'] = tempShelf['clusterSizeAvgExcludingLonersAllFrames']
            dfSummary.loc[dataID, 'raftPairwiseEdgeEdgeDistancesSmallestMean'] = tempShelf['raftPairwiseEdgeEdgeDistancesSmallestMean']
            dfSummary.loc[dataID, 'raftPairwiseEdgeEdgeDistancesSmallestStd'] = tempShelf['raftPairwiseEdgeEdgeDistancesSmallestStd']
            dfSummary.loc[dataID, 'numOfLonersAvgAllFrames'] = tempShelf['numOfLonersAvgAllFrames']
        
        if analysisType == 2 or analysisType == 4 or analysisType == 5:
            dfSummary.loc[dataID, 'entropyByNeighborCountAvgAllFrames'] = tempShelf['entropyByNeighborCount'].mean()
            dfSummary.loc[dataID, 'entropyByNeighborCountStdAllFrames'] = tempShelf['entropyByNeighborCount'].std()
            dfSummary.loc[dataID, 'entropyByNeighborCountWeightedAvgAllFrames'] = tempShelf['entropyByNeighborCountWeighted'].mean()
            dfSummary.loc[dataID, 'entropyByNeighborCountWeightedStdAllFrames'] = tempShelf['entropyByNeighborCountWeighted'].std()
            dfSummary.loc[dataID, 'entropyByNeighborDistancesAvgAllFrames'] = tempShelf['entropyByNeighborDistances'].mean()
            dfSummary.loc[dataID, 'entropyByNeighborDistancesStdAllFrames'] = tempShelf['entropyByNeighborDistances'].std()
            dfSummary.loc[dataID, 'entropyByLocalDensitiesAvgAllFrames'] = tempShelf['entropyByLocalDensities'].mean()
            dfSummary.loc[dataID, 'entropyByLocalDensitiesStdAllFrames'] = tempShelf['entropyByLocalDensities'].std()
            dfSummary.loc[dataID, 'neighborDistanceAvgAllRaftsAvgAllFrames'] = tempShelf['neighborDistanceAvgAllRafts'].mean()
            dfSummary.loc[dataID, 'neighborDistanceAvgAllRaftsStdAllFrames'] = tempShelf['neighborDistanceAvgAllRafts'].std()
            dfSummary.loc[dataID, 'neighborDistanceWeightedAvgAllRaftsAvgAllFrames'] = tempShelf['neighborDistanceWeightedAvgAllRafts'].mean()
            dfSummary.loc[dataID, 'neighborDistanceWeightedAvgAllRaftsStdAllFrames'] = tempShelf['neighborDistanceWeightedAvgAllRafts'].std()
            dfSummary.loc[dataID, 'deltaR'] = tempShelf['deltaR']
            dfSummary.loc[dataID, 'hexaticOrderParameterAvgNormsAvgAllFrames'] = tempShelf['hexaticOrderParameterAvgNorms'].mean()
            dfSummary.loc[dataID, 'hexaticOrderParameterAvgNormsStdAllFrames'] = tempShelf['hexaticOrderParameterAvgNorms'].std()
            dfSummary.loc[dataID, 'hexaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames'] = tempShelf['hexaticOrderParameterModuliiAvgs'].mean()
            dfSummary.loc[dataID, 'hexaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames'] = tempShelf['hexaticOrderParameterModuliiAvgs'].std()
            dfSummary.loc[dataID, 'pentaticOrderParameterAvgNormsAvgAllFrames'] = tempShelf['pentaticOrderParameterAvgNorms'].mean()
            dfSummary.loc[dataID, 'pentaticOrderParameterAvgNormsStdAllFrames'] = tempShelf['pentaticOrderParameterAvgNorms'].std()
            dfSummary.loc[dataID, 'pentaticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames'] = tempShelf['pentaticOrderParameterModuliiAvgs'].mean()
            dfSummary.loc[dataID, 'pentaticOrderParameterModuliiAvgsAllRaftsStdsAllFrames'] = tempShelf['pentaticOrderParameterModuliiAvgs'].std()
            dfSummary.loc[dataID, 'tetraticOrderParameterAvgNormsAvgAllFrames'] = tempShelf['tetraticOrderParameterAvgNorms'].mean()
            dfSummary.loc[dataID, 'tetraticOrderParameterAvgNormsStdAllFrames'] = tempShelf['tetraticOrderParameterAvgNorms'].std()
            dfSummary.loc[dataID, 'tetraticOrderParameterModuliiAvgsAllRaftsAvgsAllFrames'] = tempShelf['tetraticOrderParameterModuliiAvgs'].mean()
            dfSummary.loc[dataID, 'tetraticOrderParameterModuliiAvgsAllRaftsStdsAllFrames'] = tempShelf['tetraticOrderParameterModuliiAvgs'].std()
            columnName = str(batchNum) + '_' + str(spinSpeed).zfill(4)
            dfRadialDistributionFunction[columnName] = tempShelf['radialDistributionFunction'][frameNumToLookAt,:]
            dfSpatialCorrHexaOrderPara[columnName] = tempShelf['spatialCorrHexaOrderPara'][frameNumToLookAt,:]
            dfSpatialCorrPentaOrderPara[columnName] = tempShelf['spatialCorrPentaOrderPara'][frameNumToLookAt,:]
            dfSpatialCorrTetraOrderPara[columnName] = tempShelf['spatialCorrTetraOrderPara'][frameNumToLookAt,:]
            dfSpatialCorrHexaBondOrientationOrder[columnName] = tempShelf['spatialCorrHexaBondOrientationOrder'][frameNumToLookAt,:]
            dfSpatialCorrPentaBondOrientationOrder[columnName] = tempShelf['spatialCorrPentaBondOrientationOrder'][frameNumToLookAt,:]
            dfSpatialCorrTetraBondOrientationOrder[columnName] = tempShelf['spatialCorrTetraBondOrientationOrder'][frameNumToLookAt,:]
            dfTemporalCorrHexaBondOrientationOrderOfOneRaft = dfTemporalCorrHexaBondOrientationOrderOfOneRaft.join(pd.Series(np.real(tempShelf['temporalCorrHexaBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_real'),how='outer')
            dfTemporalCorrHexaBondOrientationOrderOfOneRaft = dfTemporalCorrHexaBondOrientationOrderOfOneRaft.join(pd.Series(np.imag(tempShelf['temporalCorrHexaBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_imag'),how='outer')
            dfTemporalCorrHexaBondOrientationOrderOfOneRaft = dfTemporalCorrHexaBondOrientationOrderOfOneRaft.join(pd.Series(np.absolute(tempShelf['temporalCorrHexaBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_abs'),how='outer')
            dfTemporalCorrPentaBondOrientationOrderOfOneRaft = dfTemporalCorrPentaBondOrientationOrderOfOneRaft.join(pd.Series(np.real(tempShelf['temporalCorrPentaBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_real'),how='outer')
            dfTemporalCorrPentaBondOrientationOrderOfOneRaft = dfTemporalCorrPentaBondOrientationOrderOfOneRaft.join(pd.Series(np.imag(tempShelf['temporalCorrPentaBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_imag'),how='outer')
            dfTemporalCorrPentaBondOrientationOrderOfOneRaft = dfTemporalCorrPentaBondOrientationOrderOfOneRaft.join(pd.Series(np.absolute(tempShelf['temporalCorrPentaBondOrientationOrder'][raftIDToLookAt,:]),name=columnName + '_abs'),how='outer')
            dfTemporalCorrTetraBondOrientationOrderOfOneRaft = dfTemporalCorrTetraBondOrientationOrderOfOneRaft.join(pd.Series(np.real(tempShelf['temporalCorrTetraBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_real'),how='outer')
            dfTemporalCorrTetraBondOrientationOrderOfOneRaft = dfTemporalCorrTetraBondOrientationOrderOfOneRaft.join(pd.Series(np.imag(tempShelf['temporalCorrTetraBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_imag'),how='outer')
            dfTemporalCorrTetraBondOrientationOrderOfOneRaft = dfTemporalCorrTetraBondOrientationOrderOfOneRaft.join(pd.Series(np.absolute(tempShelf['temporalCorrTetraBondOrientationOrder'][raftIDToLookAt,:]), name=columnName + '_abs'),how='outer')
            dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.join(pd.Series(np.real(tempShelf['temporalCorrHexaBondOrientationOrderAvgAllRafts']), name=columnName + '_real'),how='outer')
            dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.join(pd.Series(np.imag(tempShelf['temporalCorrHexaBondOrientationOrderAvgAllRafts']), name=columnName + '_imag'),how='outer')
            dfTemporalCorrHexaBondOrientationOrderAvgAllRafts = dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.join(pd.Series(np.absolute(tempShelf['temporalCorrHexaBondOrientationOrderAvgAllRafts']), name=columnName + '_abs'), how='outer')
            dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.join(pd.Series(np.real(tempShelf['temporalCorrPentaBondOrientationOrderAvgAllRafts']), name=columnName + '_real'),how='outer')
            dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.join(pd.Series(np.imag(tempShelf['temporalCorrPentaBondOrientationOrderAvgAllRafts']), name=columnName + '_imag'),how='outer')
            dfTemporalCorrPentaBondOrientationOrderAvgAllRafts = dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.join(pd.Series(np.absolute(tempShelf['temporalCorrPentaBondOrientationOrderAvgAllRafts']), name=columnName + '_abs'),how='outer')
            dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.join(pd.Series(np.real(tempShelf['temporalCorrTetraBondOrientationOrderAvgAllRafts']), name=columnName + '_real'), how='outer')
            dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.join(pd.Series(np.imag(tempShelf['temporalCorrTetraBondOrientationOrderAvgAllRafts']), name=columnName + '_imag'),how='outer')
            dfTemporalCorrTetraBondOrientationOrderAvgAllRafts = dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.join(pd.Series(np.absolute(tempShelf['temporalCorrTetraBondOrientationOrderAvgAllRafts']), name=columnName + '_abs'), how='outer')
            dfHexaBondOrientatiotionOrderModuliiAvgTime = dfHexaBondOrientatiotionOrderModuliiAvgTime.join(pd.Series(tempShelf['hexaticOrderParameterModuliiAvgs'], name = columnName + '_time'), how = 'outer' )
            dfHexaBondOrientatiotionOrderAvgNormTime = dfHexaBondOrientatiotionOrderAvgNormTime.join(pd.Series(tempShelf['hexaticOrderParameterAvgNorms'], name = columnName + '_time'), how = 'outer' )
            dfPentaBondOrientatiotionOrderModuliiAvgTime = dfPentaBondOrientatiotionOrderModuliiAvgTime.join(pd.Series(tempShelf['pentaticOrderParameterModuliiAvgs'], name = columnName + '_time'), how = 'outer' )
            dfPentaBondOrientatiotionOrderAvgNormTime = dfPentaBondOrientatiotionOrderAvgNormTime.join(pd.Series(tempShelf['pentaticOrderParameterAvgNorms'], name = columnName + '_time'), how = 'outer' )
            dfTetraBondOrientatiotionOrderModuliiAvgTime = dfTetraBondOrientatiotionOrderModuliiAvgTime.join(pd.Series(tempShelf['tetraticOrderParameterModuliiAvgs'], name = columnName + '_time'), how = 'outer' )
            dfTetraBondOrientatiotionOrderAvgNormTime = dfTetraBondOrientatiotionOrderAvgNormTime.join(pd.Series(tempShelf['tetraticOrderParameterAvgNorms'], name = columnName + '_time'), how = 'outer' )
            dfNeighborDistTime = dfNeighborDistTime.join(pd.Series(tempShelf['neighborDistanceAvgAllRafts'], name = columnName + '_time'), how = 'outer')
            dfEntropyNeighborDistTime = dfEntropyNeighborDistTime.join(pd.Series(tempShelf['entropyByNeighborDistances'], name = columnName + '_time'), how = 'outer')
            dfEntropyNeighborCountTime = dfEntropyNeighborCountTime.join(pd.Series(tempShelf['entropyByNeighborCount'], name = columnName + '_time'), how = 'outer')
            dfEntropyLocalDensitiesTime = dfEntropyLocalDensitiesTime.join(pd.Series(tempShelf['entropyByLocalDensities'], name = columnName + '_time'), how = 'outer')
            dfClusterSizeIncludingLonersTime = dfClusterSizeIncludingLonersTime.join(pd.Series(tempShelf['clusterSizeAvgIncludingLoners'], name = columnName + '_time'), how = 'outer')

        if analysisType == 3 or analysisType == 4:
            mutualInfoAvg = tempShelf['mutualInfoAvg']
            mutualInfoAvgExcludingSelfMI = tempShelf['mutualInfoAvgExcludingSelfMI']
            mutualInfoAvgSelfMIOnly = tempShelf['mutualInfoAvgSelfMIOnly']
            
            for ii in range(10):
                dfSummary.loc[dataID, 'mutualInfoAvg'+str(ii)] = mutualInfoAvg[ii]
                dfSummary.loc[dataID, 'mutualInfoAvgExcludingSelfMI'+str(ii)] = mutualInfoAvgExcludingSelfMI[ii]
                dfSummary.loc[dataID, 'mutualInfoAvgSelfMIOnly'+str(ii)] = mutualInfoAvgSelfMIOnly[ii]
        
        if analysisType == 5:
            # velocity unit conversion: (pixel/frame) * (frameRate frame/second) * (scaleBar um/pixel) = um/sec
            dfSummary.loc[dataID, 'raftKineticEnergiesSumAllRaftsAvgAllFrames'] = tempShelf['raftKineticEnergiesSumAllRafts'].mean() * (frameRate * scaleBar) ** 2
            dfSummary.loc[dataID, 'raftKineticEnergiesSumAllRaftsStdAllFrames'] = tempShelf['raftKineticEnergiesSumAllRafts'].std() * (frameRate * scaleBar) ** 2
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(pd.Series(mainDataList[dataID]['raftOrbitingDistances'].flatten(order='F') * scaleBar, name=columnName + '_orbitingDistances'),how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(pd.Series(tempShelf['raftVelocityXFiltered'].flatten(order='F') * (frameRate * scaleBar), name=columnName + '_VelocityX'),how='outer') # rafts in one frame upon rafts in another frame
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(pd.Series(tempShelf['raftVelocityYFiltered'].flatten(order='F') * (frameRate * scaleBar), name=columnName + '_VelocityY'),how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(pd.Series(tempShelf['raftVelocityNormFiltered'].flatten(order='F') * (frameRate * scaleBar), name=columnName + '_VelocityNorm'),how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(pd.Series(tempShelf['raftRadialVelocity'].flatten(order='F') * (frameRate * scaleBar), name=columnName + '_RadialVelocity'),how='outer')
            dfRaftVelocitiesVsOrbitingDistances = dfRaftVelocitiesVsOrbitingDistances.join(pd.Series(tempShelf['raftTangentialVelocity'].flatten(order='F') * (frameRate * scaleBar), name=columnName + '_TangentialVelocity'),how='outer')
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(pd.Series(mainDataList[dataID]['raftLocations'][raftIDToLookAt, :, 0] * scaleBar, name=columnName + '_RaftX'),how='outer')
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(pd.Series((mainDataList[dataID]['currentFrameGray'].shape[1] - mainDataList[dataID]['raftLocations'][raftIDToLookAt,:,1]) * scaleBar, name=columnName + '_RaftY'),how='outer') # make the Y values from bottom to top
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(pd.Series(tempShelf['particleMSD'][raftIDToLookAt, :] * scaleBar**2, name=columnName + '_MSD'),how='outer')
            dfRaftXYAndMSD = dfRaftXYAndMSD.join(pd.Series(tempShelf['particleMSDstd'][raftIDToLookAt, :] * scaleBar**2, name=columnName + '_MSDstd'),how='outer')
    
        tempShelf.close()
    else:
        print('missing data file: ' + shelveDataFileName)

dfSummaryConverted = dfSummary.infer_objects()
dfSummarySorted = dfSummaryConverted.sort_values(by = ['batchNum','spinSpeed'], ascending = [True, False])

#csvColNames = ['batchNum','spinSpeed',
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
dfSummarySorted.to_csv(dataFileName + '_summary.csv', index = False, columns = summaryDataFrameColNames )

dfRadialDistributionFunction.to_csv(dataFileName + '_gr-frame{}.csv'.format(frameNumToLookAt), index = False)
dfSpatialCorrHexaOrderPara.to_csv(dataFileName + '_g6r-frame{}.csv'.format(frameNumToLookAt), index = False)
dfSpatialCorrPentaOrderPara.to_csv(dataFileName + '_g5r-frame{}.csv'.format(frameNumToLookAt), index = False)
dfSpatialCorrTetraOrderPara.to_csv(dataFileName + '_g4r-frame{}.csv'.format(frameNumToLookAt), index = False)
dfSpatialCorrHexaBondOrientationOrder.to_csv(dataFileName + '_g6r-over-gr-frame{}.csv'.format(frameNumToLookAt), index = False)
dfSpatialCorrPentaBondOrientationOrder.to_csv(dataFileName + '_g5r-over-gr-frame{}.csv'.format(frameNumToLookAt), index = False)
dfSpatialCorrTetraBondOrientationOrder.to_csv(dataFileName + '_g4r-over-gr-frame{}.csv'.format(frameNumToLookAt), index = False)

dfTemporalCorrHexaBondOrientationOrderOfOneRaft.to_csv(dataFileName + '_g6t-raft{}.csv'.format(raftIDToLookAt), index = False)
dfTemporalCorrPentaBondOrientationOrderOfOneRaft.to_csv(dataFileName + '_g5t-raft{}.csv'.format(raftIDToLookAt), index = False)
dfTemporalCorrTetraBondOrientationOrderOfOneRaft.to_csv(dataFileName + '_g4t-raft{}.csv'.format(raftIDToLookAt), index = False)
dfTemporalCorrHexaBondOrientationOrderAvgAllRafts.to_csv(dataFileName + '_g6t-avgAllRafts.csv', index = False)
dfTemporalCorrPentaBondOrientationOrderAvgAllRafts.to_csv(dataFileName + '_g5t-avgAllRafts.csv', index = False)
dfTemporalCorrTetraBondOrientationOrderAvgAllRafts.to_csv(dataFileName + '_g4t-avgAllRafts.csv', index = False)

dfRaftVelocitiesVsOrbitingDistances.to_csv(dataFileName + '_velocitiesVsOrbitingDistances.csv', index = False)
dfRaftXYAndMSD.to_csv(dataFileName + '_raftXYAndMSD-raft{}.csv'.format(raftIDToLookAt), index = False)

dfHexaBondOrientatiotionOrderModuliiAvgTime.to_csv(dataFileName + '_psi6ModuliiAvg-time.csv', index = False)
dfHexaBondOrientatiotionOrderAvgNormTime.to_csv(dataFileName + '_psi6AvgNorm-time.csv', index = False)
dfPentaBondOrientatiotionOrderModuliiAvgTime.to_csv(dataFileName + '_psi5ModuliiAvg-time.csv', index = False)
dfPentaBondOrientatiotionOrderAvgNormTime.to_csv(dataFileName + '_psi5AvgNorm-time.csv', index = False)
dfTetraBondOrientatiotionOrderModuliiAvgTime.to_csv(dataFileName + '_psi4ModuliiAvg-time.csv', index = False)
dfTetraBondOrientatiotionOrderAvgNormTime.to_csv(dataFileName + '_psi4AvgNorm-time.csv', index = False)

dfNeighborDistTime.to_csv(dataFileName + 'neighborDist-time.csv', index = False)
dfEntropyNeighborDistTime.to_csv(dataFileName + '_entropybyNeighborDist-time.csv', index = False)
dfEntropyNeighborCountTime.to_csv(dataFileName + '_entropybyNeighborCount-time.csv', index = False)
dfEntropyLocalDensitiesTime.to_csv(dataFileName + '_entropybyLocalDensities-time.csv', index = False)
dfClusterSizeIncludingLonersTime.to_csv(dataFileName + '_clustersizeincludingloners-time.csv', index = False)



#%% load data corresponding to a specific experiment (subfolder or video) into variables

dataID = 0

variableListFromProcessedFile = list(mainDataList[dataID].keys())

for key, value in mainDataList[dataID].items(): # loop through key-value pairs of python dictionary
    globals()[key] = value

outputDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_' + commentsSub 


#%% load all variables from postprocessed file corresponding to the specific experiment above

analysisType = 2 # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi

shelveDataFileName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_' + 'postprocessed' + str(analysisType)

shelveDataFileExist = glob.glob(shelveDataFileName+'.dat')

if shelveDataFileExist:
    print(shelveDataFileName + ' exists, load additional variables. ' )
    tempShelf = shelve.open(shelveDataFileName)
    variableListFromPostProcessedFile = list(tempShelf.keys())
    
    for key in tempShelf: # just loop through all the keys in the dictionary
        globals()[key] = tempShelf[key]
    
    tempShelf.close()
    print('loading complete.' )
    
elif len(shelveDataFileExist) == 0:
    print(shelveDataFileName + ' does not exist')


#%% extracting entropies of various window sizes and plotting
dfEntropyNeighborDistTimeByWindowSizes = pd.DataFrame(columns = ['timeDifferenceInSeconds'])
dfEntropyNeighborCountTimeByWindowSizes = pd.DataFrame(columns = ['timeDifferenceInSeconds'])
dfEntropyLocalDensitiesTimeByWindowSizes = pd.DataFrame(columns = ['timeDifferenceInSeconds'])
dfEntropyByWindowSizesSummary = pd.DataFrame(columns = ['windowSizes', 'entropyNeighborDist_mean', 'entropyNeighborDist_std', 
                                                        'entropyNeighborCount_mean', 'entropyNeighborCount_std',
                                                        'entropyLocalDensities_mean', 'entropyLocalDensities_std'])

frameRate = 75 # unit fps, check this every time
timeDifferenceArray = np.arange(numOfFrames) / frameRate #unit: s
dfEntropyNeighborDistTimeByWindowSizes['timeDifferenceInSeconds'] = timeDifferenceArray
dfEntropyNeighborCountTimeByWindowSizes['timeDifferenceInSeconds'] = timeDifferenceArray
dfEntropyLocalDensitiesTimeByWindowSizes['timeDifferenceInSeconds'] = timeDifferenceArray

dfEntropyByWindowSizesSummary['windowSizes'] = samplingWindowSizes

for windowID, samplingWindowRadius in enumerate(samplingWindowSizes):
    columnName = 'WindowSize_' + str(samplingWindowRadius) + 'R'
    dfEntropyNeighborDistTimeByWindowSizes[columnName] = entropyByNeighborDistancesInWindows[:, windowID]
    dfEntropyNeighborCountTimeByWindowSizes[columnName] = entropyByNeighborCountInWindows[:, windowID]
    dfEntropyLocalDensitiesTimeByWindowSizes[columnName] = entropyByLocalDensitiesInWindows[:, windowID]
    dfEntropyByWindowSizesSummary.loc[windowID, 'entropyNeighborDist_mean'] = entropyByNeighborDistancesInWindows[:, windowID].mean()
    dfEntropyByWindowSizesSummary.loc[windowID, 'entropyNeighborDist_std'] = entropyByNeighborDistancesInWindows[:, windowID].std()
    dfEntropyByWindowSizesSummary.loc[windowID, 'entropyNeighborCount_mean'] = entropyByNeighborCountInWindows[:, windowID].mean()
    dfEntropyByWindowSizesSummary.loc[windowID, 'entropyNeighborCount_std'] = entropyByNeighborCountInWindows[:, windowID].std()
    dfEntropyByWindowSizesSummary.loc[windowID, 'entropyLocalDensities_mean'] = entropyByLocalDensitiesInWindows[:, windowID].mean()
    dfEntropyByWindowSizesSummary.loc[windowID, 'entropyLocalDensities_std'] = entropyByLocalDensitiesInWindows[:, windowID].std()


dfEntropyByWindowSizesSummary = dfEntropyByWindowSizesSummary.infer_objects()
dfEntropyNeighborDistTimeByWindowSizes = dfEntropyNeighborDistTimeByWindowSizes.infer_objects()
dfEntropyNeighborCountTimeByWindowSizes = dfEntropyNeighborCountTimeByWindowSizes.infer_objects()
dfEntropyLocalDensitiesTimeByWindowSizes = dfEntropyLocalDensitiesTimeByWindowSizes.infer_objects()


dataFileName = shelveDataFileName
dfEntropyByWindowSizesSummary.to_csv(dataFileName + '_summary.csv', index = False)
dfEntropyNeighborDistTimeByWindowSizes.to_csv(dataFileName + '_neighborDist.csv',index = False)
dfEntropyNeighborCountTimeByWindowSizes.to_csv(dataFileName + '_neighborCount.csv',index = False)
dfEntropyLocalDensitiesTimeByWindowSizes.to_csv(dataFileName + '_localDensities.csv',index = False)


dfEntropyByWindowSizesSummary.plot(x = 'windowSizes', y = 'entropyNeighborDist_mean')
dfEntropyByWindowSizesSummary.plot(x = 'windowSizes', y = 'entropyNeighborDist_std')
dfEntropyByWindowSizesSummary.plot(x = 'windowSizes', y = 'entropyNeighborCount_mean')
dfEntropyByWindowSizesSummary.plot(x = 'windowSizes', y = 'entropyNeighborCount_std')
dfEntropyByWindowSizesSummary.plot(x = 'windowSizes', y = 'entropyLocalDensities_mean')
dfEntropyByWindowSizesSummary.plot(x = 'windowSizes', y = 'entropyLocalDensities_std')



#%%Kinetic Energy calculation (Gaurav)
raftlocx=raftLocations[:,::5,0]
raftlocy=raftLocations[:,::5,1]
raftvelx=np.gradient(raftlocx,axis=1)
raftvely=np.gradient(raftlocy,axis=1)
raftVelnormxy = np.square(raftvely) + np.square(raftvelx)
mean_vel = raftVelnormxy.mean(axis=0)
mean_vel_fps = mean_vel.mean()
energy_fps = mean_vel_fps + (raftRadii.mean()**2)*(2*np.pi*spinSpeed*5)**2 

#%% velocity and MSD analysis, velocity as a function of radial distance from COM
# SSA parameters 
embeddingDimension = 20
reconstructionComponents = np.arange(5)
            
raftLocationsX = raftLocations[:,:,0]
raftLocationsY = raftLocations[:,:,1]

raftVelocityX = np.gradient(raftLocationsX, axis = 1) # unit pixel/frame
raftVelocityY = np.gradient(raftLocationsY, axis = 1) # unit pixel/frame
raftVelocityNorm = np.sqrt(raftVelocityX ** 2 + raftVelocityY ** 2)

raftVelocityXFiltered = np.zeros_like(raftVelocityX)
raftVelocityYFiltered = np.zeros_like(raftVelocityY)
for raftID in np.arange(numOfRafts):
    raftVelocityXFiltered[raftID,:] = SSAFull(raftVelocityX[raftID,:], embeddingDimension, reconstructionComponents)
    raftVelocityYFiltered[raftID,:] = SSAFull(raftVelocityY[raftID,:], embeddingDimension, reconstructionComponents)

raftVelocityNormFiltered = np.sqrt(raftVelocityXFiltered ** 2 + raftVelocityYFiltered ** 2)

raftKineticEnergies = raftVelocityNormFiltered ** 2
raftKineticEnergiesSumAllRafts = raftKineticEnergies.sum(axis = 0)

# get the radial and tangential vectors
raftOrbitingCentersXBroadcasted  = np.broadcast_to(raftOrbitingCenters[:,0], raftLocationsX.shape)
raftOrbitingCentersYBroadcasted  = np.broadcast_to(raftOrbitingCenters[:,1], raftLocationsY.shape)
raftRadialVectorX = raftLocationsX - raftOrbitingCentersXBroadcasted
raftRadialVectorY = raftLocationsY - raftOrbitingCentersYBroadcasted
raftRadialVectorXUnitized = raftRadialVectorX / np.sqrt(raftRadialVectorX ** 2 + raftRadialVectorY ** 2)
raftRadialVectorYUnitized = raftRadialVectorY / np.sqrt(raftRadialVectorX ** 2 + raftRadialVectorY ** 2)
raftTangentialVectorXUnitized = -raftRadialVectorYUnitized # negative sign is assigned such that the tangential velocity is positive
raftTangentialVectorYUnitized =  raftRadialVectorXUnitized 
# get the radial and tangential velocities
raftRadialVelocity = raftVelocityXFiltered * raftRadialVectorXUnitized + raftVelocityYFiltered * raftRadialVectorYUnitized
raftTangentialVelocity = raftVelocityXFiltered * raftTangentialVectorXUnitized + raftVelocityYFiltered * raftTangentialVectorYUnitized

particleMSD = np.zeros((numOfRafts, numOfFrames))
particleMSDstd = np.zeros((numOfRafts, numOfFrames))
particleRMSD = np.zeros((numOfRafts, numOfFrames))
    
for raftID in np.arange(numOfRafts):
    corrX = raftLocations[raftID,:,0]
    corrY = raftLocations[raftID,:,1]
    
    # construct Toeplitz matrix, 1st column is the corrdinateX and Y, top right half all zeros
    corrXToeplitz = scipy.linalg.toeplitz(corrX, np.zeros(numOfFrames))
    corrYToeplitz = scipy.linalg.toeplitz(corrY, np.zeros(numOfFrames))
    
    # broad cast the column of coordinate x and y to the size of Toeplitz matrix
    corrXBroadcasted = np.transpose(np.broadcast_to(corrX, corrXToeplitz.shape))
    corrYBroadcasted = np.transpose(np.broadcast_to(corrY, corrYToeplitz.shape))
    
    # substrate Toeplitz matrix from broadcasted matrix,  
    # for each column, the rows on and below the diagonal are the displacement in x and y coordinates
    # step size is the column index. 
    corrXdiffMatrixSquared = (corrXBroadcasted - corrXToeplitz)**2
    corrYdiffMatrixSquared = (corrYBroadcasted - corrYToeplitz)**2
    particleSquareDisplacement = corrXdiffMatrixSquared + corrYdiffMatrixSquared
    
    # calculate mean square displacement
    for stepSize in np.arange(numOfFrames):
        particleMSD[raftID, stepSize] = np.average(particleSquareDisplacement[stepSize:,stepSize])
        particleMSDstd[raftID, stepSize] = np.std(particleSquareDisplacement[stepSize:,stepSize])

    particleRMSD[raftID, :] = np.sqrt(particleMSD[raftID, :])


# plot to see XY_MSD 
raftToLookAt = 50

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (18,9))
ax[0].plot(raftLocations[raftToLookAt,:,0], currentFrameGray.shape[1] - raftLocations[raftToLookAt,:,1])
ax[0].set_xlabel('Position x (pixel)')
ax[0].set_ylabel('Position y (pixel)')
ax[0].set_xlim([0, currentFrameGray.shape[0]])
ax[0].set_ylim([0, currentFrameGray.shape[1]])
ax[0].set_title('raft ID = {}'.format(raftToLookAt))
ax[1].errorbar(np.arange(numOfFrames), particleMSD[raftToLookAt,:], yerr = particleMSDstd[raftToLookAt,:], errorevery = 20)
ax[1].set_xlabel('step size')
ax[1].set_ylabel('Mean Square Displacement')
fig.show()


# plotting raft kinetic energy sums
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(np.arange(numOfFrames), raftKineticEnergiesSumAllRafts,'-o')
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('frame #',size=20)
ax.set_ylabel('kinetic energy sum over all rafts',size=20)
ax.set_title('kinetic energy sum over all rafts, assume mass = 2', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
#ax.legend()
fig.show()


# comparing before and after SSA: check embedding and reconstruction parameters
embeddingDimension = 20
reconstructionComponents = np.arange(5)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
colors = plt.cm.viridis(np.linspace(0,1,numOfRafts))
for i in range(0, numOfRafts, 90):
    ax.plot(np.arange(0, numOfFrames),raftVelocityX[i,:],label='before SSA {}'.format(i))
    ax.plot(np.arange(0, numOfFrames),SSAFull(raftVelocityX[i,:], embeddingDimension, reconstructionComponents),label='after SSA {}'.format(i))
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('Time (frame)',size=20)
ax.set_ylabel('raft velocity in x',size=20)
ax.set_title('raft velocity in x', size = 20)
ax.legend()
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# plotting tangential velocity  vs orbiting distances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(raftOrbitingDistances.flatten(),raftTangentialVelocity.flatten(),'o')
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('orbiting distance',size=20)
ax.set_ylabel('tangential velocity',size=20)
ax.set_title('tangential velocity vs orbiting distance', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plotting radial velocity  vs orbiting distances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftOrbitingDistances.flatten(),raftRadialVelocity.flatten(),'o')
    
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('orbiting distance',size=20)
ax.set_ylabel('radial velocity',size=20)
ax.set_title('radial velocity vs orbiting distance', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plot to check the direction of the tangential vector
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftRadialVectorXUnitized[0,0],raftRadialVectorYUnitized[0,0],'o', label = 'radial')
ax.plot(raftTangentialVectorXUnitized[0,0],raftTangentialVectorYUnitized[0,0],'o', label = 'tangential')
ax.plot(raftVelocityX[0,0],raftVelocityY[0,0],'o', label = 'velocity')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])  
ax.set_xlabel(' x',size=20)
ax.set_ylabel('y',size=20)
ax.set_title('test of tangential vector direction', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plot raft velocity x vs velocity y
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftVelocityXFiltered.flatten(),raftVelocityYFiltered.flatten(),'o')
    
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('raft velocity in x',size=20)
ax.set_ylabel('raft velocity in y',size=20)
ax.set_title('raft velocity in x', size = 20)
ax.legend()
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# plotting radial vector x and y
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftRadialVectorXUnitized.flatten(),raftRadialVectorYUnitized.flatten(),'o')
    
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('radial vector x',size=20)
ax.set_ylabel('radial vector y',size=20)
ax.set_title('radial vectors', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plotting velocity norm vs orbiting distances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(raftOrbitingDistances.flatten(),raftVelocityNormFiltered.flatten(),'o')
    
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('orbiting distance',size=20)
ax.set_ylabel('orbiting velocity norm',size=20)
ax.set_title('orbiting velocity vs orbiting distance', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()

# plot velocity in x direction
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0,1,numOfRafts))

for i in range(0, numOfRafts, 10):
    ax.plot(np.arange(0, numOfFrames),raftVelocityXFiltered[i,:],c=colors[i],label='{}'.format(i))
    
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('Time (frame)',size=20)
ax.set_ylabel('raft velocity in x',size=20)
ax.set_title('raft velocity in x', size = 20)
ax.legend()
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()


# plot the velocity in y direction
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0,1,numOfRafts))

for i in range(0, numOfRafts, 10):
    ax.plot(np.arange(0, numOfFrames),raftVelocityYFiltered[i,:],c=colors[i],label='{}'.format(i))
    
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('Time (frame)',size=20)
ax.set_ylabel('raft velocity in y',size=20)
ax.set_title('raft velocity in y', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()


# plot the velocity norm
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0,1,numOfRafts))

for i in range(0, numOfRafts, 10):
    ax.plot(np.arange(0, numOfFrames),raftVelocityNormFiltered[i,:],c=colors[i],label='{}'.format(i))
    
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('Time (frame)',size=20)
ax.set_ylabel('raft velocity norm',size=20)
ax.set_title('raft velocity norm', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()


#%% some simple plots just to look at the data for one specific experiment

# plot the center of mass
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(raftOrbitingCenters[:,0], currentFrameGray.shape[1] - raftOrbitingCenters[:,1])
fig.show()



# plot the center of mass, x and y coordinate separately
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.plot(raftOrbitingCenters[:,0], label = 'x')
ax.plot(raftOrbitingCenters[:,1], label = 'y')
ax.legend()
fig.show()

# plot orbiting distances vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0,1,numOfRafts))

for i in range(0, numOfRafts):
    ax.plot(np.arange(numOfFrames),raftOrbitingDistances[i,:],c=colors[i],label='{}'.format(i))
    
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, raftOrbitingDistances.max()])  
ax.set_xlabel('Time (frame)',size=20)
ax.set_ylabel('distance to center of mass',size=20)
ax.set_title('distance to center of mass, {} Rafts'.format(numOfRafts), size = 20)
ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()



#dfRaftOrbitingDistances = pd.DataFrame(np.transpose(raftOrbitingDistances))
#dfRaftOrbitingDistances.to_csv(outputDataFileName + '_distances.csv') 

# plot orbiting Angles vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.viridis(np.linspace(0,1,numOfRafts))

for i in range(0, numOfRafts):
    ax.plot(np.arange(numOfFrames),raftOrbitingAngles[i,:],'-', c=colors[i],label='{}'.format(i))
    
ax.set_xlim([0, numOfFrames])
ax.set_ylim([raftOrbitingAngles.min(), raftOrbitingAngles.max()])  
ax.set_xlabel('Frames(Time)',size=20)
ax.set_ylabel('raft orbiting angles',size=20)
ax.set_title('Raft orbiting angles, {} Rafts'.format(numOfRafts), size = 20)
ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
ax.legend()
fig.show()


#dfRaftOrbitingAngles= pd.DataFrame(np.transpose(raftOrbitingAngles))
#dfRaftOrbitingAngles.to_csv(outputDataFileName + '_angles.csv') 

#plt.close('all')

#%% diffusion data treatment; mainly uses data from raftLocations

for dataID in np.arange(18):
    numOfFrames = mainDataList[dataID]['numOfFrames']
    raftLocations = mainDataList[dataID]['raftLocations']
    currentFrameGray = mainDataList[dataID]['currentFrameGray']
    subfolderName = mainDataList[dataID]['subfolders'][mainDataList[dataID]['expID']]


    corrX = raftLocations[0,:,0]
    corrY = raftLocations[0,:,1]

    # construct Toeplitz matrix, 1st column is the corrdinateX and Y, top right half all zeros
    corrXToeplitz = scipy.linalg.toeplitz(corrX, np.zeros(numOfFrames))
    corrYToeplitz = scipy.linalg.toeplitz(corrY, np.zeros(numOfFrames))

    # broad cast the column of coordinate x and y to the size of Toeplitz matrix
    corrXBroadcasted = np.transpose(np.broadcast_to(corrX, corrXToeplitz.shape))
    corrYBroadcasted = np.transpose(np.broadcast_to(corrY, corrYToeplitz.shape))

    # substrate Toeplitz matrix from broadcasted matrix,  
    # for each column, the rows on and below the diagonal are the displacement in x and y coordinates
    # step size is the column index. 
    corrXdiffMatrixSquared = (corrXBroadcasted - corrXToeplitz)**2
    corrYdiffMatrixSquared = (corrYBroadcasted - corrYToeplitz)**2
    particleSquareDisplacement = corrXdiffMatrixSquared + corrYdiffMatrixSquared

    particleMSD = np.zeros(numOfFrames)
    particleMSDstd = np.zeros(numOfFrames)
    # calculate mean square displacement
    for stepSize in np.arange(numOfFrames):
        particleMSD[stepSize] = np.average(particleSquareDisplacement[stepSize:,stepSize])
        particleMSDstd[stepSize] = np.std(particleSquareDisplacement[stepSize:,stepSize])


    particleRMSD = np.sqrt(particleMSD)

    diffusionDataFrame = pd.DataFrame(columns = ['StepSize', 'particleMSD', 'particleMSDstd',
                                                 'particleRMSD', 'frameNum','corrX', 'corrY'])
    diffusionDataFrame['StepSize'] = np.arange(numOfFrames)
    diffusionDataFrame['particleMSD'] = particleMSD
    diffusionDataFrame['particleMSDstd'] = particleMSDstd
    diffusionDataFrame['particleRMSD'] = particleRMSD
    diffusionDataFrame['frameNum'] = np.arange(numOfFrames)
    diffusionDataFrame['corrX'] = corrX
    diffusionDataFrame['corrY'] = corrY
    
    
#    diffusionDataFrame.to_csv(subfolderName + '_diffusion.csv', index = False) 
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (18,9))
    ax[0].plot(corrX, currentFrameGray.shape[1] - corrY)
    ax[0].set_xlabel('Position x (pixel)')
    ax[0].set_ylabel('Position y (pixel)')
    ax[0].set_xlim([0, currentFrameGray.shape[0]])
    ax[0].set_ylim([0, currentFrameGray.shape[1]])
    ax[1].errorbar(np.arange(numOfFrames), particleMSD, yerr = particleMSDstd, errorevery = 20)
    ax[1].set_xlabel('step size')
    ax[1].set_ylabel('Mean Square Displacement')
    fig.savefig(subfolderName + '_XY_MSD.png')

#plt.show()
plt.close('all')

#%% region search data treatment for effusion experiments only, mostly using raftLocationsInRegion

startingFrameNum = 0 # check raftsEffused, and pick the frame number when the first raft is effused (assume effusing state)
maxDisplacement = 36 # check mainDataList[dataID]['maxDisplacement'] to see if it exists.
radius = 15.5 #raftRadiiInRegion[np.nonzero(raftRadiiInRegion)].mean() # pixel
scaleBar = 300 / radius /2 # micron per pixel
frameRate = 30 # frame per second

if 'raftLocationsInRegion' in mainDataList[dataID]: # dataID is assigned in the section where the data from processed file is loaded
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


dfRegionSearch = pd.DataFrame(columns = ['passingRaftCount','enteringFrameNum','exitingFrameNum',
                                         'rowIndices', 'positionsXY', 
                                         'avgYsInPixel', 'avgSpeedInXInPixel', 'avgSpeedInYInPixel',
                                         'avgYsInMicron', 'avgSpeedInXInMicronPerSec','avgSpeedInYInMicronPerSec'])


passingRaftCount = 0
numOfRaftsInPrevFrame = np.count_nonzero(raftLocationsInRegion[:,startingFrameNum,0])
for prevRaftNum in range(0,numOfRaftsInPrevFrame):
    passingRaftCount = passingRaftCount + 1
    dfRegionSearch.loc[passingRaftCount, 'passingRaftCount'] = passingRaftCount
    dfRegionSearch.loc[passingRaftCount, 'enteringFrameNum'] = startingFrameNum
    dfRegionSearch.loc[passingRaftCount, 'exitingFrameNum'] = startingFrameNum
    dfRegionSearch.loc[passingRaftCount, 'rowIndices'] = [prevRaftNum] # make it a list
    dfRegionSearch.loc[passingRaftCount, 'positionsXY'] = [raftLocationsInRegion[prevRaftNum,startingFrameNum,:]]
    
    
    

for currentFrameNum in progressbar.progressbar(range(startingFrameNum+1,numOfFrames)):
    numOfRaftsInCurrFrame = np.count_nonzero(raftLocationsInRegion[:,currentFrameNum,0])
    # in raftPairwiseDistances, rows - prevRaftNum; columns - currRaftNum 
    raftPairwiseDistances = scipyDistance.cdist(raftLocationsInRegion[:numOfRaftsInPrevFrame,currentFrameNum-1,:], raftLocationsInRegion[:numOfRaftsInCurrFrame,currentFrameNum,:], 'euclidean')
    
    #loop over all currRaftNum. It necessitates looking for the corresponding raft in the previous frame
    if numOfRaftsInCurrFrame > 0 and numOfRaftsInPrevFrame > 0: # otherwise raftPairwiseDistances[:,currRaftNum].min() gives error
        for currRaftNum in range(0,numOfRaftsInCurrFrame):
            if raftPairwiseDistances[:,currRaftNum].min() < maxDisplacement:
                # this is an old raft, get its prevRaftNum from raftPairwiseDistances
                prevRowNum = np.nonzero(raftPairwiseDistances[:,currRaftNum] == raftPairwiseDistances[:,currRaftNum].min())[0][0] # [0][0] just to remove array
                # use rowNumsSeries to loop over raftIndex and and rowNumsList to look for corresponding raft in the previous frame
                # note that the raftIndex in rowNumsSeries is the same as in dfRegionSearch
                rowNumsSeries = dfRegionSearch[dfRegionSearch.exitingFrameNum == currentFrameNum-1]['rowIndices']
                for raftIndex, rowNumsList in rowNumsSeries.iteritems():
                    # for the correct raft, the last entry of the rowNumsList should be the prevRowNum
                    if rowNumsList[-1] == prevRowNum: 
                        dfRegionSearch.loc[raftIndex, 'rowIndices'].append(currRaftNum)
                        dfRegionSearch.loc[raftIndex, 'positionsXY'].append(raftLocationsInRegion[currRaftNum,currentFrameNum,:])
                        dfRegionSearch.loc[raftIndex, 'exitingFrameNum'] = currentFrameNum
            else:
                # this is a new raft, add it into the dfRegionSearch
                passingRaftCount = passingRaftCount + 1
                dfRegionSearch.loc[passingRaftCount, 'passingRaftCount'] = passingRaftCount
                dfRegionSearch.loc[passingRaftCount, 'enteringFrameNum'] = currentFrameNum
                dfRegionSearch.loc[passingRaftCount, 'exitingFrameNum'] = currentFrameNum
                dfRegionSearch.loc[passingRaftCount, 'rowIndices'] = [currRaftNum] # make it a list
                dfRegionSearch.loc[passingRaftCount, 'positionsXY'] = [raftLocationsInRegion[currRaftNum,currentFrameNum,:]]
    
    # reset numOfRaftsInPrevFrame
    numOfRaftsInPrevFrame = numOfRaftsInCurrFrame
        
# loop over all raftIndex to fill the dfRegionSearch
positionsXYListSeries = dfRegionSearch.positionsXY

for raftIndex, positionsXYList in positionsXYListSeries.iteritems():
    positionXYArray = np.array(positionsXYList)
    if positionXYArray[:,0].size > 1: # rafts that show up for at least two frame 
        avgYsInPixel = positionXYArray[:,1].mean()
        avgSpeedInXInPixel = (positionXYArray[0,0] - positionXYArray[-1,0]) / (positionXYArray[:,0].size - 1) # unit is pixel per frame
        avgSpeedInYInPixel = (positionXYArray[0,1] - positionXYArray[-1,1]) / (positionXYArray[:,1].size - 1)
        
        dfRegionSearch.loc[raftIndex, 'avgYsInPixel'] = avgYsInPixel
        dfRegionSearch.loc[raftIndex, 'avgYsInMicron'] = avgYsInPixel * scaleBar
        dfRegionSearch.loc[raftIndex, 'avgSpeedInXInPixel'] = avgSpeedInXInPixel
        dfRegionSearch.loc[raftIndex, 'avgSpeedInXInMicronPerSec'] = avgSpeedInXInPixel * scaleBar * frameRate
        dfRegionSearch.loc[raftIndex, 'avgSpeedInYInPixel'] = avgSpeedInYInPixel
        dfRegionSearch.loc[raftIndex, 'avgSpeedInYInMicronPerSec'] = avgSpeedInYInPixel * scaleBar * frameRate

avgYsInPixelSeries = dfRegionSearch.avgYsInPixel
avgYsInPixelArray = np.array(avgYsInPixelSeries[avgYsInPixelSeries == avgYsInPixelSeries].tolist()) # using the fact that np.nan != np.nan to remove nan; https://stackoverflow.com/questions/20235401/remove-nan-from-pandas-series
avgSpeedInXInPixelSeries = dfRegionSearch.avgSpeedInXInPixel
avgSpeedInXInPixelArray = np.array(avgSpeedInXInPixelSeries[avgSpeedInXInPixelSeries == avgSpeedInXInPixelSeries].tolist())
avgSpeedInYInPixelSeries = dfRegionSearch.avgSpeedInYInPixel
avgSpeedInYInPixelArray = np.array(avgSpeedInYInPixelSeries[avgSpeedInYInPixelSeries == avgSpeedInYInPixelSeries].tolist())

 
# average speeds in X direction vs average Y positions
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(avgYsInPixelArray,avgSpeedInXInPixelArray, 'o')
ax.legend(loc='best')
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, clusters[raftNum, 1, :].max()])  
ax.set_xlabel('averge of Y positions',size=20)
ax.set_ylabel('average speed in x direction',size=20)
ax.set_title('average speeds in X direction vs average Y positions', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()


# average speeds in y direction vs average Y positions
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.plot(avgYsInPixelArray,avgSpeedInYInPixelArray, 'o')
ax.legend(loc='best')
#ax.set_xlim([0, numOfFrames])
#ax.set_ylim([0, clusters[raftNum, 1, :].max()])  
ax.set_xlabel('averge of Y positions',size=20)
ax.set_ylabel('average speed in y direction',size=20)
ax.set_title('average speeds in y direction vs average Y positions', size = 20)
#ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()



csvColNamesRegionSearch = ['passingRaftCount','enteringFrameNum','exitingFrameNum',
                           'avgYsInPixel', 'avgSpeedInXInPixel', 'avgSpeedInYInPixel',
                           'avgYsInMicron', 'avgSpeedInXInMicronPerSec','avgSpeedInYInMicronPerSec']

outputDataFileNameRegionalCSV = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + \
                                'rps_' + str(magnification) + 'x_' + commentsSub + '_RegionalSearch.csv'
                                
dfRegionSearch.to_csv(outputDataFileNameRegionalCSV , index = False, columns = csvColNamesRegionSearch )



#%% drawing for regional search - for effusion experiments only

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
    outputVideoName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_RegionalSearch.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName,fourcc, outputFrameRate, (frameH, frameW), 1)

for currentFrameNum in progressbar.progressbar(range(len(tiffFileList))): # range(len(tiffFileList))
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    numOfRaftsInCurrFrame = np.count_nonzero(raftLocationsInRegion[:,currentFrameNum,0])
    currentFrameDraw = DrawRafts(currentFrameDraw, raftLocationsInRegion[:,currentFrameNum,:], raftRadiiInRegion[:,currentFrameNum], numOfRaftsInCurrFrame)
    currentFrameDraw = cv.rectangle(currentFrameDraw, (regionTopLeftX, regionTopLeftY), (regionTopLeftX + regionWidth, regionTopLeftY + regionHeight), (0,0,255), 2)
    
    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_RegionalSearch_' + str(currentFrameNum+1).zfill(4) + '.jpg'
        cv.imwrite(outputImageName,currentFrameDraw)
    if outputVideo == 1:
        videoOut.write(currentFrameDraw)

if outputVideo == 1:
    videoOut.release()
    
plt.figure()
plt.imshow(currentFrameDraw[:,:,::-1])

#%% clucster analysis
radius = raftRadii.mean() #pixel  check raftRadii.mean()
scaleBar = 300 / radius /2 # micron per pixel

raftPairwiseDistances = np.zeros((numOfRafts, numOfRafts, numOfFrames))
raftPairwiseEdgeEdgeDistancesSmallest = np.zeros((numOfRafts, numOfFrames))
raftPairwiseDistancesInRadius = np.zeros((numOfRafts, numOfRafts, numOfFrames))
raftPairwiseConnectivity = np.zeros((numOfRafts, numOfRafts, numOfFrames))

# using scipy distance module
t1 = time.perf_counter()
for frameNum in np.arange(numOfFrames):
    raftPairwiseDistances[:,:,frameNum] = scipyDistance.cdist(raftLocations[:,frameNum,:], raftLocations[:,frameNum,:], 'euclidean')
    # smallest nonzero eedistances is assigned to one raft as the pairwise distance, connected rafts will be set to 0 later
    raftPairwiseEdgeEdgeDistancesSmallest[:,frameNum] = np.partition(raftPairwiseDistances[:,:,frameNum], 1, axis = 1)[:,1] - radius *2
    
t2 = time.perf_counter()
timeTotal = t2 - t1 # in seconds
print(timeTotal)

raftPairwiseDistancesInRadius = raftPairwiseDistances / radius


# plot the histogram of pairwise distance in radius to look at the selection
# of radius value for thresholding connectivity
frameNumToLookAt = 0
raftPairwiseDistancesInRadius_oneFrame = raftPairwiseDistancesInRadius[:,:,frameNumToLookAt]
binsForPairwiseDisttances = np.arange(0,5,0.1)
count, edges = np.histogram(raftPairwiseDistancesInRadius_oneFrame,bins=binsForPairwiseDisttances)

fig, ax = plt.subplots(1,1, figsize = (20,10))
ax.bar(edges[:-1], count, align = 'edge', width = 0.05)
ax.set_xlabel('pairwise distances',  {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of pairwise distances of frame {}'.format(frameNumToLookAt), {'size': 15})
ax.legend(['pairwise distances'])
fig.show()

# re-adjust connectivity thresholding if necessary
# Caution: this way of determing clusters produces errors, mostly false positive. 
connectivityThreshold = 2.3 # unit: radius
# re-thresholding the connectivity matrix. 
# Note that the diagonal self-distance is zero, and needs to be taken care of seperately
raftPairwiseConnectivity = np.logical_and((raftPairwiseDistancesInRadius < connectivityThreshold), (raftPairwiseDistancesInRadius > 0)) *1

# to correct false positive, if the rafts are not connected in the next frame, 
# then it is not connected in the present frame
for currentFrameNum in range(numOfFrames-1):
    raftAs, raftBs = np.nonzero(raftPairwiseConnectivity[:,:,currentFrameNum])
    for raftA, raftB in zip(raftAs, raftBs):
        if raftPairwiseConnectivity[raftA, raftB,currentFrameNum+1] == 0:
            raftPairwiseConnectivity[raftA, raftB,currentFrameNum] = 0
            
# information about clusters in all frames. For reach frame, the array has two columns, 
# 1st col: cluster number, 2nd col: cluster size (excluding loners)
clusters = np.zeros((numOfRafts, 2, numOfFrames))
# clusterSizeCounts stores the number of clusters of each size for all frames. 
# the first index is used as the size of the cluster
clusterSizeCounts = np.zeros((numOfRafts+1,numOfFrames)) 

# fill in clusters matrix
t1 = time.perf_counter()
for frameNum in np.arange(numOfFrames):
    clusterNum = 1
    raftAs, raftBs = np.nonzero(raftPairwiseConnectivity[:,:,frameNum])
    # determine the cluster number and store the cluster number in the first column
    for raftA, raftB in zip(raftAs, raftBs):
        # to see if A and B are already registered in the raftsInClusters
        raftsInClusters = np.nonzero(clusters[:,0,frameNum])
        A = any(raftA in raft for raft  in raftsInClusters)
        B = any(raftB in raft for raft  in raftsInClusters)
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
        # if neigher is new and if their cluster numbers differ, then change the larger cluster number to the smaller one
        # note that this could lead to a cluster number being jumped over
        if (A == True) and (B == True) and (clusters[raftA, 0, frameNum] != clusters[raftB, 0, frameNum]):
            clusterNumLarge = max(clusters[raftA,0, frameNum], clusters[raftB,0, frameNum])
            clusterNumSmall = min(clusters[raftA,0, frameNum], clusters[raftB,0, frameNum])
            clusters[clusters[:,0, frameNum] == clusterNumLarge,0, frameNum] = clusterNumSmall 
    # Count the number of rafts in each cluster and store the cluster size in the second column
    numOfClusters = clusters[:,0,frameNum].max()
    if numOfClusters > 0:
        for clusterNum in np.arange(1, numOfClusters+1):
            clusterSize = len(clusters[clusters[:,0,frameNum] == clusterNum,0,frameNum])
            clusters[clusters[:,0,frameNum] == clusterNum,1,frameNum] = clusterSize
    raftPairwiseEdgeEdgeDistancesSmallest[np.nonzero(clusters[:,0,frameNum]),frameNum] = 0
t2 = time.perf_counter()
timeTotal = t2 - t1 # in seconds
print(timeTotal)


# fill in clusterSizeCounts matrix        
t1 = time.perf_counter()
for frameNum in np.arange(numOfFrames):
    largestClusterSize = clusters[:,1,frameNum].max()
    # count loners
    numOfLoners = len(clusters[clusters[:,1,frameNum] == 0,1,frameNum])
    clusterSizeCounts[1,frameNum] = numOfLoners
    # for the rest, the number of occurrence of cluster size in the 2nd column is the cluster size times the number of clusters of that size
    for clusterSize in np.arange(2, largestClusterSize+1):
        numOfClusters = len(clusters[clusters[:,1,frameNum] == clusterSize,1,frameNum])/clusterSize
        clusterSizeCounts[int(clusterSize),frameNum] = numOfClusters

t2 = time.perf_counter()
timeTotal = t2 - t1 # in seconds
print(timeTotal)


# some averageing
dummyArray = np.arange((numOfRafts + 1) * numOfFrames).reshape((numOfFrames,-1)).T
dummyArray = np.mod(dummyArray, (numOfRafts + 1)) # rows are cluster sizes, and columns are frame numbers
clusterSizeAvgIncludingLoners = np.average(dummyArray, axis = 0, weights = clusterSizeCounts)
clusterSizeAvgIncludingLonersAllFrames = clusterSizeAvgIncludingLoners.mean()
print('clusterSizeAvgIncludingLonersAllFrames = {:.4}'.format(clusterSizeAvgIncludingLonersAllFrames))

clusterSizeCountsExcludingLoners = clusterSizeCounts.copy()
clusterSizeCountsExcludingLoners[1,:] = 0

clusterSizeAvgExcludingLoners, sumOfWeights = np.ma.average(dummyArray, axis = 0, weights = clusterSizeCountsExcludingLoners, returned = True)
clusterSizeAvgExcludingLonersAllFrames = clusterSizeAvgExcludingLoners.mean()
print('clusterSizeAvgExcludingLonersAllFrames = {:.4} '.format(clusterSizeAvgExcludingLonersAllFrames))


raftPairwiseEdgeEdgeDistancesSmallestMean = raftPairwiseEdgeEdgeDistancesSmallest.mean() * scaleBar
raftPairwiseEdgeEdgeDistancesSmallestStd = raftPairwiseEdgeEdgeDistancesSmallest.std() * scaleBar
numOfLonersAvgAllFrames = clusterSizeCounts[1,:].mean()
print('raftPairwiseEdgeEdgeDistancesSmallestMean = {:.3} micron'.format(raftPairwiseEdgeEdgeDistancesSmallestMean))
print('raftPairwiseEdgeEdgeDistancesSmallestStd = {:.3} micron'.format(raftPairwiseEdgeEdgeDistancesSmallestStd))
print('average number of loners = {:.3}'.format(numOfLonersAvgAllFrames))

#%% some plots to look at pairwise data and cluster information. 

# plot pairwise distance to a specific raft vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

colors = plt.cm.jet(np.linspace(0,1,numOfRafts))

raft1Num = 0

for raft2Num in range(0, numOfRafts):
    ax.plot(np.arange(numOfFrames),raftPairwiseDistancesInRadius[raft1Num, raft2Num, :],c=colors[raft2Num],label='{}'.format(raft2Num))
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, raftPairwiseDistancesInRadius[raft1Num, :, :].max()])  
ax.set_xlabel('Frames(Time)',size=20)
ax.set_ylabel('distance to raft {}'.format(raft1Num),size=20)
ax.set_title('distance to raft {}, {} Rafts'.format(raft1Num, numOfRafts), size = 20)
ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# plot the size of the cluster one specific raft belongs to vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

raftNum = 2

ax.plot(np.arange(numOfFrames),clusters[raftNum,1,:])
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusters[raftNum, 1, :].max()])  
ax.set_xlabel('Frames(Time)',size=20)
ax.set_ylabel('cluster size',size=20)
ax.set_title('the size of the cluster that include raft {}'.format(raftNum), size = 20)
ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()


# plot the number of clusters  vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
colors = plt.cm.jet(np.linspace(0,1,numOfRafts))

ax.plot(np.arange(numOfFrames),np.count_nonzero(clusterSizeCounts, axis = 0),label='num of clusters')
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusters[:, 0, :].max()+0.5])  
ax.set_xlabel('Frames(Time)',size=20)
ax.set_ylabel('cluster number',size=20)
ax.set_title('cluster number', size = 20)
ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()

# plot the number of clusters with 2, 3, 4, ...  rafts vs frame number
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

rows, _ = np.nonzero(clusterSizeCounts)
maxRaftsInACluster = rows.max()

colors = plt.cm.jet(np.linspace(0,1,maxRaftsInACluster+1))

for numOfRaftInACluster in range(1, maxRaftsInACluster+1):
    ax.plot(np.arange(numOfFrames),clusterSizeCounts[numOfRaftInACluster, :],c=colors[numOfRaftInACluster],label='{}'.format(numOfRaftInACluster))
    
ax.legend(loc='best')  
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusterSizeCounts.max()+0.5])  
ax.set_xlabel('Time(Frames)',size=20)
ax.set_ylabel('cluster count'.format(raft1Num),size=20)
ax.set_title(' the counts of clusters of various sizes over time', size = 20)
ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()


# plot average cluster sizes vs frame number

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
colors = plt.cm.jet(np.linspace(0,1,numOfRafts))

ax.plot(np.arange(numOfFrames),clusterSizeAvgIncludingLoners,label='average cluster size including loners')
ax.plot(np.arange(numOfFrames),clusterSizeAvgExcludingLoners,label='average cluster size excluding loners')
ax.legend(loc='best')
ax.set_xlim([0, numOfFrames])
ax.set_ylim([0, clusterSizeAvgExcludingLoners.max()+0.5])  
ax.set_xlabel('Times(Frames)',size=20)
ax.set_ylabel('average size of clusters',size=20)
ax.set_title('average size of clusters for {}'.format(outputDataFileName), size = 20)
ax.tick_params(axis='both', labelsize=18, width = 2, length = 10)
fig.show()
#fig.savefig(outputDataFileName+'_' + 'averageSizeOfClusters.png',dpi=300)

#plt.close('all')

#%% drawing clusters and saving into movies
if os.path.isdir(subfolderName):
    os.chdir(subfolderName)
else:
    print(subfolderName + ' subfolder' + ' does not exist in the current folder.')
        

tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()

outputImage = 1
outputVideo = 0

currentFrameBGR = cv.imread(tiffFileList[0])
outputFrameRate = 5.0

if outputVideo == 1:
    outputVideoName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_clustersMarked.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName,fourcc, outputFrameRate, (frameH, frameW), 1)


for currentFrameNum in progressbar.progressbar(range(10)):
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    currentFrameDraw = DrawRafts(currentFrameDraw, raftLocations[:,currentFrameNum,:], raftRadii[:,currentFrameNum], numOfRafts)
    currentFrameDraw = DrawRaftNumber(currentFrameDraw, raftLocations[:,currentFrameNum,:], numOfRafts)
    currentFrameDraw = DrawClusters(currentFrameDraw, raftPairwiseConnectivity[:,:,currentFrameNum], raftLocations[:,currentFrameNum,:])  
    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_cluster_' + str(currentFrameNum+1).zfill(4) + '.jpg'
        cv.imwrite(outputImageName,currentFrameDraw)
    if outputVideo == 1:
        videoOut.write(currentFrameDraw)

if outputVideo == 1:
    videoOut.release()

#plt.imshow(currentFrameDraw[:,:,::-1])

#%% Voronoi analysis

if os.path.isdir(subfolderName):
    os.chdir(subfolderName)
else:
    print(subfolderName + ' subfolder' + ' does not exist in the current folder.')
  
tiffFileList = glob.glob('*.tiff')
tiffFileList.sort()

dfNeighbors = pd.DataFrame(columns = ['frameNum', 'raftID', 'localDensity',
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

dfNeighborsAllFrames = pd.DataFrame(columns = ['frameNum', 'raftID', 'localDensity',
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

entropyByNeighborCount = np.zeros(numOfFrames)
entropyByNeighborCountWeighted = np.zeros(numOfFrames)
entropyByNeighborDistances = np.zeros(numOfFrames)
entropyByLocalDensities = np.zeros(numOfFrames)

binEdgesNeighborCountWeighted = np.arange(1, 7, 1).tolist()
binEdgesNeighborDistances = np.arange(2,10,0.5).tolist() + [100]
binEdgesLocalDensities = np.arange(0,1,0.05).tolist()

deltaR = 1
sizeOfArenaInRadius = 10000/150 # 1cm square arena, 150 um raft radius
radialRangeArray = np.arange(2, 100, deltaR)

hexaticOrderParameterAvgs = np.zeros(numOfFrames, dtype = np.csingle)
hexaticOrderParameterAvgNorms = np.zeros(numOfFrames)
hexaticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype = np.csingle)
hexaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
hexaticOrderParameterModuliiStds = np.zeros(numOfFrames)

pentaticOrderParameterAvgs = np.zeros(numOfFrames, dtype = np.csingle)
pentaticOrderParameterAvgNorms = np.zeros(numOfFrames)
pentaticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype = np.csingle)
pentaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
pentaticOrderParameterModuliiStds = np.zeros(numOfFrames)

tetraticOrderParameterAvgs = np.zeros(numOfFrames, dtype = np.csingle)
tetraticOrderParameterAvgNorms = np.zeros(numOfFrames)
tetraticOrderParameterMeanSquaredDeviations = np.zeros(numOfFrames, dtype = np.csingle)
tetraticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
tetraticOrderParameterModuliiStds = np.zeros(numOfFrames)

radialDistributionFunction = np.zeros((numOfFrames, len(radialRangeArray))) # pair correlation function: g(r)
spatialCorrHexaOrderPara = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of hexatic order paramter: g6(r)
spatialCorrPentaOrderPara = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of pentatic order paramter: g5(r)
spatialCorrTetraOrderPara = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of tetratic order paramter: g4(r)

spatialCorrHexaBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of bond orientation parameter: g6(r)/g(r)
spatialCorrPentaBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of bond orientation parameter: g5(r)/g(r)
spatialCorrTetraBondOrientationOrder = np.zeros((numOfFrames, len(radialRangeArray))) # spatial correlation of bond orientation parameter: g4(r)/g(r)


drawingNeighborCountWeighted = 1 # 0- no drawing, 1- drawing neighborCount, 2 - drawing neighborCountWeighted

drawingRaftOrderParameterModulii = 6 # 4 - tetratic order, 5 - pentatic order, and 6 - hexatic order

outputImage = 1
outputVideo = 0

if outputVideo == 1:
    outputFrameRate = 5.0
    currentFrameBGR = cv.imread(tiffFileList[0])
    outputVideoName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_Voronoi' + str(drawingNeighborCountWeighted) + '.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName,fourcc, outputFrameRate, (frameH, frameW), 1)


for currentFrameNum in progressbar.progressbar(range(numOfFrames)):
    # currentFrameNum = 0
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    currentFrameDraw = DrawRafts(currentFrameDraw, raftLocations[:,currentFrameNum,:], raftRadii[:,currentFrameNum], numOfRafts)
    currentFrameDraw = DrawRaftNumber(currentFrameDraw, raftLocations[:,currentFrameNum,:], numOfRafts)
    currentFrameDraw = DrawVoronoi(currentFrameDraw,raftLocations[:,currentFrameNum,:])
    #plt.imshow(currentFrameDraw[:,:,::-1])

    vor = scipyVoronoi(raftLocations[:,currentFrameNum,:])
    allVertices = vor.vertices
    neighborPairs = vor.ridge_points # row# is the index of a ridge, columns are the two point# that correspond to the ridge 
    ridgeVertexPairs = np.asarray(vor.ridge_vertices) # row# is the index of a ridge, columns are two vertex# of the ridge
    raftPairwiseDistancesMatrix = raftPairwiseDistancesInRadius[:, :, currentFrameNum]

    for raftID in np.arange(numOfRafts):
        ridgeIndices0 =  np.nonzero(neighborPairs[:,0] == raftID)
        ridgeIndices1 =  np.nonzero(neighborPairs[:,1] == raftID)
        ridgeIndices = np.concatenate((ridgeIndices0, ridgeIndices1), axis = None) # index is for the index of neighborPairs or ridgeVertexPairs list
        neighborPairsOfOneRaft = neighborPairs[ridgeIndices,:]
        neighborsOfOneRaft = np.concatenate((neighborPairsOfOneRaft[neighborPairsOfOneRaft[:,0] == raftID,1], neighborPairsOfOneRaft[neighborPairsOfOneRaft[:,1] == raftID,0]))
        ridgeVertexPairsOfOneRaft = ridgeVertexPairs[ridgeIndices,:]
        neighborDistances = raftPairwiseDistancesMatrix[raftID, neighborsOfOneRaft]
        neighborDistanceAvg = neighborDistances.mean()
        
              
        ## order parameters and the spatial correlation functions of the order parameters
        raftLocation = raftLocations[raftID,currentFrameNum,:]
        neighborLocations = raftLocations[neighborsOfOneRaft,currentFrameNum,:]
        
        # note the negative sign, it is to make the angle Rhino-like
        neighborAnglesInRad = np.arctan2(-(neighborLocations[:,1] - raftLocation[1]),(neighborLocations[:,0] - raftLocation[0]))
        neighborAnglesInDeg = neighborAnglesInRad * 180 / np.pi
        
        raftHexaticOrderParameter = np.cos(neighborAnglesInRad*6).mean() + np.sin(neighborAnglesInRad*6).mean()*1j
        raftPentaticOrderParameter = np.cos(neighborAnglesInRad*5).mean() + np.sin(neighborAnglesInRad*5).mean()*1j
        raftTetraticOrderParameter = np.cos(neighborAnglesInRad*4).mean() + np.sin(neighborAnglesInRad*4).mean()*1j
        
        # calculate local density of each voronoi cell
        if np.all(ridgeVertexPairsOfOneRaft >= 0): 
            vertexIDsOfOneRaft = np.unique(ridgeVertexPairsOfOneRaft)
            verticesOfOneRaft = allVertices[vertexIDsOfOneRaft]
            raftXY = raftLocations[raftID, currentFrameNum, :]
            
            #polar angles in plt.plot
            polarAngles = np.arctan2((verticesOfOneRaft[:,1] - raftXY[1]), (verticesOfOneRaft[:,0] - raftXY[0])) * 180 / np.pi
            
            verticesOfOneRaftSorted = verticesOfOneRaft[polarAngles.argsort()]
            
            voronoiCellArea = PolygonArea(verticesOfOneRaftSorted[:,0], verticesOfOneRaftSorted[:,1])
            
            localDensity = radius * radius * np.pi / voronoiCellArea
        else:
            localDensity = 0

        #initialize variables related to ridge lengths
        ridgeLengths = np.zeros(len(neighborsOfOneRaft))
        ridgeLengthsScaled = np.zeros(len(neighborsOfOneRaft))
        ridgeLengthsScaledNormalizedBySum = np.zeros(len(neighborsOfOneRaft))
        ridgeLengthsScaledNormalizedByMax = np.zeros(len(neighborsOfOneRaft))
        
        #go through all ridges to calculate or assign ridge length
        for ridgeIndexOfOneRaft, neighborID in enumerate(neighborsOfOneRaft):
            neighborDistance = CalculateDistance(raftLocations[raftID,currentFrameNum,:], raftLocations[neighborID,currentFrameNum,:])
            if np.all(ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft] >= 0 ):
                vertex1ID = ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft][0]
                vertex2ID = ridgeVertexPairsOfOneRaft[ridgeIndexOfOneRaft][1]
                vertex1 = allVertices[vertex1ID]
                vertex2 = allVertices[vertex2ID]
                ridgeLengths[ridgeIndexOfOneRaft] = CalculateDistance(vertex1, vertex2)
                #for ridges that has one vertex outside the image (negative corrdinate)
                #set ridge length to the be the diameter of the raft
                if np.all(vertex1 >= 0) and np.all(vertex2 >= 0):
                    ridgeLengthsScaled[ridgeIndexOfOneRaft] = ridgeLengths[ridgeIndexOfOneRaft] * raftRadii[neighborID,currentFrameNum] * 2 / neighborDistance
                else:
                    ridgeLengthsScaled[ridgeIndexOfOneRaft] = raftRadii[neighborID,currentFrameNum] ** 2 * 4 / neighborDistance
            else:
                #for ridges that has one vertex in the infinity ridge vertex#< 0 (= -1) 
                #set ridge length to the be the diameter of the raft
                ridgeLengths[ridgeIndexOfOneRaft] = raftRadii[neighborID,currentFrameNum] * 2
                ridgeLengthsScaled[ridgeIndexOfOneRaft] = raftRadii[neighborID,currentFrameNum] ** 2 * 4 / neighborDistance
                
                
        ridgeLengthsScaledNormalizedBySum = ridgeLengthsScaled / ridgeLengthsScaled.sum()
        ridgeLengthsScaledNormalizedByMax = ridgeLengthsScaled / ridgeLengthsScaled.max()
        neighborCountWeighted = ridgeLengthsScaledNormalizedByMax.sum() # assuming the neighbor having the longest ridge (scaled) counts one. 
        neighborDistanceWeightedAvg = np.average(neighborDistances, weights = ridgeLengthsScaledNormalizedBySum)
        
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
    
    hexaticOrderParameterList =  dfNeighbors['hexaticOrderParameter'].tolist()
    pentaticOrderParameterList =  dfNeighbors['pentaticOrderParameter'].tolist()
    tetraticOrderParameterList =  dfNeighbors['tetraticOrderParameter'].tolist()
    neighborCountSeries = dfNeighbors['neighborCount']
    neighborCountWeightedList = dfNeighbors['neighborCountWeighted'].tolist()
    neighborDistancesList = np.concatenate(dfNeighbors['neighborDistances'].tolist())
    localDensitiesList = dfNeighbors['localDensity'].tolist()
    
    hexaticOrderParameterArray = np.array(hexaticOrderParameterList)
    hexaticOrderParameterAvgs[currentFrameNum] = hexaticOrderParameterArray.mean()
    hexaticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(hexaticOrderParameterAvgs[currentFrameNum].real ** 2 + hexaticOrderParameterAvgs[currentFrameNum].imag ** 2)
    hexaticOrderParameterMeanSquaredDeviations[currentFrameNum] = ((hexaticOrderParameterArray - hexaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
    hexaticOrderParameterMolulii = np.absolute(hexaticOrderParameterArray)
    hexaticOrderParameterModuliiAvgs[currentFrameNum] = hexaticOrderParameterMolulii.mean()
    hexaticOrderParameterModuliiStds[currentFrameNum] = hexaticOrderParameterMolulii.std()
    
    pentaticOrderParameterArray = np.array(pentaticOrderParameterList)
    pentaticOrderParameterAvgs[currentFrameNum] = pentaticOrderParameterArray.mean()
    pentaticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(pentaticOrderParameterAvgs[currentFrameNum].real ** 2 + pentaticOrderParameterAvgs[currentFrameNum].imag ** 2)
    pentaticOrderParameterMeanSquaredDeviations[currentFrameNum] = ((pentaticOrderParameterArray - pentaticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
    pentaticOrderParameterModulii = np.absolute(pentaticOrderParameterArray)
    pentaticOrderParameterModuliiAvgs[currentFrameNum] = pentaticOrderParameterModulii.mean()
    pentaticOrderParameterModuliiStds[currentFrameNum] = pentaticOrderParameterModulii.std()
    
    tetraticOrderParameterArray = np.array(tetraticOrderParameterList)
    tetraticOrderParameterAvgs[currentFrameNum] = tetraticOrderParameterArray.mean()
    tetraticOrderParameterAvgNorms[currentFrameNum] = np.sqrt(tetraticOrderParameterAvgs[currentFrameNum].real ** 2 + tetraticOrderParameterAvgs[currentFrameNum].imag ** 2)
    tetraticOrderParameterMeanSquaredDeviations[currentFrameNum] = ((tetraticOrderParameterArray - tetraticOrderParameterAvgs[currentFrameNum]) ** 2).mean()
    tetraticOrderParameterModulii = np.absolute(tetraticOrderParameterArray)
    tetraticOrderParameterModuliiAvgs[currentFrameNum] = tetraticOrderParameterModulii.mean()
    tetraticOrderParameterModuliiStds[currentFrameNum] = tetraticOrderParameterModulii.std()
    
    # g(r), g6(r), g5(r), and g4(r) for this frame
    for radialIndex, radialIntervalStart in enumerate(radialRangeArray): 
        radialIntervalEnd =  radialIntervalStart + deltaR
        # g(r)
        js, ks = np.logical_and(raftPairwiseDistancesMatrix>=radialIntervalStart, raftPairwiseDistancesMatrix<radialIntervalEnd).nonzero()
        count = len(js)
        density = numOfRafts / sizeOfArenaInRadius**2 
        radialDistributionFunction[currentFrameNum, radialIndex] =  count / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
        # g6(r), g5(r), g4(r)
        sumOfProductsOfPsi6 = (hexaticOrderParameterArray[js] * np.conjugate(hexaticOrderParameterArray[ks])).sum().real
        spatialCorrHexaOrderPara[currentFrameNum, radialIndex] = sumOfProductsOfPsi6 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
        sumOfProductsOfPsi5 = (pentaticOrderParameterArray[js] * np.conjugate(pentaticOrderParameterArray[ks])).sum().real
        spatialCorrPentaOrderPara[currentFrameNum, radialIndex] = sumOfProductsOfPsi5 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
        sumOfProductsOfPsi4 = (tetraticOrderParameterArray[js] * np.conjugate(tetraticOrderParameterArray[ks])).sum().real
        spatialCorrTetraOrderPara[currentFrameNum, radialIndex] = sumOfProductsOfPsi4 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
        
        # g6(r)/g(r); g5(r)/g(r); g4(r)/g(r)
        if radialDistributionFunction[currentFrameNum, radialIndex] != 0: 
            spatialCorrHexaBondOrientationOrder[currentFrameNum, radialIndex] = spatialCorrHexaOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[currentFrameNum, radialIndex]
            spatialCorrPentaBondOrientationOrder[currentFrameNum, radialIndex] = spatialCorrPentaOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[currentFrameNum, radialIndex]
            spatialCorrTetraBondOrientationOrder[currentFrameNum, radialIndex] = spatialCorrTetraOrderPara[currentFrameNum, radialIndex] / radialDistributionFunction[currentFrameNum, radialIndex]
    
    count1 = np.asarray(neighborCountSeries.value_counts())
    entropyByNeighborCount[currentFrameNum] = ShannonEntropy(count1)
    
    count2, _ = np.histogram(np.asarray(neighborCountWeightedList),binEdgesNeighborCountWeighted)
    entropyByNeighborCountWeighted[currentFrameNum] = ShannonEntropy(count2)
    
    count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
    entropyByNeighborDistances[currentFrameNum] = ShannonEntropy(count3)
    
    count4, _ = np.histogram(np.asarray(localDensitiesList), binEdgesLocalDensities)
    entropyByLocalDensities[currentFrameNum] = ShannonEntropy(count4)
    
    neighborCountWeightedList = dfNeighbors['neighborCountWeighted'].tolist()
    neighborCountList = dfNeighbors['neighborCount'].tolist()
    
    if drawingRaftOrderParameterModulii == 6:
        currentFrameDrawOrderPara = DrawAtBottomLeftOfRaftNumberFloat(currentFrameDraw.copy(), raftLocations[:,currentFrameNum,:], hexaticOrderParameterMolulii, numOfRafts)
    elif drawingRaftOrderParameterModulii == 5:
        currentFrameDrawOrderPara = DrawAtBottomLeftOfRaftNumberFloat(currentFrameDraw.copy(), raftLocations[:,currentFrameNum,:], pentaticOrderParameterModulii, numOfRafts)
    elif drawingRaftOrderParameterModulii == 4:
        currentFrameDrawOrderPara = DrawAtBottomLeftOfRaftNumberFloat(currentFrameDraw.copy(), raftLocations[:,currentFrameNum,:], tetraticOrderParameterModulii, numOfRafts)
    
    if drawingNeighborCountWeighted == 1:
        currentFrameDrawNeighborCount = DrawAtBottomLeftOfRaftNumberInteger(currentFrameDraw.copy(), raftLocations[:,currentFrameNum,:], neighborCountList, numOfRafts)
    elif drawingNeighborCountWeighted == 2:
        currentFrameDrawNeighborCount = DrawAtBottomLeftOfRaftNumberFloat(currentFrameDraw.copy(), raftLocations[:,currentFrameNum,:], neighborCountWeightedList, numOfRafts)
    
    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_Voronoi' + str(drawingNeighborCountWeighted) + '_' + str(currentFrameNum+1).zfill(4) + '.jpg'
        cv.imwrite(outputImageName,currentFrameDrawNeighborCount)
        outputImageNameOrderPara = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_OrderPara' + str(drawingRaftOrderParameterModulii) + '_' + str(currentFrameNum+1).zfill(4) + '.jpg'
        cv.imwrite(outputImageNameOrderPara,currentFrameDrawOrderPara)
    if outputVideo == 1:
        videoOut.write(currentFrameDrawNeighborCount)
        
    dfNeighborsAllFrames = dfNeighborsAllFrames.append(dfNeighbors,ignore_index=True)

if outputVideo == 1:
    videoOut.release()

dfNeighborsAllFrames = dfNeighborsAllFrames.infer_objects()
dfNeighborsAllFramesSorted = dfNeighborsAllFrames.sort_values(['frameNum','raftID'], ascending = [1,1])


#g6(t), g5(t), g4(t): each raft has its own temporal correlation of g6, the unit of deltaT is frame
temporalCorrHexaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype = complex) 
temporalCorrPentaBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype = complex) 
temporalCorrTetraBondOrientationOrder = np.zeros((numOfRafts, numOfFrames), dtype = complex) 
temporalCorrHexaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype = complex)
temporalCorrPentaBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype = complex)
temporalCorrTetraBondOrientationOrderAvgAllRafts = np.zeros(numOfFrames, dtype = complex)

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
    hexaOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(np.broadcast_to(hexaOrdParaOfOneRaftArrayConjugate, hexaOrdParaOfOneRaftToeplitzMatrix.shape))
    pentaOrdParaOfOneRaftArrayConjugate = np.conjugate(pentaOrdParaOfOneRaftArray)
    pentaOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(np.broadcast_to(pentaOrdParaOfOneRaftArrayConjugate, pentaOrdParaOfOneRaftToeplitzMatrix.shape))
    tetraOrdParaOfOneRaftArrayConjugate = np.conjugate(tetraOrdParaOfOneRaftArray)
    tetraOrdParaOfOneRaftArrayConjugateBroadcasted = np.transpose(np.broadcast_to(tetraOrdParaOfOneRaftArrayConjugate, tetraOrdParaOfOneRaftToeplitzMatrix.shape))
    
    # multiply the two matrix so that for each column, the rows on and below the diagonal are the products of 
    # the conjugate of psi6(t0) and psi6(t0 + tStepSize), the tStepSize is the same the column index. 
    hexaOrdParaOfOneRaftBroadcastedTimesToeplitz = hexaOrdParaOfOneRaftArrayConjugateBroadcasted * hexaOrdParaOfOneRaftToeplitzMatrix
    pentaOrdParaOfOneRaftBroadcastedTimesToeplitz = pentaOrdParaOfOneRaftArrayConjugateBroadcasted * pentaOrdParaOfOneRaftToeplitzMatrix
    tetraOrdParaOfOneRaftBroadcastedTimesToeplitz = tetraOrdParaOfOneRaftArrayConjugateBroadcasted * tetraOrdParaOfOneRaftToeplitzMatrix
    
    for tStepSize in np.arange(numOfFrames):
        temporalCorrHexaBondOrientationOrder[raftID, tStepSize] = np.average(hexaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:,tStepSize])
        temporalCorrPentaBondOrientationOrder[raftID, tStepSize] = np.average(pentaOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:,tStepSize])
        temporalCorrTetraBondOrientationOrder[raftID, tStepSize] = np.average(tetraOrdParaOfOneRaftBroadcastedTimesToeplitz[tStepSize:,tStepSize])

temporalCorrHexaBondOrientationOrderAvgAllRafts = temporalCorrHexaBondOrientationOrder.mean(axis = 0)
temporalCorrPentaBondOrientationOrderAvgAllRafts = temporalCorrPentaBondOrientationOrder.mean(axis = 0)
temporalCorrTetraBondOrientationOrderAvgAllRafts = temporalCorrTetraBondOrientationOrder.mean(axis = 0)




#%% plots for Voronoi analysis
frameNumToLook = 0
dfNeighborsOneFrame = dfNeighborsAllFrames[dfNeighborsAllFrames.frameNum == frameNumToLook]

dfNeighborsOneFramehexaOrdPara = dfNeighborsOneFrame['hexaticOrderParameter']
dfNeighborsOneFramePhaseAngle = np.angle(dfNeighborsOneFramehexaOrdPara, deg=1)
dfNeighborsOneFrameModulii = np.absolute(dfNeighborsOneFramehexaOrdPara)
dfNeighborsOneFrameModulii.mean()
dfNeighborsOneFrameCosPhaseAngle = np.cos(dfNeighborsOneFramePhaseAngle)

NeighborCountSeries = dfNeighborsOneFrame['neighborCount'] 
binEdgesNeighborCount = list(range(NeighborCountSeries.min(), NeighborCountSeries.max()+2))
count1, _ = np.histogram(np.asarray(NeighborCountSeries),binEdgesNeighborCount)
#count1 = np.asarray(dfNeighborsOneFrame['neighborCount'].value_counts().sort_index())
entropyByNeighborCount1 = ShannonEntropy(count1)
fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.bar(binEdgesNeighborCount[:-1], count1, align = 'edge', width = 0.5)
ax.set_xlabel('neighbor counts',  {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor counts, entropy: {:.3} bits'.format(entropyByNeighborCount1), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()


neighborCountWeightedSeries = dfNeighborsOneFrame['neighborCountWeighted']
count2, _ = np.histogram(np.asarray(neighborCountWeightedSeries),binEdgesNeighborCountWeighted)
entropyByNeighborCountWeighted2 = ShannonEntropy(count2)
fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.bar(binEdgesNeighborCountWeighted[:-1], count2, align = 'edge', width = 0.5)
ax.set_xlabel('neighbor counts weighted',  {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor counts weighted, entropy: {:.3} bits'.format(entropyByNeighborCountWeighted2), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

neighborDistancesList = np.concatenate(dfNeighborsOneFrame['neighborDistances'].tolist())
count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
entropyByNeighborDistances3 = ShannonEntropy(count3)
fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.bar(binEdgesNeighborDistances[:-1], count3, align = 'edge', width = 0.2)
ax.set_xlabel('neighbor distances',  {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of neighbor distances, entropy: {:.3} bits'.format(entropyByNeighborDistances3), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

localDensitiesList = dfNeighborsOneFrame['localDensity'].tolist()
count4, _ = np.histogram(np.asarray(localDensitiesList), binEdgesLocalDensities)
entropyByLocalDensities4 = ShannonEntropy(count4)
fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.bar(binEdgesLocalDensities[:-1], count4, align = 'edge', width = 0.02)
ax.set_xlabel('local densities',  {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of local densities, entropy: {:.3} bits'.format(entropyByLocalDensities4), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()


fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(radialRangeArray, radialDistributionFunction[frameNumToLook,:], label = 'radial distribution function g(r)')
ax.set_xlabel('radial range',  {'size': 15})
ax.set_ylabel('radial distribution function g(r)', {'size': 15})
ax.set_title('radial distribution function  g(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()


fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(radialRangeArray, spatialCorrHexaOrderPara[frameNumToLook,:], label = 'spatial correlation of hexatic order parameter g6(r)')
ax.set_xlabel('radial range',  {'size': 15})
ax.set_ylabel('spatial correlation of hexatic order parameter g6(r)', {'size': 15})
ax.set_title('spatial correlation of hexatic order parameter g6(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(radialRangeArray, spatialCorrHexaBondOrientationOrder[frameNumToLook,:], label = 'spatial correlation of hexa bond orientational order g6(r) / g(r)')
ax.set_xlabel('radial range',  {'size': 15})
ax.set_ylabel('spatial correlation of bond orientational order g6(r) / g(r)', {'size': 15})
ax.set_title('spatial correlation of bond orientational order g6(r) / g(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(radialRangeArray, spatialCorrPentaOrderPara[frameNumToLook,:], label = 'spatial correlation of Pentatic order parameter g5(r)')
ax.set_xlabel('radial range',  {'size': 15})
ax.set_ylabel('spatial correlation of hexatic order parameter g5(r)', {'size': 15})
ax.set_title('spatial correlation of hexatic order parameter g5(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(radialRangeArray, spatialCorrPentaBondOrientationOrder[frameNumToLook,:], label = 'spatial correlation of penta bond orientational order g5(r) / g(r)')
ax.set_xlabel('radial range',  {'size': 15})
ax.set_ylabel('spatial correlation of bond orientational order g5(r) / g(r)', {'size': 15})
ax.set_title('spatial correlation of bond orientational order g5(r) / g(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(radialRangeArray, spatialCorrTetraOrderPara[frameNumToLook,:], label = 'spatial correlation of tetratic order parameter g4(r)')
ax.set_xlabel('radial range',  {'size': 15})
ax.set_ylabel('spatial correlation of tetratic order parameter g4(r)', {'size': 15})
ax.set_title('spatial correlation of tetratic order parameter g4(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(radialRangeArray, spatialCorrTetraBondOrientationOrder[frameNumToLook,:], label = 'spatial correlation of tetra bond orientational order g4(r) / g(r)')
ax.set_xlabel('radial range',  {'size': 15})
ax.set_ylabel('spatial correlation of tetra bond orientational order g4(r) / g(r)', {'size': 15})
ax.set_title('spatial correlation of tetra bond orientational order g4(r) / g(r) of frame# {:}'.format(frameNumToLook), {'size': 15})
ax.legend(['frame number {}'.format(frameNumToLook)])
fig.show()

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(np.arange(numOfFrames), entropyByNeighborCount, label = 'entropyByNeighborCount')
ax.plot(np.arange(numOfFrames), entropyByNeighborCountWeighted , label = 'entropyByNeighborCountWeighted')
ax.plot(np.arange(numOfFrames), entropyByNeighborDistances , label = 'entropyByNeighborDistances')
ax.plot(np.arange(numOfFrames), entropyByLocalDensities, label = 'entropyByLocalDensities')
ax.set_xlabel('frames',  {'size': 15})
ax.set_ylabel('entropies', {'size': 15})
ax.set_title('entropies over frames', {'size': 15})
ax.legend(loc='best')
fig.show()

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(np.arange(numOfFrames), hexaticOrderParameterModuliiAvgs, label = 'hexatic order parameter modulii average' )
ax.plot(np.arange(numOfFrames), pentaticOrderParameterModuliiAvgs, label = 'pentatic order parameter modulii average' )
ax.plot(np.arange(numOfFrames), tetraticOrderParameterModuliiAvgs, label = 'tetratic order parameter modulii average' )
ax.plot(np.arange(numOfFrames), hexaticOrderParameterAvgNorms, label = 'hexatic order parameter avg norms' )
ax.plot(np.arange(numOfFrames), pentaticOrderParameterAvgNorms, label = 'pentatic order parameter avg norms' )
ax.plot(np.arange(numOfFrames), tetraticOrderParameterAvgNorms, label = 'tetratic order parameter avg norms' )
ax.set_xlabel('frames',  {'size': 15})
ax.set_ylabel('norm of the average of the order parameters', {'size': 15})
ax.set_title('norm of the average of the order parameters', {'size': 15})
ax.legend(loc='best')
fig.show()


# plot the temporal correlation of one specific raft 
raftID = 10

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(np.arange(numOfFrames)[1:], np.real(temporalCorrHexaBondOrientationOrder[raftID,1:]), label = 'real part of g6(t)')
ax.plot(np.arange(numOfFrames)[1:], np.imag(temporalCorrHexaBondOrientationOrder[raftID,1:]), label = 'imaginery part of g6(t)')
ax.set_xlabel('temporal step size (frame)',  {'size': 15})
ax.set_ylabel('temporal correlation of hexatic order parameter: g6(t)', {'size': 15})
ax.set_title('temporal correlation of hexatic order parameter: g6(t) for raft {}'.format(raftID), {'size': 15})
ax.legend()
fig.show()

# plot the temporal correlation averaged over all rafts
fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(np.arange(numOfFrames)[1:], np.real(temporalCorrHexaBondOrientationOrderAvgAllRafts[1:]), label = 'real part of g6(t) averaged over all rafts')
ax.plot(np.arange(numOfFrames)[1:], np.imag(temporalCorrHexaBondOrientationOrderAvgAllRafts[1:]), label = 'imaginery part of g6(t) averaged over all rafts')
ax.set_xlabel('temporal step size (frame)',  {'size': 15})
ax.set_ylabel('averaged temporal correlation of hexatic order parameter: g6(t)', {'size': 15})
ax.set_title('averaged temporal correlation of hexatic order parameter: g6(t) for raft {}'.format(raftID), {'size': 15})
ax.legend()
fig.show()

#%% drawing Voronoi diagrams and saving into movies
    
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
    outputVideoName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_' + str(magnification) + 'x_Voronoi.mp4'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    frameW, frameH, _ = currentFrameBGR.shape
    videoOut = cv.VideoWriter(outputVideoName,fourcc, outputFrameRate, (frameH, frameW), 1)

for currentFrameNum in progressbar.progressbar(range(len(tiffFileList))):
    currentFrameBGR = cv.imread(tiffFileList[currentFrameNum])
    currentFrameDraw = currentFrameBGR.copy()
    currentFrameDraw = DrawRafts(currentFrameDraw, raftLocations[:,currentFrameNum,:], raftRadii[:,currentFrameNum], numOfRafts)
    currentFrameDraw = DrawRaftNumber(currentFrameDraw, raftLocations[:,currentFrameNum,:], numOfRafts)
    currentFrameDraw = DrawVoronoi(currentFrameDraw,raftLocations[:,currentFrameNum,:])
    currentFrameDraw = DrawNeighborCounts(currentFrameDraw, raftLocations[:,currentFrameNum,:], numOfRafts)
    if outputImage == 1:
        outputImageName = date + '_' + str(numOfRafts) + 'Rafts_' + str(batchNum) + '_' + str(spinSpeed) + 'rps_Voronoi_' + str(currentFrameNum+1).zfill(4) + '.jpg'
        cv.imwrite(outputImageName,currentFrameDraw)
    if outputVideo == 1:
        videoOut.write(currentFrameDraw)

if outputVideo == 1:
    videoOut.release()
    
#plt.imshow(currentFrameBGR[:,:,::-1])
#scipyVoronoiPlot2D(vor)
#
#plt.show()


#%% mutual information analysis

# the durartion for which the frames are sampled to calculate one MI 
widthOfInterval = 400 # unit: number of frames,
 
numOfBins = 20

# The gap between two successive MI calculation. 
# Try keep (numOfFrames - widthOfInterval)//samplingGap an integer
samplingGap = 200 # unit: number of frames

numOfSamples = (numOfFrames - widthOfInterval)//samplingGap + 1
sampleFrameNums = np.arange(widthOfInterval,numOfFrames,samplingGap)

# pretreatment of position data
raftOrbitingAnglesAdjusted = AdjustOrbitingAngles2(raftOrbitingAngles, orbiting_angles_diff_threshold = 200)
raftVelocityR = np.gradient(raftOrbitingDistances, axis=1)
raftVelocityTheta = np.gradient(raftOrbitingAnglesAdjusted, axis=1)
raftVelocityNormPolar = np.sqrt(raftVelocityR * raftVelocityR + np.square(raftOrbitingDistances * np.radians(raftVelocityTheta)))
raftVelocityX = np.gradient(raftLocations[:,:,0],axis = 1)
raftVelocityY = np.gradient(raftLocations[:,:,1],axis = 1)
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
    distancesMatrix = raftOrbitingDistances[:, endOfInterval-widthOfInterval:endOfInterval] 
    mutualInfoAllSamplesAllRafts[:,:,i,0] = MutualInfoMatrix(distancesMatrix, numOfBins)
        
    angleMatrix = raftOrbitingAnglesAdjusted[:, endOfInterval-widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:,:,i,1] = MutualInfoMatrix(angleMatrix, numOfBins)
    
    coordinateXMatrix = raftLocations[:, endOfInterval-widthOfInterval:endOfInterval, 0]
    mutualInfoAllSamplesAllRafts[:,:,i,2] = MutualInfoMatrix(coordinateXMatrix, numOfBins)
    
    coordinateYMatrix = raftLocations[:, endOfInterval-widthOfInterval:endOfInterval, 1]
    mutualInfoAllSamplesAllRafts[:,:,i,3] = MutualInfoMatrix(coordinateYMatrix, numOfBins)
    
    velocityRMatrix = raftVelocityR[:, endOfInterval-widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:,:,i,4] = MutualInfoMatrix(velocityRMatrix, numOfBins)
    
    velocityThetaMatrix = raftVelocityTheta[:, endOfInterval-widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:,:,i,5] = MutualInfoMatrix(velocityThetaMatrix, numOfBins)
    
    velocityNormPolarMatrix = raftVelocityNormPolar[:, endOfInterval-widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:,:,i,6] = MutualInfoMatrix(velocityNormPolarMatrix, numOfBins)
    
    velocityXMatrix = raftVelocityX[:, endOfInterval-widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:,:,i,7] = MutualInfoMatrix(velocityXMatrix, numOfBins)
    
    velocityYMatrix = raftVelocityY[:, endOfInterval-widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:,:,i,8] = MutualInfoMatrix(velocityYMatrix, numOfBins)
    
    velocityNormXYMatrix = raftVelocityNormXY[:, endOfInterval-widthOfInterval:endOfInterval]
    mutualInfoAllSamplesAllRafts[:,:,i,9] = MutualInfoMatrix(velocityNormXYMatrix, numOfBins)


mutualInfoAllSamplesAvgOverAllRafts =  mutualInfoAllSamplesAllRafts.mean((0,1))
mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly = np.trace(mutualInfoAllSamplesAllRafts, axis1 = 0, axis2 = 1) / numOfRafts
mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI = (mutualInfoAllSamplesAvgOverAllRafts * numOfRafts - mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly) / (numOfRafts - 1)

mutualInfoAvg = mutualInfoAllSamplesAvgOverAllRafts.mean(axis = 0)
mutualInfoAvgSelfMIOnly = mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly.mean(axis = 0)
mutualInfoAvgExcludingSelfMI = mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI.mean(axis = 0)


t2 = time.perf_counter()
timeTotal = t2 - t1 # in seconds
print(timeTotal)


#%% plots for mutual information calculations

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
    
count, edges = np.histogram(timeSeries,numOfBins)
entropy = ShannonEntropy(count)

fig, ax = plt.subplots(2,1, figsize = (10,15))
ax[0].bar(edges[:-1], count, align = 'edge', width = (edges.max() - edges.min())/numOfBins/2)
ax[0].set_xlabel('time series id = {}'.format(dataSelection),  {'size': 15})
ax[0].set_ylabel('count', {'size': 15})
ax[0].set_title('histogram of time series id = {} for raft {} at frame {}, entropy: {:.3} bits,  {} bins'.format(dataSelection, raftNum, frameNum, entropy, numOfBins), {'size': 15})
ax[0].legend(['raft {}'.format(raftNum)])
ax[1].plot(timeSeries, np.arange(widthOfInterval), '-o')
ax[1].set_xlabel('time series id = {}'.format(dataSelection),  {'size': 15})
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
  
count1, _ = np.histogram(timeSeries1,numOfBins)
entropy1 = ShannonEntropy(count1)

count2, _ = np.histogram(timeSeries2,numOfBins)
entropy2 = ShannonEntropy(count2)

count, xEdges, yEdges = np.histogram2d(timeSeries1, timeSeries2, numOfBins)
jointEntropy = ShannonEntropy(count)
mutualInformation = mutual_info_score(None, None, contingency=count) * np.log2(np.e) # in unit of nats, * np.log2(np.e) to convert it to bits

fig, ax = plt.subplots(1,1, figsize = (7,7))
X, Y = np.meshgrid(xEdges, yEdges)
im = ax.pcolormesh(X, Y, count, cmap='inferno')
ax.set_title('2D histogram of raft {} and {} at frame {},\n '
             'with individual entropies {:.3} and {:.3} bits, \n'
             'joint entropy: {:.3} bits, mutual information: {:.3} bits,\n'
             'using {} bins'.format(raft1Num,raft2Num, frameNum, entropy1, entropy2, jointEntropy, mutualInformation, numOfBins))
cb = fig.colorbar(im)
fig.show()


#  plotting  averaged mutual information over time

# 0 - orbiting distances, 1 - orbiting angles, 2 - coordinate x, 3 - coordinate y
# 4 - velocity R, 5 - velocity theta, 6 - velocity norm in polar coordinate
# 7 - velocity x, 8 - velocity y, 9 - velocity norm in xy
dataSelection = 9

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.plot(sampleFrameNums, mutualInfoAllSamplesAvgOverAllRafts[:, dataSelection],'-o', label = 'mutualInfo averaged of all rafts data ID = {}'.format(dataSelection))
ax.plot(sampleFrameNums, mutualInfoAllSamplesAvgOverAllRaftsSelfMIOnly[:, dataSelection],'-o', label = 'mutualInfo self MI only averaged of all rafts data ID = {}'.format(dataSelection))
ax.plot(sampleFrameNums, mutualInfoAllSamplesAvgOverAllRaftsExcludingSelfMI[:, dataSelection],'-o', label = 'mutualInfo excluding self averaged of all rafts data ID = {}'.format(dataSelection))
ax.legend()
ax.set_xlabel('frame numbers', {'size': 15})
ax.set_ylabel('mutual information in bits',{'size': 15})
ax.set_xticks(np.arange(0,sampleFrameNums[-1],100))
fig.show()

# plotting mutual information matrix for all rafts at a specific frame number
sampleIndex = 5
frameNum = sampleFrameNums[sampleIndex]

# 0 - orbiting distances, 1 - orbiting angles, 2 - coordinate x, 3 - coordinate y
# 4 - velocity R, 5 - velocity theta, 6 - velocity norm in polar coordinate
# 7 - velocity x, 8 - velocity y, 9 - velocity norm in xy
dataSelection = 0 

mutualInformationMatrix = mutualInfoAllSamplesAllRafts[:,:,sampleIndex, dataSelection].copy()
    
fig, ax = plt.subplots(1,1, figsize = (7,7))
X, Y = np.meshgrid(np.arange(numOfRafts+1), np.arange(numOfRafts+1)) # +1 for the right edge of the last raft. 
im = ax.pcolormesh(X, Y, mutualInformationMatrix, cmap='inferno')
ax.set_xticks(np.arange(1,numOfRafts+1,10))
ax.set_yticks(np.arange(1,numOfRafts+1,10))
ax.set_title('mutual information matrix for data ID = {} at frame number {}'.format(dataSelection, frameNum))
cb = fig.colorbar(im)
fig.show()

# plotting mutual information matrix for all rafts at a specific frame number, excluding self MI
sampleIndex = 5
frameNum = sampleFrameNums[sampleIndex]

dataSelection = 9

mutualInformationMatrix = mutualInfoAllSamplesAllRafts[:,:,sampleIndex,dataSelection].copy()

np.fill_diagonal(mutualInformationMatrix,0)

fig, ax = plt.subplots(1,1, figsize = (7,7))
X, Y = np.meshgrid(np.arange(numOfRafts+1), np.arange(numOfRafts+1)) # +1 for the right edge of the last raft. 
im = ax.pcolormesh(X, Y, mutualInformationMatrix, cmap='inferno')
ax.set_xticks(np.arange(1,numOfRafts+1,10))
ax.set_yticks(np.arange(1,numOfRafts+1,10))
ax.set_title('mutual information matrix for data ID = {} at frame number {}'.format(dataSelection, frameNum))
cb = fig.colorbar(im)
fig.show()

# plotting the mutual information between one raft and the rest rafts over time, line plot
raft1Num = 90
colors = plt.cm.jet(np.linspace(0,1,numOfRafts))

dataSelection = 1 

mutualInformationSeries = mutualInfoAllSamplesAllRafts[:,:,:,dataSelection].copy()
 
fig, ax = plt.subplots(1,1,figsize=(15,15))
for raft2Num in range(0, numOfRafts):
    if raft1Num != raft2Num :
        ax.plot(sampleFrameNums, mutualInformationSeries[raft1Num, raft2Num,:],'-o', color = colors[raft2Num], label = '{}'.format(raft2Num))
    
ax.legend(loc='best') 
ax.set_xlabel('frame numbers', {'size': 15})
ax.set_ylabel('mutual information between raft {} and another raft in bits'.format(raft1Num),{'size': 15})
ax.set_xlim([0, sampleFrameNums.max()])
ax.set_ylim([0, mutualInformationSeries.max()+0.5])  
ax.set_xticks(np.arange(0,sampleFrameNums[-1],100))
ax.set_title('mutual information of data ID = {} between raft {} and another raft in bits'.format(dataSelection, raft1Num))
fig.show()

# plotting the mutual information of one raft with the rest over time, color maps
raftNum = 89
colors = plt.cm.jet(np.linspace(0,1,numOfRafts))

dataSelection = 0 

mutualInformationSeries = mutualInfoAllSamplesAllRafts[:,:,:,dataSelection].copy()

oneRaftMIMatrix = mutualInformationSeries[raftNum,:,:].copy()

oneRaftMIMatrixExcludingSelfMI = oneRaftMIMatrix.copy()
oneRaftMIMatrixExcludingSelfMI[raftNum, :] = 0

fig, ax = plt.subplots(1,1, figsize = (7,7))
X, Y = np.meshgrid(np.arange(numOfSamples+1), np.arange(numOfRafts+1)) # +1 for the right edge of the last raft. 
im = ax.pcolormesh(X, Y, oneRaftMIMatrixExcludingSelfMI, cmap='inferno')
ax.set_xticks(np.arange(1,numOfSamples+1,1))
ax.set_yticks(np.arange(1,numOfRafts+1,10))
ax.set_title('mutual information of data ID = {} of raft {} over time'.format(dataSelection, raftNum))
cb = fig.colorbar(im)
fig.show()

#fig.savefig(outputDataFileName+'_raftNum{}_'.format(raftNum) + 'MIoverTime.png',dpi=300)



#plt.close('all')



#%%   Analysis with cross-correlation
raft1Num = 89
raft2Num = 90

frameNum = 200
widthOfInterval = 100

traj1 = raftOrbitingDistances[raft1Num, frameNum - widthOfInterval:frameNum + widthOfInterval]
traj2 = raftOrbitingDistances[raft2Num, frameNum - widthOfInterval:frameNum]

fig1, ax1 = plt.subplots()
ax1.plot(np.arange(frameNum - widthOfInterval,frameNum + widthOfInterval), traj1)
ax1.plot(np.arange(frameNum - widthOfInterval,frameNum), traj2)
fig1.show()

fluctuation1 = traj1 - np.mean(traj1)
fluctuation2 = traj2 - np.mean(traj2)

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(frameNum - widthOfInterval,frameNum + widthOfInterval),fluctuation1)
ax2.plot(np.arange(frameNum - widthOfInterval,frameNum),fluctuation2)
fig2.show()

rollingCorr = np.correlate(fluctuation1, fluctuation2, 'valid')
fig3, ax3 = plt.subplots()
ax3.plot(rollingCorr)
fig3.show()

frameRate = 200
f, powerSpectrum = FFTDistances(frameRate, rollingCorr)

fig4, ax4 = plt.subplots()
ax4.plot(f,powerSpectrum)
fig4.show()

plt.close('all')

#%% testing permuatation entropy

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


time_series = raftOrbitingDistances[0,:].copy()
m = 3 # order = 2
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

pe = permutation_entropy(time_series,3,1)

#%% loading a matlab data file

data = loadmat("2018-02-23_19Rafts_1_2500rpm_20x_Cam_20189_CineF5mat.mat")

numOfRafts = data['numOfRafts'][0][0]
numOfFrames = data['numOfFrames'][0][0]
raftRadii = data['raftRadii']
raftLocations = data['raftLocations']
raftOrbitingCenters = centers = np.mean(raftLocations,axis = 0)
raftOrbitingDistances = np.zeros((numOfRafts, numOfFrames))
raftOrbitingAngles = np.zeros((numOfRafts, numOfFrames))

for currentFrameNum in np.arange(numOfFrames):
    for raftID in np.arange(numOfRafts):
        raftOrbitingDistances[raftID,currentFrameNum] = CalculateDistance(raftOrbitingCenters[currentFrameNum,:], raftLocations[raftID,currentFrameNum,:])
        raftOrbitingAngles[raftID,currentFrameNum] = CalculatePolarAngle(raftOrbitingCenters[currentFrameNum,:], raftLocations[raftID,currentFrameNum,:])
 

aa = np.array([3, 16, 49, 49, 21, 6, 1])
bb = np.array([498, 116, 116, 8, 4, 22, 2, 2, 4, 8, 8, 0, 2, 0, 2, 24])
Eaa = ShannonEntropy(aa)
Ebb = ShannonEntropy(bb)
