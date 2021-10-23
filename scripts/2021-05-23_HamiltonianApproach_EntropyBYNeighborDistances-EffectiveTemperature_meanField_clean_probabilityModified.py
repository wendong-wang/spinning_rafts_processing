#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:01:27 2020

@author: gardi
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.io import loadmat
# import scipy.stats
from scipy.stats import entropy
# from scipy.signal import find_peaks
from scipy import integrate
from scipy import special

import glob
import os
import shelve
import platform
import progressbar

# import cv2 as cv
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import decimal
# import progressbar
# from scipy.integrate import RK45
# from scipy.integrate import solve_ivp
# from scipy.spatial import Voronoi as scipyVoronoi
# import scipy.io
# from scipy.spatial import distance as scipy_distance

"""
This module processes the data of simulated pairwise interactions.
The maximum characters per line is set to be 120.
"""

# float_epsilon = np.finfo(float).eps


def ShannonEntropy(c):
    """calculate the Shannon entropy of 1 d data. The unit is bits """

    c_normalized = c / float(np.sum(c))
    c_normalized_nonzero = c_normalized[np.nonzero
                                        (c_normalized)]  # gives 1D array
    H = -sum(c_normalized_nonzero *
             np.log2(c_normalized_nonzero))  # unit in bits
    return H

   
def HistogramP(H, Beta):
    
    binEdgesEEDistances = np.arange(2,10,0.5).tolist() + [100]
    
    BetaH = Beta*H 
    P = np.exp(-BetaH)
    
    Hist_P = np.zeros((len(binEdgesEEDistances[:-1])))
    
    for i in range(Hist_P.shape[0] - 1) :
        Hist_P[i] = P[i*75:(i+1)*75].sum()/P[:].sum() # 75 because of 0.5R binning size
        
    Hist_P[-1] = P[(i+1)*75:].sum()/P[:].sum()

    return Hist_P, binEdgesEEDistances


def optimiseBetaAlpha(magneticFieldRotationRPS, Hist_exp, E_total, HydrodynamicsLift_Energy, E_cap_AA, E_magdp_AA,
                      Boundary_Energy_Hydro_main) :
#    KLD_min = 100
    Beta_opt = 1e18
    alpha_opt_mdp = 1
    alpha_opt_bd = 1
    cost_min = 100
    r = 1 #0.1
    alpha_max = 10 + r
    
    H_boundary = Boundary_Energy_Hydro_main.copy()
    H = E_total + HydrodynamicsLift_Energy + alpha_opt_bd*H_boundary
    H_opt = H
    Hist_P_opt, _ = HistogramP(H, 1e12)
    
    for index, Beta in enumerate(np.arange(1,10000,10)*1e11) :
        for alpha_bd in np.arange(10, alpha_max, r):  # alpha bd is the prefactor fo the boundary term
                      
#            H = E_cap_AA + alpha_mdp*E_magdp_AA + HydrodynamicsLift_Energy + alpha_bd*H_boundary
            H =  E_total + HydrodynamicsLift_Energy + alpha_bd*H_boundary
#            H =  E_total + alpha_bd*HydrodynamicsLift_Energy + H_boundary
#            H =  alpha_bd*E_cap_AA + E_magdp_AA + HydrodynamicsLift_Energy + H_boundary
            
            #### to check for lower spin speeds
            Hist_P, binEdgesEEDistances_opt = HistogramP(H, Beta)

            KLD = entropy(Hist_exp[:-1], qk = Hist_P[:-1] + 1e-3)
            cost = KLD
            
            if cost < cost_min: 
                cost_min = cost
                Beta_opt = Beta
                Hist_P_opt = Hist_P
                alpha_opt_bd = alpha_bd
                H_opt = H
                
#                print("alpha" + str(alpha_bd) + "beta" + str(Beta) + "cost_min" + str(cost_min))

    return Beta_opt, alpha_opt_mdp, alpha_opt_bd, cost_min, Hist_P_opt, binEdgesEEDistances_opt, H_opt

#%% changing of directory related to simulations
simulationFolderName = '/home/gardi/spinning_rafts_sim2' 
os.chdir(simulationFolderName)

if platform.node() == 'NOTESIT43' and platform.system() == 'Windows':
    projectDir = "D:\\simulationFolder\\spinning_rafts_sim2"
elif platform.node() == 'NOTESIT71' and platform.system() == 'Linux':
    projectDir = r'/media/wwang/shared/spinning_rafts_simulation/spinning_rafts_sim2'
else:
    projectDir = os.getcwd()

if projectDir != os.getcwd():
    os.chdir(projectDir)

import scripts.functions_spinning_rafts as fsr

dataDir = os.path.join(projectDir, 'data')

capSym6Dir = os.path.join(projectDir, '2019-05-13_capillaryForceCalculations-sym6')
#%%  change to directory contatining experiments data

#rootFolderNameFromWindows = '/media/gardi/DataMPI_10/Data_PhantomMiroLab140'Integrate[fr[\[Theta]], {\[Theta], 0, 2*Pi}]
#rootFolderNameFromWindows = '/media/gardi/VideoFiles_Raw_PP/Data_Camera_Basler-acA2500-60uc/'
rootFolderNameFromWindows = '/home/gardi/Rafts/Experiments Data/Data_Camera_Basler-acA2500-60uc/'
os.chdir(rootFolderNameFromWindows)
rootFolderTreeGen = os.walk(rootFolderNameFromWindows)
_, mainFolders_experiments, _ = next(rootFolderTreeGen) 


#%% loading all the data in a specific main folder into mainDataList
# at the moment, it handles one main folder at a time. 

#for mainFolderID in np.arange(0,1):
#    os.chdir(mainFolders[mainFolderID])

mainFolderID_experiments = 2 # 5
os.chdir(mainFolders_experiments[mainFolderID_experiments])
expDataDir = os.getcwd()
dataFileList_experiments = glob.glob('*.dat')
#dataFileList_experiments = glob.glob('*.dat')[128:252]
dataFileList_experiments.sort()
dataFileListExcludingPostProcessed_experiments = dataFileList_experiments.copy()
numberOfPostprocessedFiles_experiments = 0

mainDataList_experiments = []
variableListsForAllMainData_experiments = []

for dataID in range(len(dataFileList_experiments)):
#for dataID in range(126, 252):
    dataFileToLoad_experiments = dataFileList_experiments[dataID].partition('.dat')[0]
    
    if 'postprocessed' in dataFileToLoad_experiments:
        # the list length changes as items are deleted
        del dataFileListExcludingPostProcessed_experiments[dataID - numberOfPostprocessedFiles_experiments] 
        numberOfPostprocessedFiles_experiments = numberOfPostprocessedFiles_experiments + 1
        continue
    
    tempShelf = shelve.open(dataFileToLoad_experiments)
    variableListOfOneMainDataFile_experiments = list(tempShelf.keys())
    
    expDict = {}
    for key in tempShelf:
        try:
            expDict[key] = tempShelf[key]
        except TypeError:
            pass
    
    tempShelf.close()
    mainDataList_experiments.append(expDict)
    variableListsForAllMainData_experiments.append(variableListsForAllMainData_experiments)
    
    
#    # go one level up to the root folder
#    os.chdir('..')

#%% capillary energy landscape extraction from .mat file

os.chdir(capSym6Dir)

#x = loadmat('ResultsCombined_L4_amp2_arcAngle30_ccDist301-350um-count50_rotAngle61_bathRad500.mat')
#capillary_data_341_to_1300 = loadmat('Results_sym6_arcAngle30_ccDistance341to1300step1um_angleCount361_errorPower-10_treated.mat')
capillary_data_341_to_1600 = loadmat('Results_sym6_arcAngle30_ccDistance341to1600step1um_angleCount361_errorPower-10_treated.mat')
capillary_data_1600_to_8000 = loadmat('Results_sym6_arcAngle30_ccDistance1600to8000step1um_angleCount361_errorPower-10_treated.mat')
capillary_data_301_to_350 = loadmat('ResultsCombined_L4_amp2_arcAngle30_ccDist301-350um-count50_rotAngle61_bathRad500.mat')

#fig = plt.figure()
#ax = fig.gca(projection='3d')

# Make data.
EEDistances_capillary = np.arange(1, 1001, 1) # E-E distamce unit um
orientationAngles_capillary = np.arange(0, 360, 1) # relative orientationm angle in degrees
#E_capillary_341_to_1300 = capillary_data_341_to_1300['energyScaledToRealSurfaceTensionRezeroed']/1e15 # unit: J
E_capillary_341_to_1600 = capillary_data_341_to_1600['energyScaledToRealSurfaceTensionRezeroed']/1e15 # unit: J
E_capillary_1600_to_8000 = capillary_data_1600_to_8000['energyScaledToRealSurfaceTensionRezeroed']/1e15 # unit: J
E_capillary_301_to_350 = (capillary_data_301_to_350['netEnergy_2Rafts'].transpose() - capillary_data_301_to_350['netEnergy_2Rafts'].transpose()[49,0] )/1e15  # unit: J
#E_capillary_combined = np.vstack((E_capillary_301_to_350[:40,:], E_capillary_341_to_1300[:,:61]))
E_capillary_combined = np.vstack((E_capillary_301_to_350[:40,:], E_capillary_341_to_1600[:,:61], E_capillary_1600_to_8000[:,:61],))
E_capillary_combined_All360 = np.hstack((E_capillary_combined[:,:60],E_capillary_combined[:,:60],
                                         E_capillary_combined[:,:60],E_capillary_combined[:,:60],
                                         E_capillary_combined[:,:60],E_capillary_combined[:,:60]))


# %% magnetic force and torque calculation:
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m

# from the data 2018-09-28, 1st increase:
# (1.4e-8 A.m**2 for 14mT), (1.2e-8 A.m**2 for 10mT), (0.96e-8 A.m**2 for 5mT), (0.78e-8 A.m**2 for 1mT)
# from the data 2018-09-28, 2nd increase:
# (1.7e-8 A.m**2 for 14mT), (1.5e-8 A.m**2 for 10mT), (1.2e-8 A.m**2 for 5mT), (0.97e-8 A.m**2 for 1mT)
magneticMomentOfOneRaft = 1e-8  # unit: A.m**2

orientationAngles = np.arange(0, 361)  # unit: degree;
orientationAnglesInRad = np.radians(orientationAngles)

magneticDipoleEEDistances = np.arange(0, 10001) / 1e6  # unit: m

radiusOfRaft = 1.5e-4  # unit: m

magneticDipoleCCDistances = magneticDipoleEEDistances + radiusOfRaft * 2  # unit: m

magDpEnergy = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: J
magDpForceOnAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpForceOffAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpTorque = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N.m

for index, d in enumerate(magneticDipoleCCDistances):
    magDpEnergy[index, :] = \
        miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 3)

E_magDipole = magDpEnergy

#%% HYdrodynamic life force energy and potential ebergy due to boundary
miu = 1e-3 # dynamic viscosity of water units Pa.s = 1e-3 N.s/m^2
densityOfWater = 1e3  # units : 1000 kg/m^3 = 1e-15 kg/um^3
Hydrodynamics_EEDistances = np.arange(0, 10001) / 1e6  # unit: m
radiusOfRaft = 1.5e-4  # unit: m

cuttOff_distance = 1300 #um

numOfBatches = 60
Hydrodynamics_CCDistances = Hydrodynamics_EEDistances + radiusOfRaft * 2  # unit: m
HydrodynamicsLift_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J

for dataID in progressbar.progressbar(range(0, numOfBatches)) :
#    magneticFieldRotationRPS = magneticFieldRotationRPS_all[dataID]
    magneticFieldRotationRPS = dataID + 11
    for index, d in enumerate(Hydrodynamics_CCDistances): # d is in units: m
        HydrodynamicsLift_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)/(2*(d)**2)
    

#%% angle averaged energies
        
cuttOff_distance = 1300 #um
cuttOff_distance_cap = 7500 #um
cuttOff_time = 2500 # frame number

miu = 1e-3 # dynamic viscosity of water units Pa.s = 1e-3 N.s/m^2
densityOfWater = 1e3  # units : 1000 kg/m^3 = 1e-15 kg/um^3
Beta = 1e14

E_total = np.zeros((cuttOff_distance, 1))
#H = np.zeros((cuttOff_distance , numOfBatches))
#P = np.zeros((cuttOff_distance , numOfBatches))

E_cap_AA = E_capillary_combined_All360.mean(axis=1)
E_magdp_AA = E_magDipole[:,0:360].mean(axis=1)


E_total = E_cap_AA[0:cuttOff_distance] + 0.5*E_magdp_AA[0:cuttOff_distance]

#%% Boundary energy calculation

Boundary_Energy_Hydro = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J
p_r_uniform = np.zeros((cuttOff_distance, len(mainDataList_experiments)))  # uniform porbability distribution

r_thresh = 50*radiusOfRaft # fixing r_thresh as arena size # unit m

for dataID_experiments in progressbar.progressbar(range(1, len(mainDataList_experiments))) :
    for index, d in enumerate(Hydrodynamics_CCDistances[:cuttOff_distance]): 
        stepsize = 1
        range_r = range(int(3*d*1e6/2), int(r_thresh*1e6), stepsize)
#        p_r_uniform[index, dataID_experiments] = 0.5e-6*numOfRafts/((r_thresh - 3*d/2)) # uniform porbability distribution # unit (um)-1
        r_prime = np.zeros(len(range_r)) # unit microns
        fraction_E = np.zeros(len(range_r))
        energy_term = 0 # unit J
        
        for j, r in enumerate(range_r): # unit of r: micron
            r_prime[j] = int(r - (d/2)*1e6)
            
            # energy term = sigma ((E_cap + E_mag-dp + E_hydro)*r_prime) # unit J.um
            energy_term += -2*(E_cap_AA[int(r_prime[j] - 300)] \
                            + E_magdp_AA[int(r_prime[j] - 300)] + \
                            + HydrodynamicsLift_Energy[int(r_prime[j] - 300), dataID_experiments]) \
                            * r_prime[j]
            
        # uniform porbability density normalized # unit unit less (constant probability density for homoenous distributions of disks)
        p_r_uniform[index, dataID_experiments] = 1e6/r_prime.sum()
        # Boundary energy divided by normalization factor of probability distribution
        # Bundary energy = (energy term)/sigma(r_prime) # unit J
#        Boundary_Energy_Hydro[index, dataID_experiments] = energy_term*2e-6*np.pi
        Boundary_Energy_Hydro[index, dataID_experiments] = energy_term/(r_prime.sum())
        
        
#        Boundary_Energy_Hydro[index, dataID_experiments] = -2*1e-6* \
#                                                    (E_cap_AA[int(d*1e6) - 300: int((r_thresh - d/2)*1e6) - 300].sum() \
#                                                    + E_magdp_AA[int(d*1e6) - 300: int((r_thresh - d/2)*1e6) - 300].sum() \
#                + HydrodynamicsLift_Energy[int(d*1e6) - 300: int((r_thresh - d/2)*1e6) - 300, dataID_experiments].sum())/ \
#                                                    (2*(r_thresh - 3*d/2)) 
        

np.savetxt("BoundaryEnergy", Boundary_Energy_Hydro, delimiter=",")

#%% load data corresponding to a specific experiment (subfolder or video) into variables
#os.chdir("/home/gardi/spinning_rafts_sim2/2019-05-13_capillaryForceCalculations-sym6/alpha(20-200)-Beta-optimisation_Lintegrated_mean(area)-10R-20R_hydro_customKLD-qk-calculated")
binEdgesNeighborDistances = np.arange(2,10,0.5).tolist() + [100]
Beta_optm = np.zeros((len(mainDataList_experiments)))
L_optm = np.zeros((len(mainDataList_experiments)))
KLD_minm = np.zeros((len(mainDataList_experiments)))
Hist_P_optm = np.zeros((len(binEdgesNeighborDistances[:-1]), len(mainDataList_experiments)))
DeltaEntropy_minm = np.zeros((len(mainDataList_experiments)))
Std_diff_minm = np.zeros((len(mainDataList_experiments)))
KLD = np.zeros((len(mainDataList_experiments)))
entropy_exp = np.zeros((len(mainDataList_experiments)))
alpha_optm = np.zeros((len(mainDataList_experiments)))
alpha_optm_bd = np.zeros((len(mainDataList_experiments)))

H_opt = np.zeros((cuttOff_distance , len(mainDataList_experiments)))
Force = np.zeros((cuttOff_distance , len(mainDataList_experiments)))

count_neighborDistances = np.zeros((len(binEdgesNeighborDistances[:-1]), len(mainDataList_experiments)))
neighborCount_avg = np.zeros((len(mainDataList_experiments)))
neighborCount_Weightedavg = np.zeros((len(mainDataList_experiments)))


sizeOfArenaInRadius = 15000/150 # 1.5m square arena, 150 um raft radius
binwidth  = 0.5
binEdgesR = np.arange(0,sizeOfArenaInRadius/2,binwidth).tolist()
c_r = np.zeros((len(binEdgesR) - 1, len(mainDataList_experiments)))
r = np.zeros((len(binEdgesR) - 1, len(mainDataList_experiments)))

#Boundary_Energy_Hydro = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J

#density_rafts = np.zeros((len(mainDataList_experiments)))
r0 = np.zeros((len(mainDataList_experiments)))
#p_r_uniform = np.zeros((cuttOff_distance, len(mainDataList_experiments))) # uniform porbability distribution

mode_r= np.zeros((len(mainDataList_experiments)))
mean_r= np.zeros((len(mainDataList_experiments)))
std_r = np.zeros((len(mainDataList_experiments)))

std_r3 = np.zeros((len(mainDataList_experiments)))
mean_r3 = np.zeros((len(mainDataList_experiments)))

std_r13 = np.zeros((len(mainDataList_experiments)))
mean_r13 = np.zeros((len(mainDataList_experiments)))

numOfRafts = 218
r_mean_individual = np.zeros((numOfRafts, len(mainDataList_experiments)))
r_std_individual = np.zeros((numOfRafts, len(mainDataList_experiments)))


for dataID_experiments in progressbar.progressbar(range(0,len(mainDataList_experiments))) :
#for dataID_experiments in progressbar.progressbar(range(0, 12)) :
    numOfRafts = 218 # ideally read from the data file
    date_experiments = mainDataList_experiments[dataID_experiments]['date']
    batchNum_experiments = mainDataList_experiments[dataID_experiments]['batchNum']
    spinSpeed_experiments = mainDataList_experiments[dataID_experiments]['spinSpeed']
    numOfRafts_experiments = mainDataList_experiments[dataID_experiments]['numOfRafts']
    numOfFrames_experiments = mainDataList_experiments[dataID_experiments]['numOfFrames']
    raftRadii_experiments = mainDataList_experiments[dataID_experiments]['raftRadii']
    raftLocations_experiments = mainDataList_experiments[dataID_experiments]['raftLocations']
# =============================================================================
# #    raftOrbitingCenters_experiments = mainDataList[dataID_experiments]['raftOrbitingCenters']
# =============================================================================
    raftOrbitingDistances_experiments = mainDataList_experiments[dataID_experiments]['raftOrbitingDistances']
#    raftOrbitingAngles_experiments = mainDataList[dataID_experiments]['raftOrbitingAngles']
#    raftOrbitingLayerIndices_experiments = mainDataList[dataID_experiments]['raftOrbitingLayerIndices']
    magnification_experiments = mainDataList_experiments[dataID_experiments]['magnification']
    commentsSub_experiments = mainDataList_experiments[dataID_experiments]['commentsSub']
    #dataID_experiments = 3
    variableListFromProcessedFile_experiments = list(mainDataList_experiments[dataID_experiments].keys())

    for key, value in mainDataList_experiments[dataID_experiments].items(): # loop through key-value pairs of python dictionary
        globals()[key] = value

    ######### load all variables from postprocessed file corresponding to the specific experiment above

    analysisType_experiments = 5 # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi
    
    shelveDataFileName_experiments = expDataDir + '/' + date_experiments + '_' + str(numOfRafts_experiments) + 'Rafts_' + str(batchNum_experiments) + '_' + str(spinSpeed_experiments) + 'rps_' + str(magnification_experiments) + 'x_' + 'postprocessed' + str(analysisType_experiments)
        
    shelveDataFileExist = glob.glob(shelveDataFileName_experiments +'.dat')
    
    if shelveDataFileExist:
        print(shelveDataFileName_experiments + ' exists, load additional variables. ' )
        tempShelf = shelve.open(shelveDataFileName_experiments)
        variableListFromPostProcessedFile_experiments = list(tempShelf.keys())
        
        for key in tempShelf: # just loop through all the keys in the dictionary
            globals()[key] = tempShelf[key]
        
        tempShelf.close()
        print('loading complete.' )
        
    elif len(shelveDataFileExist) == 0:
        print(shelveDataFileName_experiments + ' does not exist')
    
    ######## experimental distributixon ######################
    binEdgesNeighborDistances = np.arange(2,10,0.5).tolist() + [100]
    neighborDistancesList = np.concatenate(dfNeighborsAllFrames['neighborDistances'].iloc[-numOfRafts_experiments:].tolist())
    neighborDistancesList_all = np.concatenate(dfNeighborsAllFrames['neighborDistances'].iloc[:].tolist())
    count_neighborDistances[:, dataID_experiments], _ = np.histogram(np.asarray(neighborDistancesList_all), binEdgesNeighborDistances)
#    entropyByNeighborDistances[currentFrameNum] = ShannonEntropy(count_neighborDistances)
    
    neighborCount_avg[dataID_experiments] = dfNeighborsAllFrames['neighborCount'].iloc[-numOfRafts_experiments:].mean()
    neighborCount_Weightedavg[dataID_experiments] = dfNeighborsAllFrames['neighborCountWeighted'].iloc[-numOfRafts_experiments:].mean()

    entropy_exp[dataID_experiments] = ShannonEntropy(count_neighborDistances[:, dataID_experiments])
    
   ################## optimise Beta ###################
    dataID = dataID_experiments
    Beta = 1e14
        
    Beta_optm[dataID], alpha_optm[dataID], alpha_optm_bd[dataID], KLD_minm[dataID], \
                                                        Hist_P_optm[:, dataID], _ , H_opt[:, dataID]= optimiseBetaAlpha(
            spinSpeed_experiments, count_neighborDistances[:, dataID], E_total, 
            HydrodynamicsLift_Energy[:cuttOff_distance,dataID], E_cap_AA[:cuttOff_distance],
                        E_magdp_AA[:cuttOff_distance], Boundary_Energy_Hydro[:cuttOff_distance, dataID]) # E_magdp_AA)

    
    KLD[dataID] = entropy(count_neighborDistances[:, dataID], qk = Hist_P_optm[:,dataID])
    
    Force[:,dataID] = -np.gradient(H_opt[:,dataID])
    
    ######## save histograms for each spin speed ###################
    fig, ax = plt.subplots(1,1, figsize = (10,15))
    ax.plot(binEdgesNeighborDistances[:-1], Hist_P_optm[:,dataID]/Hist_P_optm[:,dataID].sum(), label='calculated_alpha-beta-opt')
    ax.plot(binEdgesNeighborDistances[:-1], count_neighborDistances[:, dataID]/count_neighborDistances[:, dataID].sum(), label='experiments')
    ax.set_xlabel('EEDistances',  {'size': 15})
    ax.set_ylabel('count', {'size': 15})
    ax.set_title('histogram of EEDistances at ' + str(spinSpeed_experiments) + ' rps')
    ax.legend()
    
    filename = 'histogram of EEDistances at ' + str(int(spinSpeed_experiments)) + ' rps'
    plt.savefig(filename)
    
    plt.close()

###### save a plot of sum of forces vs d for all spinspeeds
fig1, ax1 = plt.subplots(1,1, figsize = (10,15))
ax1.hlines(0, 60/150, 650/150, label='zerosLIne')
for i in progressbar.progressbar(range(4,15)) :
    ax1.plot(np.arange(60,650)/150, Force[60:650,i], '.', label = str(i + 11) + 'rps' )

ax1.set_xlabel('EEDistances(R)',  {'size': 15})
ax1.set_ylabel('Sum of Forces', {'size': 15})
ax1.set_title('Sum of forces vs edge-edge distance for all rps')
ax1.legend()

filename = 'Sum of forces vs edge-edge distance for all rps '
plt.savefig(filename)

plt.close()
    
#%% Emtropy calculation coreespinding to the optimised parameters
entropy_opt = np.zeros((60))
Std_optm = np.zeros((60))
for i in range(0,60) :
    entropy_opt[i] = ShannonEntropy(Hist_P_optm[:,i])
    Std_optm[i] = (Hist_P_optm[:,i]/Hist_P_optm[:,i].sum()).std()

#%% saving plots 
# entropy    
fig, ax = plt.subplots(1,1, figsize = (10,15))
#    ax.bar(binEdgesNeighborDistances[:-1], Hist_P_optm[:,dataID]/Hist_P_optm[:,dataID].sum(), align = 'edge', width = 0.25)
ax.plot(range(11,71), entropy_exp, 'o-', label='experiments')
ax.plot(range(11,71), entropy_opt, 'o-', label='calculated')
ax.set_xlabel('Spin speed',  {'size': 15})
ax.set_ylabel('Entropy', {'size': 15})
ax.set_title('entropy vs spin speed')
ax.legend()
    
filename = 'Entropy_exp_calculated-vs-spinspeed'
plt.savefig(filename)

plt.close()

# beta
fig, ax = plt.subplots(1,1, figsize = (10,15))
#    ax.bar(binEdgesNeighborDistances[:-1], Hist_P_optm[:,dataID]/Hist_P_optm[:,dataID].sum(), align = 'edge', width = 0.25)
ax.plot(range(11,71), Beta_optm, 'o-', label='Beta')
ax.set_xlabel('Spin speed',  {'size': 15})
ax.set_ylabel('Beta', {'size': 15})
ax.set_title('beta vs spinspeed')
ax.legend()
    
filename = 'Beta'
plt.savefig(filename)

plt.close()

# alpha mdp
fig, ax = plt.subplots(1,1, figsize = (10,15))
#    ax.bar(binEdgesNeighborDistances[:-1], Hist_P_optm[:,dataID]/Hist_P_optm[:,dataID].sum(), align = 'edge', width = 0.25)
ax.plot(range(11,71), alpha_optm, 'o-', label='alpha')
ax.set_xlabel('Spin speed',  {'size': 15})
ax.set_ylabel('Alpha', {'size': 15})
ax.set_title('alpha_mdp vs spinspeed')
ax.legend()
    
filename = 'Alpha_mdp'
plt.savefig(filename)

plt.close()

# alpha boundary
fig, ax = plt.subplots(1,1, figsize = (10,15))
#    ax.bar(binEdgesNeighborDistances[:-1], Hist_P_optm[:,dataID]/Hist_P_optm[:,dataID].sum(), align = 'edge', width = 0.25)
ax.plot(range(11,71), alpha_optm_bd, 'o-', label='alpha')
ax.set_xlabel('Spin speed',  {'size': 15})
ax.set_ylabel('Alpha', {'size': 15})
ax.set_title('alpha_bd vs spinspeed')
ax.legend()
    
filename = 'Alpha_bd'
plt.savefig(filename)

plt.close()


# KLD
fig, ax = plt.subplots(1,1, figsize = (10,15))
#    ax.bar(binEdgesNeighborDistances[:-1], Hist_P_optm[:,dataID]/Hist_P_optm[:,dataID].sum(), align = 'edge', width = 0.25)
ax.plot(range(11,71), KLD, 'o-', label='KLD')
ax.set_xlabel('Spin speed',  {'size': 15})
ax.set_ylabel('KLD', {'size': 15})
ax.set_title('KLD vs spinspeed')
ax.legend()
    
filename = 'KLD'
plt.savefig(filename)

plt.close()

#%% storing data to csv file
dfpairwiseDistributions = pd.DataFrame(columns = ['spinspeed'])

dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(Beta_optm, name = 'Beta'), how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(L_optm, name = 'L_optm/R'), how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(DeltaEntropy_minm, name = 'DeltaEntropy_minm'),
                                                                                               how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(entropy_opt, name = 'entropy_opt'), how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(entropy_exp, name = 'entropy_exp'), how = 'outer')
#dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(Std_optm, name = 'Std_opt'), how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(KLD, name = 'KLD'), how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(alpha_optm, name = 'Alpha_mdp'), how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(alpha_optm_bd, name = 'Alpha_bd'), how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(neighborCount_avg, name = 'neighborCount_avg'), 
                                                                                               how = 'outer')
dfpairwiseDistributions = dfpairwiseDistributions.join(pd.Series(neighborCount_Weightedavg, name = 
                                                                 'neighborCount_weightedavg'), how = 'outer')


dataFileName = mainFolders_experiments[mainFolderID_experiments]
dfpairwiseDistributions.to_csv(dataFileName + 'pairwise_dist_KLD_BetaAlpha-LongOpt_NumIntegratedPot_Hydro_Lexp.csv'
                                                                                                           , index = False)

np.savetxt("Histogram_calculated_BetaAlpha-LOng_Opt.csv", Hist_P_optm, delimiter=",")
np.savetxt("Histogram_experiments.csv", count_neighborDistances, delimiter=",")

np.savetxt("Hamiltonian_calculated_BetaAlpha-LOng_Opt.csv", H_opt, delimiter=",")
np.savetxt("E_boundary_BetaAlpha-LOng_Opt.csv", Boundary_Energy_Hydro[:cuttOff_distance,:], delimiter=",")
np.savetxt("Alpha_X_E_boundary_BetaAlpha-LOng_Opt.csv", alpha_optm_bd*Boundary_Energy_Hydro[:cuttOff_distance,:],
                                                                                                       delimiter=",")
np.savetxt("Alpha_X_E_mdp_BetaAlpha-LOng_Opt.csv", 
           alpha_optm*(np.repeat(E_magdp_AA[:cuttOff_distance],60)).reshape(cuttOff_distance,60), delimiter=",")
np.savetxt("Pairwise_BetaAlpha-LOng_Opt.csv", (np.repeat(E_total,60)).reshape(cuttOff_distance, 60) \
                                                   + HydrodynamicsLift_Energy[:cuttOff_distance,:], delimiter=",")
np.savetxt("Cap-magdp_BetaAlpha-LOng_Opt.csv", E_total, delimiter=",")
np.savetxt("HydroLift_BetaAlpha-LOng_Opt.csv", HydrodynamicsLift_Energy[:cuttOff_distance,:], delimiter=",")
np.savetxt("Sum-of-forces_calculated_BetaAlpha-LOng_Opt.csv", Force, delimiter=",")

np.savetxt("Threshold-distances-vs-spinspeed.csv", r0, delimiter=",")
np.savetxt("uniform-prob-value-d-omega.csv", p_r_uniform, delimiter=",")


#%% some extra calculations to compare alpha_bd abd beta with physical quantities like mean and std of neighbor distances
Std_sum_pairwise = np.zeros((len(mainDataList_experiments)))
sum_std_pairwise = np.zeros((len(mainDataList_experiments)))
sum_std_cap = np.zeros((len(mainDataList_experiments)))
sum_std_magdp = np.zeros((len(mainDataList_experiments)))
mean_pairwise = np.zeros((len(mainDataList_experiments)))


PhysicalBoundary_Energy = np.zeros((360, len(mainDataList_experiments)))
PhysicalBoundary_Energy_std = np.zeros((len(mainDataList_experiments)))

boundary_radialDistance = np.zeros(360) # distance of boundary from centre of arena as a function of angle theta
r_boundary = 50 # unit R

for theta in np.arange(0,45):
    boundary_radialDistance[theta] = r_boundary/np.cos(theta*np.pi/180) 

for theta in np.arange(45,90):
    boundary_radialDistance[theta] = r_boundary/np.cos((90 - theta)*np.pi/180)

for theta in np.arange(90,360):
    boundary_radialDistance[theta] = boundary_radialDistance[theta - 90]

max_sum = np.zeros((len(mainDataList_experiments)))
min_sum = np.zeros((len(mainDataList_experiments)))
for dataID in range(len(mainDataList_experiments)):
    r_mean = int(mean_r[dataID]*radiusOfRaft*1e6) - 300
    max_sum[dataID] = (E_capillary_combined_All360[r_mean,:] + E_magDipole[r_mean, :-1]).max()
    min_sum[dataID] = (E_capillary_combined_All360[r_mean,:] + E_magDipole[r_mean, :-1]).min()
    Std_sum_pairwise[dataID] = (E_capillary_combined_All360[r_mean,:] + E_magDipole[r_mean, :-1]).std()
    sum_std_pairwise[dataID] = E_capillary_combined_All360[r_mean,:].std() + E_magDipole[r_mean, :-1].std()
    sum_std_cap[dataID] = E_capillary_combined_All360[r_mean,:].std()
    sum_std_magdp[dataID] = E_magDipole[r_mean, :-1].std()
    mean_pairwise[dataID] = E_total[r_mean]
#    Std_sum_pairwise[dataID] = (F_mdp[r_mean,:].std())/mean_r[dataID]**2
#    magneticFieldRotationRPS = dataID + 11
#    dist = (boundary_radialDistance - r0[dataID])*radiusOfRaft
#    for index, d in enumerate(dist): # d is in units: m
#        PhysicalBoundary_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)\
#                                                    *(1/(2*(d)**2))
#    PhysicalBoundary_Energy_std[dataID] = PhysicalBoundary_Energy[:,dataID].std()
#    
    
spinSpeeds = np.arange(11,71,1)

X = np.genfromtxt("2019-07-09_o-D300-sym6-amp2-arcAngle30-19Jun2019_Co500Au64_16.5mT_manyRafts_Experimentspairwise_dist_KLD_BetaAlpha-LongOpt_NumIntegratedPot_Hydro_Lexp.csv", delimiter=',')
Beta = X[1:,1] # getting Beta from the hamiltonian data
Alpha_bd = X[1:,7] # getting Alpha_bd from hamiltonian data

fig, ax = plt.subplots(1,1, figsize = (10,15))
#ax.plot(range(11,71), sum_std_pairwise*spinSpeeds, 'o-', label='standard deviations of energy per time')
ax.plot(range(11,71), 1/std_r**2, 'o-', label='1/(standard deviations of neighbor distances)')
ax.plot(range(11,71), 1/r_std_individual.mean(axis=0)**2, 'o-', label='1/(mean (N) of std(time) of neighbor dist)')
ax.plot(range(11,71), Beta*5e-15, 'o-', label='5e-15*Beta for E_bd fit alpha in [1,5]')
ax.set_xlabel('spinspeed (rps)',  {'size': 15})
ax.set_ylabel('omega*std', {'size': 15})
ax.set_title('1/(std of neighbor distances) vs omega')
ax.legend()

filename = '(standard deviation of r)-1 and beta vs spinspeed.png'
plt.savefig(filename)

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.plot(range(11,71), mean_r, 'o-', label='mean neighbor distance')
ax.plot(range(11,71), mode_r, 'o-', label='neighbor distance with most frequency (from hist)')
ax.plot(range(11,71), 2*Alpha_bd, 'o-', label='alpha_bd from experiments')
ax.set_xlabel('spinspeed (rps)',  {'size': 15})
ax.set_ylabel('mean r, alpha', {'size': 15})
ax.set_title('r_mean-r_mode_alpha_bd_vs_spinspeeds')
ax.legend()

filename = 'r_mean-r_mode_alpha_bd_vs_spinspeeds'
plt.savefig(filename)


np.savetxt("std_angle_cap-magdp_power_vs_omega.csv", sum_std_pairwise*spinSpeeds, delimiter=',')
np.savetxt("r_mean.csv", mean_r, delimiter=',')
np.savetxt("r_mode.csv", mode_r, delimiter=',')
np.savetxt("r_std_all.csv", std_r, delimiter=',')
np.savetxt("r_std.csv", r_std_individual, delimiter=',')
