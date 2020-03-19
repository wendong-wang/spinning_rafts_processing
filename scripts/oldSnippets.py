# -*- coding: utf-8 -*-
"""
Sections:
- Kinetic Energy calculation (Gaurav)
- loading a matlab data file
"""

#%% Kinetic Energy calculation (Gaurav)
raftlocx=raftLocations[:,::5,0]
raftlocy=raftLocations[:,::5,1]
raftvelx=np.gradient(raftlocx,axis=1)
raftvely=np.gradient(raftlocy,axis=1)
raftVelnormxy = np.square(raftvely) + np.square(raftvelx)
mean_vel = raftVelnormxy.mean(axis=0)
mean_vel_fps = mean_vel.mean()
energy_fps = mean_vel_fps + (raftRadii.mean()**2)*(2*np.pi*spinSpeed*5)**2

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
        raftOrbitingDistances[raftID, currentFrameNum] = CalculateDistance(raftOrbitingCenters[currentFrameNum, :],
                                                                           raftLocations[raftID, currentFrameNum, :])
        raftOrbitingAngles[raftID, currentFrameNum] = CalculatePolarAngle(raftOrbitingCenters[currentFrameNum, :],
                                                                          raftLocations[raftID, currentFrameNum, :])

aa = np.array([3, 16, 49, 49, 21, 6, 1])
bb = np.array([498, 116, 116, 8, 4, 22, 2, 2, 4, 8, 8, 0, 2, 0, 2, 24])
Eaa = ShannonEntropy(aa)
Ebb = ShannonEntropy(bb)

