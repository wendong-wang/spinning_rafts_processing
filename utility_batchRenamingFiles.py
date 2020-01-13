# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:17:25 2018

@author: wwang

Batch renaming
"""
import glob, os

'''
Below is the primary reference: 
    
def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, 
                  os.path.join(dir, titlePattern % title + ext))
'''


path = r'C:\VideoProcessingFolder\2018-10-25b_o-D300-sym2-amp2-arcAngle180-Batch19Oct2018_Co500Au60_14mT_effusion'

#filenamePattern = '250*'
#pathAndFileOrFolderName = glob.glob(os.path.join(path, filenamePattern))
#fileOrFolderName0 = os.path.basename(pathAndFileOrFolderName[0])

filenamePattern = '100*'
fileOrFolderNames = glob.glob(filenamePattern)

partitionString = '100'

for ii in range(len(fileOrFolderNames)):
    part1, part2, part3 = fileOrFolderNames[ii].partition(partitionString)
    newName = '97' + part3
    os.rename(fileOrFolderNames[ii], newName)