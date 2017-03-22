from os import listdir
from os.path import isfile, join
import humanDetectionAndTracking as hdt
import cv2
import re
import numpy as np
import videoPreProcess as vpp
import ut_interaction


path_set1 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1"
files_set1 = np.array([f for f in listdir(path_set1) if isfile(join(path_set1,f)) and re.search('.avi',f) is not None and ut_interaction.Label(f) != 3])
files_set1_l3 = np.array([f for f in listdir(path_set1) if isfile(join(path_set1,f)) and re.search('.avi',f) is not None and ut_interaction.Label(f) == 3])
path0_set1 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1/vOut"
path1_set1 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1/vOut_0"
path2_set1 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1/vOut_1"

path_set2 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2"
files_set2 = np.array([f for f in listdir(path_set2) if isfile(join(path_set2,f)) and re.search('.avi',f) is not None and ut_interaction.Label(f) != 3])
files_set2_l3 = np.array([f for f in listdir(path_set2) if isfile(join(path_set2,f)) and re.search('.avi',f) is not None and ut_interaction.Label(f) == 3])
path0_set2 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2/vOut"
path1_set2 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2/vOut_0"
path2_set2 = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2/vOut_1"

#for file in files_set1:
#    print(join(path_set1,file))
#    vin = vpp.videoRead(join(path_set1,file), grayMode=False)
#    vDet,picks = hdt.humanDetector(vin, dispBBEn=False)
#    vOut,vOut_0,vOut_1 = hdt.humanTracking(vDet,picks)
#    vpp.videoSave(vOut,join(path0_set1,file))
#    vpp.videoSave(vOut_0,join(path1_set1,file))
#    vpp.videoSave(vOut_1,join(path2_set1,file))
for file in files_set1_l3:
    print(join(path_set1,file))
    vin = vpp.videoRead(join(path_set1,file), grayMode=False)
    vOut = np.array([cv2.resize(img,(80,112),interpolation=cv2.INTER_AREA) for img in vin])
    print(vOut.shape)
    vpp.videoSave(vOut,join(path1_set1,file))
    vpp.videoSave(vOut,join(path2_set1,file))
    

#for file in files_set2:
#    print(join(path_set2,file))
#    vin = vpp.videoRead(join(path_set2,file), grayMode=False)
#    vDet,picks = hdt.humanDetector(vin, dispBBEn=False)
#    vOut,vOut_0,vOut_1 = hdt.humanTracking(vDet,picks)
#    vpp.videoSave(vOut,join(path0_set2,file))
#    vpp.videoSave(vOut_0,join(path1_set2,file))
#    vpp.videoSave(vOut_1,join(path2_set2,file))

for file in files_set2_l3:
    print(join(path_set2,file))
    vin = vpp.videoRead(join(path_set2,file), grayMode=False)
    vOut = np.array([cv2.resize(img,(80,112),interpolation=cv2.INTER_AREA) for img in vin])
    vpp.videoSave(vOut,join(path1_set2,file))
    vpp.videoSave(vOut,join(path2_set2,file))
    