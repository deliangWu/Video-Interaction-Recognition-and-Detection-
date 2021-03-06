'''video pre processing common files, such as loading videos, video normalizations, video to batchs, etc.'''
from __future__ import division
import numpy as np
import sys
sys.path.insert(1,'/home/wdl/opencv/lib')
import cv2
from functools import reduce
import time

def videoRead(fileName,grayMode=True):
    cap = cv2.VideoCapture(fileName)
    ret,frame = cap.read()
    video = []
    while(ret):
        if grayMode == True:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        video.append(frame)
        ret,frame = cap.read()
    video = np.array(video)
    return video 

#def videoRead(fileName,grayMode=True):
#    cap = cv2.VideoCapture(fileName)
#    ret,frame = cap.read()
#    frmNo = 0
#    video = []
#    while(ret):
#        frm_r = np.reshape(frame,(-1,)+frame.shape)
#        if frmNo == 0:
#            video = frm_r
#        else:
#            video = np.append(video,frm_r)
#        ret,frame = cap.read()
#        frmNo +=1
#        if (frmNo%100 == 0):
#            print(frmNo)
#    return video 
    
def videoNorm(video,normEn = True):
    vmax = np.amax(video)
    vmin = np.amin(video)
    if normEn == True:
        video = (video - vmin) / max(vmax-vmin,0.00001)
    else:
        video = video / 255
    return video 

def videoNorm1(video,normMode = 0):
    video = videoNorm(video)
    mean = np.mean(video)
    std = np.std(video)
    if normMode == 0:
        video = (video - mean) / std
    elif normMode == 1:
        video = video - mean
    return video 

def videoPlay(video,fps = 25):
    cv2.namedWindow('Video Player',cv2.WINDOW_AUTOSIZE)
    video = np.reshape(video,(-1,)+video.shape[-3:])
    for i,img in enumerate(video):
        img_show = img.copy()
        cv2.putText(img_show, str(i), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Video Player',img_show)
        if cv2.waitKey(int(1000/fps)) == 27:
            break
    cv2.destroyAllWindows()

def videoToImages(video):
    cv2.namedWindow('Video Player',cv2.WINDOW_AUTOSIZE)
    video_copy = video.copy()
    images = video_copy[0]
    for i in range(1,video_copy.shape[0]):
        images = np.append(images,video_copy[i],axis = 1)
    while (True):
        cv2.imshow('Video Player',images)
        if cv2.waitKey(40) == 27:
            break
    cv2.destroyAllWindows()
    
def videofliplr(videoIn):
    videoOut = np.array([np.fliplr(img) for img in videoIn])
    return videoOut

def downSampling(video,n=2):
    frameN = video.shape[0]
    #if (frameN > n):
    #    sample = np.sort(np.random.randint(0,frameN,n))
    #else:
    #    sample = np.sort(np.random.randint(0,frameN,int(frameN/16)*16))
    if (frameN > 96):
        video = video[frameN//2-48:frameN//2+48]
    frameN = video.shape[0]
    if n == 0:
        sample = np.arange(np.random.randint(int(frameN/16)),frameN,int(frameN/16)) 
    else:
        if frameN >= 64:
            sample = np.arange(np.random.randint(n),frameN,n) 
        elif frameN >= 48:
            sample = np.arange(np.random.randint(n),frameN,n)
        elif frameN >= 32:
            sample = np.arange(np.random.randint(n),frameN,n)
        else:
            sample = np.arange(np.random.randint(n),frameN,1)
    return video[sample]

def videoSave(video,fileName):
    frmSize = (video.shape[2],video.shape[1])
    if cv2.__version__  == '3.2.0':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fileName, fourcc, 20.0,frmSize)
    else:
        out = cv2.VideoWriter(fileName, cv2.cv.CV_FOURCC('X','V','I','D'),20,frmSize)
        
    for i in range(video.shape[0]):
        out.write(video[i])
        cv2.waitKey(33)
    out.release()

def videoSimplify(videoIn):
    frames = videoIn
    firstFrame = True
    frameAfterSimpilfied = 0
    for i in range(frames.shape[0]):
        img = frames[i] 
        if i > 0:
            # calculate the difference between two adjacent frames
            img_diff = cv2.absdiff(img,img_pre)
            sum_diff = np.sum(img_diff)/1000
            
            # save current frame if it's abs_diff larger than a threshold 
            if sum_diff > 200:
                img_save_4d = np.reshape(img,((1,) + img.shape))
                if firstFrame:
                    videoOut = img_save_4d
                    firstFrame = False 
                else:
                    videoOut = np.append(videoOut,img_save_4d,axis=0)
                frameAfterSimpilfied += 1
        img_pre = img
    
    if frameAfterSimpilfied < 64:
        videoOut = videoIn
    return videoOut

def batchFormat(videoIn,cropEn = True):
    videoBatch = []
    i = 0
    while(True):
        if i*8 + 16 > videoIn.shape[0]:
            break
        seq = np.arange(i*8,i*8 + 16)
        videoBatch.append(videoIn[seq])
        i += 1
    videoBatch = np.array(videoBatch)
    clips = videoBatch.shape[0]
    assert clips > 0, 'The Number of frames of input videos in less than 16, the size of videoIn is '+ str(videoIn.shape[0])
    if clips == 1:
        index = [0,0,0]
    elif clips == 2:
        index = [0,1,1]
    else:
        index = range(int(clips/2)-1,int(clips/2)+2)
    if cropEn is True:
        return videoBatch[index]
    else:
        return videoBatch
    #return np.array([videoIn])

def videoFormat(batchIn):
    return np.reshape(batchIn,((batchIn.shape[0] * batchIn.shape[1]),) + batchIn.shape[2:5])

def videoRezise(videoIn,frmSize):
    def imgResize(img,frmSize):
        bg = np.array([119,136,153] * frmSize[0] *frmSize[1],dtype=np.uint8)
        imgOut = np.reshape(bg, (frmSize[0],frmSize[1],3))
        if frmSize[2] ==1:
            imgOut = cv2.cvtColor(imgOut,cv2.COLOR_BGR2GRAY)
        whRatio = float(img.shape[1]) / img.shape[0]
        refRatio = float(frmSize[1]) / frmSize[0]
        if whRatio < refRatio * 0.8:
            imgResized = cv2.resize(img,(int(img.shape[1] * frmSize[0] / (img.shape[0] * 2)) * 2 ,frmSize[0]), interpolation= cv2.INTER_AREA)
            imgOut[:, int((frmSize[1] - imgResized.shape[1])/2) : int((frmSize[1] + imgResized.shape[1])/2)] = imgResized
        elif whRatio > refRatio * 1.2:
            cropWidth = img.shape[0] * 1.2 * refRatio
            imgCrop = img[:,int((img.shape[1] - cropWidth)/2):int((img.shape[1] + cropWidth)/2)]
            imgOut = cv2.resize(imgCrop,(frmSize[1], frmSize[0]), interpolation= cv2.INTER_AREA)
        else:
            imgOut = cv2.resize(img,(frmSize[1], frmSize[0]), interpolation= cv2.INTER_AREA)
        imgOut = np.reshape(imgOut,frmSize)
        return imgOut
    videoOut = np.array([imgResize(image,frmSize) for image in videoIn])
    return videoOut


def videoProcess(fileName,frmSize,downSample = 2, NormEn = False, RLFlipEn = True, batchMode = True, cropEn = True,numOfRandomCrop = 1):
    vIn = videoRead(fileName,grayMode=frmSize[2] == 1)
    out = videoProcessVin(vIn, frmSize, downSample, NormEn, RLFlipEn, batchMode, cropEn,numOfRandomCrop)
    return out

def videoProcessVin(vIn,frmSize,downSample = 2, NormEn = False, RLFlipEn = True, batchMode = True, cropEn = True,numOfRandomCrop = 1):
    if vIn.shape[0] >= 16:
        vBatch = []
        for i in range(numOfRandomCrop):
            video = videoRezise(vIn,(frmSize[0]+16,frmSize[1]+16,frmSize[2]))
            offset_x = np.random.randint(16) 
            offset_y = np.random.randint(16) 
            video = video[:,offset_x:offset_x+frmSize[0],offset_y:offset_y+frmSize[1]]
            #vSimp = videoSimplify(vRS)
            #video = videoNorm1(video,NormEn)
            video = downSampling(video,downSample)
            if RLFlipEn is True:
                video_flipped = videofliplr(video)
                #vBatch = np.append(vBatch, batchFormat(vDS,cropEn),axis=0)
                #vBatch = np.append(vBatch, batchFormat(vDS_Flipped,cropEn),axis=0)
                vBatch.append(batchFormat(video,cropEn))
                vBatch.append(batchFormat(video_flipped,cropEn))
            else:
                #vBatch = np.append(vBatch, batchFormat(vDS,cropEn), axis=0)
                vBatch.append(batchFormat(video,cropEn))
            
        batchOut = np.reshape(np.array(vBatch),(-1,16)+frmSize)
        del(vBatch)
        
        if batchMode is True:
            return batchOut
        else:
            return vDS
    else:
        return None

def int2OneHot(din,range):
    code = np.zeros(range,dtype=np.float32)
    code[int(din)] = 1
    return code
