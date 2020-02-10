##################################################################################
# clear workspace      
def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
clearall()

##################################################################################
# import module  
import numpy as np
from matplotlib import pylab as plt

import modules as mm
import dnnModel
import cf_iDGT as udF
import dgtForGPU

import chainer
from chainer import cuda, serializers
import chainer.functions as F
import os
import sys
import time
import glob

##################################################################################
# exp. param
red = 2

winMode = 'dual'

#DNNmode = 'LSTM2'
#DNNmode = 'BLSTM2'

#DNNmode = 'ERNN_K1'
#DNNmode = 'ERNN_K3'
DNNmode = 'ERNN_K5'

lossMode = 'timeDomMAE'

stateSize = 512
#stateSize = 256

hiddenSize = 512
#hiddenSize = 256
#hiddenSize = 128
#hiddenSize = 64
#hiddenSize = 32

################################################################################

# parameter setting
# stft setting
winLen = 512
shiftLen = int(winLen/red)
fftLen = winLen # ch Num
fftDecimate = 1
chNum = int(winLen/fftDecimate/2+1)

Log_reg = 10**(-6)

condTxt = 'DNN:'+DNNmode\
    +' STFTchNum:'+str(chNum)\
    +' winLen:'+str(winLen)\
    +' shift:'+str(shiftLen)\
    +' Nfft:'+str(fftLen)\
    +' winMode:'+winMode\
    +' loss:'+lossMode\


print(condTxt)

dnnName = DNNmode+'_s'+str(stateSize)+'h'+str(hiddenSize)

##################################################################################
# parameter setting

# GPU setting
DEVICE_NUM  = 0 # M40
cuda.check_cuda_available()
xp          = cuda.cupy 
DEVICE_INFO = cuda.get_device_from_id( DEVICE_NUM )

# training data directory
ctestDir  = 'D:/sound_data/Voicebank_DEMAND/clean_testset_wav2'
ntestDir  = 'D:/sound_data/Voicebank_DEMAND/noisy_testset_wav2'

testDir = 'D:/sound_data/test_ERNN_ver2'
if(os.path.isdir(testDir)==False):
            os.mkdir(testDir)
            
            
tmp = dnnName\
    +str(chNum)\
    +'_winLen'+str(winLen)\
    +'_shift'+str(shiftLen)\
    +'_'+winMode\
    +'_'+lossMode\
            
print(tmp)

condDir = testDir+'/'+tmp
if(os.path.isdir(condDir)==False):
    os.mkdir(condDir)
            
        
# save dnn directory
dnn_dir  = './dnn_dir/'

###################################################################################
# DGT setting
window = mm.hannWin(winLen)
windowD = mm.calcCanonicalDualWindow(window,shiftLen)[:len(window)]

    
fb_chNum = int(winLen/2)+1

##################################################################################
#load testData
sdataFns  = glob.glob(ctestDir + "/*.wav")
xdataFns  = glob.glob(ntestDir + "/*.wav")
testNum = len(sdataFns)

##################################################################################
#load DNN 
with cuda.Device( DEVICE_INFO ):
    dgt = dgtForGPU.dgtOnGPU(window,shiftLen,fftLen)
    start = time.time()
    windowG = cuda.to_gpu(window)
    windowDG = cuda.to_gpu(windowD)
    shiftLenG = cuda.to_gpu(np.array(shiftLen))
    fftLenG = cuda.to_gpu(np.array(fftLen))

    ##################################################################################
    # DNN setting
    init_dr = 0.0
    epoch = 200
    # DNN setting
    
    inputSize = chNum
    
    exec('dnnEst = dnnModel.'+DNNmode+'(inputSize, hiddenSize, stateSize).to_gpu( DEVICE_INFO )')
    print("params: "+str(sum(p.data.size for p in dnnEst.params())))
        
    loadDNNfns = mm.DNNfn( dnn_dir, dnnName, chNum, winLen, shiftLen, fftLen, winMode, lossMode, epoch, '' )
    loadDNNfns = loadDNNfns[:-4]
    DNNfiles = glob.glob(loadDNNfns+'*')
    loadDNNfn = DNNfiles[0]
    
    serializers.load_hdf5(loadDNNfn, dnnEst)
    
    with chainer.no_backprop_mode(): 
        with chainer.using_config('train', False):
            for n in range(len(sdataFns)):
                sys.stdout.write('\rtestSet: '+str(n+1)+'/'+str(len(sdataFns))) 
                sys.stdout.flush()
                s = cuda.to_gpu(mm.wavread(sdataFns[n])[0])
                x = cuda.to_gpu(mm.wavread(xdataFns[n])[0])
                s = mm.zeroPadForDGT(s,shiftLen,fftLen).astype(np.float32) 
                x = mm.zeroPadForDGT(x,shiftLen,fftLen).astype(np.float32)
                X = dgt.dgt(x)
                Xr = xp.real(X).astype(np.float32)
                Xi = xp.imag(X).astype(np.float32)
                
                X = xp.expand_dims(X, axis=0)
                fet = xp.log(xp.abs(X).astype(np.float32) + Log_reg )
                G = F.squeeze(dnnEst(fet))
                
                GXr = Xr*G
                GXi = Xi*G
                gx  = udF.iDGT(GXr,GXi,windowDG,shiftLenG,fftLenG).data
                
                tmpgx = cuda.to_cpu(gx)
                tmpgx = tmpgx/np.max(np.abs(tmpgx))
                
                saveFn = condDir+'/'+sdataFns[n][len(ctestDir)+1:]
                mm.wavwrite(saveFn, tmpgx, 16000)
    
