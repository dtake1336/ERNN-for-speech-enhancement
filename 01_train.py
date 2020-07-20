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
import cf_iDGT as iDGTcf
import dgtForGPU as dgt

from chainer import cuda,optimizers, serializers
import chainer.functions as F
import os
import sys
import time
from datetime import datetime
import math


##################################################################################
# exp. param
red = 2

winMode = 'dual'

#DNNmode = 'LSTM2'
#DNNmode = 'BLSTM2'

DNNmode = 'ERNN_K1'
#DNNmode = 'ERNN_K3'
#DNNmode = 'ERNN_K5'

lossMode = 'timeDomMAE'

stateSize = 512
#stateSize = 256

hiddenSize = 512
#hiddenSize = 256
#hiddenSize = 128
#hiddenSize = 64
#hiddenSize = 32

# flag
testFlag = 0 # 0:full train mode 1:test mode (few files) 
##################################################################################
# parameter setting
# stft setting
winLen = 512
shiftLen = 256
fftLen = winLen # ch Num
fftDecimate = 1
chNum = int(winLen/fftDecimate/2+1)

# GPU setting
DEVICE_NUM  = 0
cuda.check_cuda_available()
xp          = cuda.cupy 
DEVICE_INFO = cuda.get_device_from_id( DEVICE_NUM )

# training data directory
cleanDir  = '/sound_data/Voicebank_DEMAND/clean_trainset_wav'
noisyDir  = '/sound_data/Voicebank_DEMAND/noisy_trainset_wav'

# save dnn directory
dnn_dir  = './dnn_dir/'
if(os.path.isdir(dnn_dir)==False):
    os.mkdir(dnn_dir)
  
# train parameter
loadSize = 2048
batchSize = 16 
Log_reg = 10**(-6)
batchTimeFrame = 68

MAX_EPOCH         = 200
# Adam setting
lr           = 0.0001

# loading train data   
S_all = mm.loadDataset(cleanDir, noisyDir, loadSize, testFlag)

##################################################################################
# condition 
condTxt = 'DNN:'+DNNmode\
+' STFTchNum:'+str(chNum)\
+' winLen:'+str(winLen)\
+' shift:'+str(shiftLen)\
+' Nfft:'+str(fftLen)\
+' winMode:'+winMode\
+' loss:'+lossMode\
+' stateSize:'+str(stateSize)\
+' hiddenSize:'+str(hiddenSize)

###################################################################################
# DGT setting
with cuda.Device( DEVICE_INFO ):
    window = mm.hannWin(winLen)
    windowD = mm.calcCanonicalDualWindow(window,shiftLen)
        
    dgt = dgt.dgtOnGPU(window,shiftLen,fftLen)
    print(condTxt)
    
##################################################################################
# DNN setting

    inputSize = chNum

    exec('dnnEst = dnnModel.'+DNNmode+'(inputSize, hiddenSize, stateSize).to_gpu( DEVICE_INFO )')
    print("params: "+str(sum(p.data.size for p in dnnEst.params())))
    # Optimizer
    optm_dnn = optimizers.Adam(alpha=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    optm_dnn.setup(dnnEst)

##################################################################################
# start train 
    print("train start")  

    start = time.time()
    windowG = cuda.to_gpu(window)
    windowDG = cuda.to_gpu(windowD)
    shiftLenG = cuda.to_gpu(np.array(shiftLen))
    fftLenG = cuda.to_gpu(np.array(fftLen))
    olddatetime = 'none'
    
    def calcLoss(G, X, S):
        GXr = G*xp.real(X).astype(np.float32)
        GXi = G*xp.imag(X).astype(np.float32)
        Sr = xp.real(S).astype(np.float32)
        Si = xp.imag(S).astype(np.float32)
        gxL = [F.expand_dims(iDGTcf.iDGT(GXr[ii],GXi[ii],windowDG,shiftLenG,fftLenG), axis=0) for ii in range(len(G))]
        sL = [F.expand_dims(iDGTcf.iDGT(Sr[ii],Si[ii],windowDG,shiftLenG,fftLenG), axis=0) for ii in range(len(G))]
        gx = F.vstack(gxL)
        s = F.vstack(sL)
        loss = F.mean_absolute_error(gx,s)
        return loss

    # train iteration
    for epoch in range(1, MAX_EPOCH+1):
        loss      = 0.0
        total_cnt = 0
        loss_cnt  = 0
        sum_loss  = 0.0
        perm1     = np.random.permutation( len(S_all) )
        settime = time.time()
        
        print('\n'+condTxt)
        for ii in range( len(S_all) ):
            
            
            S_set = mm.list_to_gpu( S_all[ perm1[ii] ] )
            perm2 = np.random.permutation( len(S_set) )
            batchNum = math.floor(len(S_set)/batchSize)
            
            
            for jj in range( batchNum ):
                sys.stdout.write('\repoch: '+str(epoch)+' TrnSet: '+str(jj+1)+'/'+str(batchNum) )
                sys.stdout.flush()
                
#                 s = xp.reshape(cuda.to_gpu(np.array([])).astype(np.float32), (0,batchTimeFrame*shiftLen) )
#                 x = xp.reshape(cuda.to_gpu(np.array([])).astype(np.float32), (0,batchTimeFrame*shiftLen) )
#                 S = xp.reshape(cuda.to_gpu(np.array([])).astype(np.float32), (0,chNum,batchTimeFrame) )
#                 X = xp.reshape(cuda.to_gpu(np.array([])).astype(np.float32), (0,chNum,batchTimeFrame) )
                
                s = xp.zeros((batchSize, batchTimeFrame*shiftLen)).astype(np.float32)
                x = xp.zeros((batchSize, batchTimeFrame*shiftLen)).astype(np.float32)
                S = xp.zeros((batchSize, chNum, batchTimeFrame)).astype(np.float32)
                X = xp.zeros((batchSize, chNum, batchTimeFrame)).astype(np.float32)
                
                for kk in range( batchSize ):
                    sx = S_set[ perm2[kk+jj*batchSize] ]
                    st = np.random.randint(len(sx[0])-batchTimeFrame*shiftLen)
                    end = st+batchTimeFrame*shiftLen
                    stmp = mm.zeroPadForDGT(sx[0][st:end],shiftLen,fftLen).astype(np.float32) 
                    xtmp = mm.zeroPadForDGT(sx[1][st:end],shiftLen,fftLen).astype(np.float32)
                    Stmp = dgt.dgt(stmp)
                    Xtmp = dgt.dgt(xtmp)
                    
#                     stmp =  xp.reshape(stmp, (1, batchTimeFrame*shiftLen) )
#                     xtmp =  xp.reshape(xtmp, (1, batchTimeFrame*shiftLen) )
#                     Stmp =  xp.reshape(Stmp, (1, chNum, batchTimeFrame) )
#                     Xtmp =  xp.reshape(Xtmp, (1, chNum, batchTimeFrame) )
                    
#                     s = xp.concatenate( (s,stmp), axis=0)
#                     x = xp.concatenate( (x,xtmp), axis=0)
#                     S = xp.concatenate( (S,Stmp), axis=0)
#                     X = xp.concatenate( (X,Xtmp), axis=0)
                    
                    s[kk,:len(stmp)] = stmp
                    x[kk,:len(xtmp)] = xtmp
                    S[kk,:,:len(Stmp[0])] = Stmp
                    X[kk,:,:len(Xtmp[0])] = Xtmp
                
                fet = xp.log(xp.abs(X).astype(np.float32) + Log_reg )
                G = dnnEst(fet)
                loss = calcLoss(G, X, S)
                sum_loss  = float(loss.data)
                total_cnt += len(G)
                # backpropagation
                dnnEst.cleargrads()        
                loss.backward()
                loss.unchain_backward()
                optm_dnn.update()
                loss     = 0.0
                loss_cnt = 0
        
            sys.stdout.write('\n')
        ave_loss = sum_loss/total_cnt
        
        print("Average Loss = "+ str( ave_loss ))
        
        
        
        print("Save DNN...")
        newdatetime = datetime.now().strftime('%m%d_%H%M%S')

        dnnName = DNNmode+'_s'+str(stateSize)+'h'+str(hiddenSize)
        saveDNNfn = mm.DNNfn( dnn_dir, dnnName, chNum, winLen, shiftLen, fftLen, winMode, lossMode, epoch, newdatetime )
        serializers.save_hdf5(saveDNNfn, dnnEst)
        process_time = time.time() - start
        print("exeTime(sec):"+str(process_time))
        
        start = time.time()
        
        try:
            delfn = mm.DNNfn( dnn_dir, dnnName, chNum, winLen, shiftLen, fftLen, winMode, lossMode, epoch-1, olddatetime )
            os.remove(delfn)
        except:
            print('no file to delete')
            
        olddatetime = newdatetime
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.flipud(cuda.to_cpu(G[0].data)),aspect='auto')
        plt.title('mask')
        
        plt.subplot(1,2,2)
        plt.imshow(np.flipud(cuda.to_cpu(xp.log10(xp.abs(S[0])+Log_reg))),aspect='auto')
        plt.title('speech')
        plt.show()
