import numpy as np
from chainer import cuda
cuda.check_cuda_available()
xp = cuda.cupy 

class dgtOnGPU:
    def __init__(self,window,shiftLen,fftLen):
        self.window = cuda.to_gpu(window)
        self.idgtwin = xp.hstack([self.window,xp.zeros(int(fftLen-len(window)))])
        self.winLen = cuda.to_gpu(np.array(len(window)))
        self.iwinLen = cuda.to_gpu(np.array(len(self.idgtwin)))
        self.shiftLen = cuda.to_gpu(np.array(shiftLen))
        self.lapLen = self.winLen-self.shiftLen
        self.fftLen = cuda.to_gpu(np.array(fftLen))
        self.idgtZerosWidth = xp.ceil(self.fftLen/shiftLen).astype(xp.int32)
        self.winIdx = xp.arange(len(window)).reshape(int(self.winLen),1)
        self.ifftIdx = xp.arange(int(fftLen)).reshape(int(fftLen),1)
#        a = cuda.to_cpu(self.winMat)

        
    def dgt(self,signal):
        sigLen = len(signal)
        idx = self.dgtIdxGenerator(sigLen)
        _,T = idx.shape
        winMat = xp.repeat(self.window.reshape(int(self.winLen),1),T,axis=1)
        s = winMat*signal[idx]
        S = xp.fft.rfft(s,n=int(self.fftLen),axis=0)
        return S
        
        
        
    def idgt(self,S):
        O,T = S.shape
        sigLen =T*self.shiftLen
        winMat = xp.repeat(self.idgtwin.reshape(int(self.iwinLen),1),T,axis=1)
        tmpSig = xp.zeros( (int(sigLen+self.lapLen),int(self.idgtZerosWidth)) )
        tmp = xp.fft.irfft(S,axis = 0)*winMat
        idxList = self.idgtIdxGenerator(T)
        for ii in range(int(self.idgtZerosWidth)):
            tmpSig1 = tmp.T[ii::self.idgtZerosWidth].flatten()
            tmpSig[idxList[ii][:len(tmpSig1)],ii] = tmpSig1
        tmpSig = xp.sum(tmpSig, axis=1)
        signal = tmpSig[:sigLen]
        signal[:int(self.winLen-self.shiftLen)] += tmpSig[sigLen:]        
        return signal
        
        
    def dgtIdxGenerator(self,sigLen):
        idxShift = xp.arange(0,sigLen,int(self.shiftLen))
        sigIdx = xp.mod(self.winIdx + idxShift, sigLen)
        return sigIdx.astype(xp.int32)
    
    def idgtIdxGenerator(self,T):
        idxList = []
        for ii in range(int(self.idgtZerosWidth)):
            idxShift = self.idgtZerosWidth*self.shiftLen*xp.arange(int(T/self.idgtZerosWidth))
            +self.shiftLen*ii
            idxTmp = (self.ifftIdx+idxShift+self.shiftLen*ii).T.flatten()
            idxList.append(idxTmp)
        return idxList
    