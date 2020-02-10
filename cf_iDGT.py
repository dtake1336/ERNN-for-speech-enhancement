import sys
import numpy as np
from chainer.backends import cuda
from chainer import Function
cuda.check_cuda_available()
xp = cuda.cupy 

class inverseDGT(Function):
    def forward_cpu(self, inputs):
        print("use cpu")
        sys.exit()
        
    def forward_gpu(self, inputs):
        Xr,Xi,window,shiftLen,fftLen = inputs
        S = Xr+Xi*1j
        O,T = S.shape
        sigLen =T*int(shiftLen)
        winLen = len(window)
        idgtwin = xp.hstack([window,xp.zeros(int(fftLen-len(window)))])
        winMat = xp.repeat(idgtwin.reshape(int(len(idgtwin)),1),T,axis=1)
        idgtZerosWidth = xp.ceil(fftLen/shiftLen).astype(xp.int32)
        tmpSig = xp.zeros( (int(sigLen+winLen-shiftLen), int(idgtZerosWidth)))
        tmp = xp.fft.irfft(S,axis = 0)*winMat
        idxList = self.idgtIdxGenerator(T,idgtwin,idgtZerosWidth,shiftLen)
        
        for ii in range(int(idgtZerosWidth)):
            tmpSig1 = tmp.T[ii::idgtZerosWidth].flatten()
            tmpSig[idxList[ii][:len(tmpSig1)],ii] = tmpSig1 
        tmpSig = xp.sum(tmpSig, axis=1)
        signal = tmpSig[:sigLen]
        signal[:int(winLen-shiftLen)] += tmpSig[sigLen:]        
        
        return signal.astype(np.float32),


    def backward_cpu(self, inputs):
        print("use cpu")
        sys.exit()

    def backward_gpu(self, inputs, grad_outputs):
        Xr,Xi,window,shiftLen,fftLen = inputs
        gf = grad_outputs[0]
        sigLen = len(gf)
        idx = self.dgtIdxGenerator(sigLen,window,int(shiftLen))
        _,T = idx.shape
        winMat = xp.repeat(window.reshape(int(len(window)),1),T,axis=1)
        s = winMat*gf[idx]
        S = xp.fft.rfft(s,n=int(fftLen),axis=0)
        Sr = xp.real(S).astype(np.float32)
        Si = xp.imag(S).astype(np.float32)          
        return Sr,Si
    
    def dgtIdxGenerator(self,sigLen,window,shiftLen):
        winIdx = xp.arange(len(window)).reshape(len(window),1)
        idxShift = xp.arange(0,sigLen,int(shiftLen))
        sigIdx = xp.mod(winIdx + idxShift, sigLen)
        return sigIdx.astype(xp.int32)
    
    def idgtIdxGenerator(self,T,window,idgtZerosWidth,shiftLen):
        idxList = []
        winIdx = xp.arange(len(window)).reshape(len(window),1)
        for ii in range(int(idgtZerosWidth)):
            idxShift = idgtZerosWidth*shiftLen*xp.arange(int(T/idgtZerosWidth))
            +shiftLen*ii
            idxTmp = (winIdx+idxShift+shiftLen*ii).T.flatten()
            idxList.append(idxTmp)
        return idxList
    
def iDGT(Xr,Xi,window,shiftLen,fftLen):
    f = inverseDGT()(Xr,Xi,window,shiftLen,fftLen)
    return f