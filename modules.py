#import modules
import numpy as np
import scipy as scipy
from scipy.io import wavfile
from chainer import cuda
import glob # ファイル一覧取得
import sys
import math
cuda.check_cuda_available()
xp = cuda.cupy 


#####################################################################
# transfer to GPU 
def list_to_gpu(ll):
    rr = []
    for ii in range(len(ll)):
        rr.append( cuda.to_gpu( ll[ii] ) )
    return rr

#####################################################################
# read & write audio file
def wavread(fn):
    fs, data = wavfile.read(fn)
    data     = (data.astype(np.float32) / 2**(15))
    return data, fs

def wavwrite(fn, data, fs):
    data = scipy.array(scipy.around(data * 2**(15)), dtype = "int32")
    wavfile.write(fn, fs, data)

def loadDataset(clean_dir, noisy_dir, speech_per_set, testFlag):
    c_files = glob.glob(clean_dir + "/" + "*.wav")
    n_files = glob.glob(noisy_dir + "/" + "*.wav")
    Num_wav   = len( c_files )
    if testFlag == 1:
        Num_wav = round(Num_wav*0.01)
    perm      = np.random.permutation( Num_wav )

    S_all = []
    S_set = []
    cnt   = 0
    print( 'Loading... (Training set)' )
    for ii in range( Num_wav ):
        if( ii%int(Num_wav/20) == 0 ):
            sys.stdout.write( '\r   '+str(ii+1)+'/'+str( Num_wav ) )
            sys.stdout.flush()
        c_fn = c_files[perm[ii]]
        n_fn = n_files[perm[ii]]
        s, org_fs = wavread( c_fn )
        x, org_fs = wavread( n_fn ) 
        S_set.append([s,x])
        cnt += 1
        if(cnt == speech_per_set):
            cnt = 0
            S_all.append( S_set )
            S_set = []
    sys.stdout.write('\n')
    S_all.append( S_set )
    return S_all

########################################################################################
# DGT config
def hannWin(N):
    window = 0.5 + 0.5*np.cos(2*np.pi*( np.arange(-N/2,N/2)+np.remainder(N,2)/2 ) / N);
    return window

def calcCanonicalDualWindow(window,shiftLen):
    winLen = len(window)
    padLen = np.ceil(winLen/shiftLen)*shiftLen - winLen;
    dualWin = np.concatenate(( window, np.zeros(int(padLen)).astype(np.float64) ), axis=0)
    dualWin = np.reshape(dualWin,(int(len(dualWin)/shiftLen),int(shiftLen)))
    dualWin /= np.sum( np.abs(dualWin)**2, axis=0)
    return dualWin[:winLen].flatten()

def zeroPadForDGT(signal,shiftLen,fftLen):
    sigLen = len(signal);
    c = np.array(int(shiftLen) * int(fftLen) / math.gcd(int(shiftLen), int(fftLen))).astype(np.float64);
    sigLong = xp.hstack( (signal, xp.zeros( int(np.ceil(sigLen/c)*c-sigLen)) ) )    
    return sigLong

########################################################################################
# DNN savename 
def DNNfn( dnn_dir, DNNmode, in_dim, winLen, shiftLen, fftLen, winMode, lossMode, epoch, datetimeStr ): 
    return dnn_dir+DNNmode+'_' \
        +'stft'\
        +str(in_dim)+'ch'\
        +'_winL'+str(winLen)\
        +'_shift'+str(shiftLen)\
        +'_fftL'+str(fftLen)\
        +'_'+winMode\
        +'_'+lossMode\
        +'_epoch'+str(epoch).zfill(3)\
        +'_'+datetimeStr\
        +".h5"

#####################################################################
if __name__ == "__main__":
     print('Debug only')   
    