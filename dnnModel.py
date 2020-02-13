import numpy as np
import chainer
from chainer import cuda, Variable, variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import copy
cuda.check_cuda_available()
xp = cuda.cupy 

###############################################################################
class LSTM2(Chain):
    def __init__(self, inputSize, hiddenSize, stateSize):
        super(LSTM2, self).__init__()
        with self.init_scope():
            self.add_layer('l1', L.NStepLSTM(n_layers=2, in_size=inputSize, out_size=hiddenSize, dropout=0))
            self.add_layer('l2', L.Linear(hiddenSize, inputSize)) 
    ###################################################################
    # add layer
    def add_layer(self, name, function):
        super(LSTM2, self).add_link(name, function) 
    # change dropout ratio
    def changeDr(self, newDr):
        self.l1.dropout = newDr
    # forward 
    def __call__(self, x0):
        x1 = F.reshape(x0, x0.shape)
        x2 = F.transpose(x1, (0,2,1))
        L1 = [x2[ii] for ii in range(len(x2))]  
        hy, cy, L2 = self.l1( hx=None, cx=None, xs=L1 )
        L3 = [self.l2(L2[ii]) for ii in range(len(L2))]
        L4 = [F.expand_dims(L3[ii], axis=0) for ii in range(len(L3))]
        x3 = F.sigmoid(F.vstack(L4))
        return F.transpose(x3, (0,2,1))
    
###############################################################################
class BLSTM2(Chain):
    def __init__(self, inputSize, hiddenSize, stateSize):
        super(BLSTM2, self).__init__()
        with self.init_scope():
            self.add_layer('l1', L.NStepBiLSTM(n_layers=2, in_size=inputSize, out_size=hiddenSize, dropout=0))
            self.add_layer('l2', L.Linear(2*hiddenSize, inputSize)) 
    ###################################################################
    # add layer
    def add_layer(self, name, function):
        super(BLSTM2, self).add_link(name, function) 
    # change dropout ratio
    def changeDr(self, newDr):
        self.l1.dropout = newDr
    # forward 
    def __call__(self, x0):
        x1 = F.reshape(x0, x0.shape)
        x2 = F.transpose(x1, (0,2,1))
        L1 = [x2[ii] for ii in range(len(x2))]  
        hy, cy, L2 = self.l1( hx=None, cx=None, xs=L1 )
        L3 = [self.l2(L2[ii]) for ii in range(len(L2))]
        L4 = [F.expand_dims(L3[ii], axis=0) for ii in range(len(L3))]
        x3 = F.sigmoid(F.vstack(L4))
        return F.transpose(x3, (0,2,1))
     

################################################################################
class ERNN_K1(Chain):
    def __init__(self, inputSize, hiddenSize, stateSize):
        super(ERNN_K1, self).__init__()
        with self.init_scope():
            self.stateSize = stateSize
            self.l1_in = L.Linear(inputSize, stateSize)
            self.l1_hid = L.Linear(stateSize, stateSize)
            self.l2 = L.Linear(stateSize, hiddenSize)
            self.l3 = L.Linear(hiddenSize, stateSize)
            self.l_out = L.Linear(stateSize, inputSize)
            
            self.initEta = 1e-2
            self.eta1 = variable.Parameter(self.initEta, 1)
            
             
    def __call__(self, x0):
        Bsize,Fsize,Tsize = x0.shape
        hp = Variable( xp.zeros((Bsize,self.stateSize)).astype(np.float32) )
        y = xp.zeros((Bsize,Fsize,0)).astype(np.float32)
        for ii in range(Tsize):
            h0 = Variable( xp.zeros((Bsize,self.stateSize)).astype(np.float32) )
            wx = self.l1_in(x0[:,:,ii])
            
            #K=1
            uh0 = self.l1_hid(hp+h0)
            F1h_hat = F.relu(uh0 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h1 = h0 + self.eta1 * (Fnh_hat - (h0+h0))
            
            y1 = F.sigmoid(self.l_out(h1))
            y1 = F.expand_dims(y1, axis=2)
            y = F.concat( (y,y1), axis=2)
            hp = copy.deepcopy(h1)
        return y
    
################################################################################
class ERNN_K3(Chain):
    def __init__(self, inputSize, hiddenSize, stateSize):
        super(ERNN_K3, self).__init__()
        with self.init_scope():
            self.stateSize = stateSize
            self.l1_in = L.Linear(inputSize, stateSize)
            self.l1_hid = L.Linear(stateSize, stateSize)
            self.l2 = L.Linear(stateSize, hiddenSize)
            self.l3 = L.Linear(hiddenSize, stateSize)
            self.l_out = L.Linear(stateSize, inputSize)
            
            
            self.initEta = 1e-2
            self.eta1 = variable.Parameter(self.initEta, 1)
            self.eta2 = variable.Parameter(self.initEta, 1)
            self.eta3 = variable.Parameter(self.initEta, 1)
            
             
    def __call__(self, x0):
        Bsize,Fsize,Tsize = x0.shape
        hp = Variable( xp.zeros((Bsize,self.stateSize)).astype(np.float32) )
        y = xp.zeros((Bsize,Fsize,0)).astype(np.float32)
        for ii in range(Tsize):
            h0 = Variable( xp.zeros((Bsize,self.stateSize)).astype(np.float32) )
            wx = self.l1_in(x0[:,:,ii])
            
            #K=1
            uh0 = self.l1_hid(h0+hp)
            F1h_hat = F.relu(uh0 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h1 = h0 + self.eta1 * (Fnh_hat - (h0+h0))
            
            #K=2
            uh1 = self.l1_hid(h1+hp)
            F1h_hat = F.relu(uh1 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h2 = h1 + self.eta2 * (Fnh_hat - (h1+h0))
            
            #K=3
            uh2 = self.l1_hid(h2+hp)
            F1h_hat = F.relu(uh2 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h3 = h2 + self.eta3 * (Fnh_hat - (h2+h0))
            
            y1 = F.sigmoid(self.l_out(h3))
            y1 = F.expand_dims(y1, axis=2)
            y = F.concat( (y,y1), axis=2)
            hp = copy.deepcopy(h3)
        return y
################################################################################
class ERNN_K5(Chain):
    def __init__(self, inputSize, hiddenSize, stateSize):
        super(ERNN_K5, self).__init__()
        with self.init_scope():
            self.stateSize = stateSize
            self.l1_in = L.Linear(inputSize, stateSize)
            self.l1_hid = L.Linear(stateSize, stateSize)
            self.l2 = L.Linear(stateSize, hiddenSize)
            self.l3 = L.Linear(hiddenSize, stateSize)
            self.l_out = L.Linear(stateSize, inputSize)
            
            
            self.initEta = 1e-2
            self.eta1 = variable.Parameter(self.initEta, 1)
            self.eta2 = variable.Parameter(self.initEta, 1)
            self.eta3 = variable.Parameter(self.initEta, 1)
            self.eta4 = variable.Parameter(self.initEta, 1)
            self.eta5 = variable.Parameter(self.initEta, 1)
            
             
    def __call__(self, x0):
        Bsize,Fsize,Tsize = x0.shape
        hp = Variable( xp.zeros((Bsize,self.stateSize)).astype(np.float32) )
        y = xp.zeros((Bsize,Fsize,0)).astype(np.float32)
        for ii in range(Tsize):
            h0 = Variable( xp.zeros((Bsize,self.stateSize)).astype(np.float32) )
            wx = self.l1_in(x0[:,:,ii])
            
            #K=1
            uh0 = self.l1_hid(h0+hp)
            F1h_hat = F.relu(uh0 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h1 = h0 + self.eta1 * (Fnh_hat - (h0+h0))
            
            #K=2
            uh1 = self.l1_hid(h1+hp)
            F1h_hat = F.relu(uh1 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h2 = h1 + self.eta2 * (Fnh_hat - (h1+h0))
            
            #K=3
            uh2 = self.l1_hid(h2+hp)
            F1h_hat = F.relu(uh2 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h3 = h2 + self.eta3 * (Fnh_hat - (h2+h0))
            
            #K=4
            uh3 = self.l1_hid(h3+hp)
            F1h_hat = F.relu(uh3 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h4 = h3 + self.eta4 * (Fnh_hat - (h3+h0))
            
            #K=5
            uh4 = self.l1_hid(h4+hp)
            F1h_hat = F.relu(uh4 + wx)
            Fnh_hat = F.relu(self.l3(F.relu(self.l2(F1h_hat)))) # kokoni DNN wo ippai irerareru
            h5 = h4 + self.eta5 * (Fnh_hat - (h4+h0))
            
            y1 = F.sigmoid(self.l_out(h5))
            y1 = F.expand_dims(y1, axis=2)
            y = F.concat( (y,y1), axis=2)
            hp = copy.deepcopy(h3)
        return y   
    

    






