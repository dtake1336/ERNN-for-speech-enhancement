# equilibriated RNN for speech enhancement
In this repository, real-time speech enhancement method using the equilibriated recurrent neural network (ERNN) for the T-F mask estimator is impremented using Chainer.
Our paper can be found [here]() (in preparation).
In our paper, VoiceBank-DEMAND dataset (available [here](http://dx.doi.org/10.7488/ds/1356)) is used\[1].

### Reference
D. Takeuchi, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada, “Real-time speech enhancement using equilibriated RNN,” in 2020 IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2020.

### Dependencies
We have tested these codes on follwoing environment:
* Python 3.6.4
* Chainer 6.2.0
* NumPy 1.17.2
* CuPy 6.2.0
* CUDA Runtime Version 10100
* cuDNN Version 7500


### Usage example
A set of Python codes for training and test are available.
<dl>
<dd> Run "01_train.py" to train a model </dd> 
<dd> Run "02_test.py" to evaluate a model </dd> 
</dl>
Note that paths in each code need to be changed for your environment.

### Reference
D. Takeuchi, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada, “Real-time speech enhancement using equilibriated RNN,” in 2020 IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2020.
