# equilibriated RNN for speech enhancement
In this repository, real-time speech enhancement method using the equilibriated recurrent neural network (ERNN) for the T-F mask estimator \[1] is impremented using Chainer.

Our paper can be found [here]() (in preparation).
If you use codes in this repository, please cite the above paper.

In our paper, VoiceBank-DEMAND dataset \[2] (available [here](http://dx.doi.org/10.7488/ds/1356)) is used.



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
\[1] D. Takeuchi, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada, “Real-time speech enhancement using equilibriated RNN,” in 2020 IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2020. (accepted)

\[2] C. Valentini-Botinho, X. Wang, S. Takaki, and J. Yamagishi, “Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech.,” in 9th ISCA Speech Synth. Workshop, 2016, pp. 146–152.
