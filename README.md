# Memory Augmented Neural Network for Source Separation
In this project, we implement neural Turing machine (NTM) for sequential signals of speech and noise in presence of different speakers and noise types.
NTM is a memory-augmented neural network which is equipped with external memory to learn long sequential data.
The information is stored with attention mechanism and read-writing scheme. 
For more details about NTM, you can refer to [Neural Turing Machine](https://arxiv.org/pdf/1410.5401.pdf).
The system architecture and experimental settings are shown in Memory Augmented Neural Network for Source Separation.

<img src="Others/NTMCell.png" width="50%">



## Setting
- Hardware:
	- CPU: Intel Core i7-4930K @3.40 GHz
	- RAM: 64 GB DDR3-1600
	- GPU: NVIDIA Tesla K20c 6 GB RAM
- Tensorflow 0.12
- Dataset
	- Wall Street Journal Corpus
	- Noises are collected from [freeSFX](http://www.freesfx.co.uk/soundeffects/) and [AudioMicro](http://www.audiomicro.com/free-sound-effects)

## Result
- An example of demixed signal

|<img src="Others/spectrum_mix.png" width="80%">|
|:--------------------------------------------:|
|Mixed signal|


|<img src="Others/spectrum_clean.png" width="80%">|
|:--------------------------------------------:|
|Clean signal|


|<img src="Others/spectrum_demix.png" width="80%">|
|:--------------------------------------------:|
|Demixed signal|

- STOI measure on other noises

<img src="Others/stoi1.png" width="80%"/>|<img src="Others/stoi2.png" width="80%/">
:----------------------------------------:|:----------------------------------------:
Seen speakers                             |Unseen speakers

- STOI measure on bus noises

<img src="Others/test1_bus.png" width="80%"/>|<img src="Others/test2_bus.png" width="80%/">
:--------------------------------------------:|:--------------------------------------------:
Seen speakers                                 |Unseen speakers

- STOI measure on caf noises

<img src="Others/test1_caf.png" width="80%"/>|<img src="Others/test2_caf.png" width="80%/">
:--------------------------------------------:|:--------------------------------------------:
Seen speakers                                 |Unseen speakers

