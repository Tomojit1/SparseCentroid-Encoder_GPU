Instruction to use the code package.

This a GPU implementation of Sparse Centroid-Encoder. PyTorch's inbuilt Adaptive Moment Estimation (Adam)
is used to update network parameters. The script 'FeatureSelectingSCEPyTorch.py' uses single center
per class, the code for multiple centers will be publised later. After the feature
selection an ANN is used to classifiy the trst samples. The ANN code is written in PyTorch.
A high dimensional biological dataset ALLAML is provided with the package.
Please uncompress the Data.zip. I used Mac's 'Compress Data' tool. For
the ALLAML data I've provided the three data partitions due to space constraint.
For any question contact Tomojit Ghosh (tomojit.ghosh@colostate.edu).


Requirements:
1. Python: 3.7.4
2. PyTorch: 1.2.0
3. Numpy: 1.17.2
4. ipython: 7.8.0
5. sklearn: 1.0.2


Notes: 
1. The ALLAML dataset is given with the package.

2. Seperate script for each data sets:
		ALLAML: testSCEPyTorch_ALLAML.py

How to run the code:
1. Download the code.

2. Unzip the package. The code was compressed in MacOS BigSur Version 11.6.2.

3. Make sure all the requirements are satisfied.

4. To run the script from ipython use the commands:
    a. ipython
	b. run testSCEPyTorch_ALLAML.py ==>>for ALLAMML data

5. To run the script directly from python:
    a. python testSCEPyTorch_ALLAML.py ==>>for ALLAMML data
