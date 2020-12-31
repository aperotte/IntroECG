# autoECG
Model and transfer learning model based on Automatic diagnosis of the 12-lead ECG using a deep neural network, Ribeiro, et al.
# 1. Background
  We tried to build an advanced model according to the paper Automatic diagnosis of the 12-lead ECG using a deep neural network. This paper uses CNN and RNN models to do
the classification. The dataset in the original paper is very similar to what we have. The original data, acquired from the Telehealth Network of Minas Gerais (TNMG) [1]
, also contains 12 leads. With each lead, the duration of the ECG recording is between 7s to 10s, with frequency ranging from 300 HZ to 600 HZ. To make input of the same size, the data was zero-padded and each lead has 4096 numbers. The dataset size is very large, containing 2,322,513 ECG records from 1,676,384 patients. The training set contains 98% of the data and the validation set contains 2% of the data.
In the paper, the DNN architecture is structured as follows:  

![alt text](/modelstructure.jpg)   


  The network consists of a convolutional layer followed by four residual blocks with two convolutional layers per block. Max Pooling and 1*1 convolutional network are included in the skip connections to make the dimensions match those output from the main branch. The output of the last layer was then fed into a fully connected dense layer with sigmoid
activation function. The result in this paper is attractive. The minimum F1 score achieved is 0.8970 (for 1dAVb) and the maximum F1 score is 1.00 (for LBBB). For all types of disease, the DNN prediction accuracy outperforms humans.  

![alt text](/modelresult.jpg)  


# 2. Modified Structure
  The original model consists of four residual blocks and the overall structure is shown in the following graph. The output has six entries, each containing a probability between 0 and 1, and can be understood as the probability of a given abnormality to be present. The abnormalities are not mutually exclusive, so the probabilities do not necessarily sum to one.  
  While in our cases, we only need to determine a certain ECG data whether is normal or abnormal (the two probabilities sum to one), so our new output should be a single probability. Therefore, we modified the final dense layer by changing the unit number from six to one while still using the ‘sigmoid’ activation function. Moreover, we used the binary cross entropy function as the new loss function.  
  The coding file of the modified model is in the autoECG-GPU.ipynb file under the root directory.

  
  
  
  
# 3. Data Generator
Each of our ECG data consists of 12 leads, with a duration of 10 seconds and a frequency of 250 Hz. Therefore, each sample is a data matrix of 12 by 2500. In our case, we worked on 2 sets
of data, one was synthetic, one was real clinical data. Although we cannot share our real clinical data, the simulation code is accessible. 
 ![3D trajectory](/3D.png)
 ![table](/table.png)
 We can see from the above table that each interval in the 3D trajectory can be fixed by 3 parameters: the starting position $\theta_i$ along the circle of each interval in radius, The amplitude of each spike $a_i$ and the width of each wave $b_i$. By altering these 3 parameters we can change the shape of the 3D trajectory and thus change the waveform of the resulting ECG. 
For the real parameters which we use in the simulation code is in the following table:
 
|                |Parameter                           |Meaning                          |
|----------------|-------------------------------|-----------------------------|
|1|ti            |$\theta_i$            |
|2          |ai            |$a_i$          |
|3          |bi|$b_i$|
  |4          |hr|heart rate|
  |5          |duration|default 10s|
  |6         |gamma|a (12,5) matrix to modify the five waves' amplitudes $a_i$ of 12 leads|
 |7         |normal_N|the number of normal samples|

Next step:
The input shape of the autoECG model should be 4096 by 12. To satisfy this requirement, the data need to be firstly padded with zeroes: from (12, 2500) to (12, 4096);  after being transposed, the data become (4096, 12). 


# 4. Transfer Learning
  Because the performance of the model was pretty good, and the training set used by the paper was super large, we assumed the original model learned enough information about how to diagnose an ECG, which suggested using transfer learning and fine-tuning would be a promising and practical step to take. We basically froze the first 5 layers making them untrainable and freed the rest layers’s weights (the pretrained model in total had 50 layers) and we built our own neural network followed by the 45th layer of the pretrained one. We added three 1d convolutional layers and three dense layers followed by the 45th layer. The structure can be viewed as follow:  
 
![alt text](/transferlearning.png)    
  
  The coding file of Transfer Learning is in the Transfer.ipynb file under the root directory.

[1] Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead
ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4
