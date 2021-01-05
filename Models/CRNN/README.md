# Convolutional Recurrent Neural Network (CRNN) Model
  We constructed a CRNN model for this task. The model processes raw data using CNNs, and then feed its output to RNNs, forming a Convolutional Recurrent Neural Network (CRNN). In such case, convolutional layers extract local features, and recurrent layers combine it to extract temporal features. We take the original data which contains 2500 time step and 12 leads (<batch_size>, 1, 2500, 12) as the input of the model.
  
  ![alt text](/CRNN/CRNN_model_architecture.png)
