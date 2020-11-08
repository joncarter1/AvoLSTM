# AvoLSTM
Time series prediction on the Kaggle Avocado dataset using an LSTM.

Uses interal neural network package "ReTorch" which re-implements some of the internals from PyTorch:
- Adam and SGD optimisers, including hyper-gradient variants
- Fully-connected, Conv1D and MaxPool1D neural network layers

Final network architecture:
- 100 dimensional LSTM hidden state
- 2 layer output network ($h_t$ -> $y_t$) with 100 hidden neurons and swish/linear activations
- Hyper-gradient Adam optimiser :  <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha=10^{-3},&space;\beta_1=0.9,&space;\beta_2=0.99,&space;\alpha_{LR}=10^{-8}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha=10^{-3},&space;\beta_1=0.9,&space;\beta_2=0.99,&space;\alpha_{LR}=10^{-8}" title="\alpha=10^{-3}, \beta_1=0.9, \beta_2=0.99, \alpha_{LR}=10^{-8}" /></a>
Work done as part of the AIMS CDT course.

aims.robots.ox.ac.uk
