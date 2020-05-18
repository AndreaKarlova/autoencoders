## name of the project  v0.1
## Main features: 
e.g.
1. Forecast balance for any selected time period into the future
2. Change between multiple different implementations of forecasting methods 

### Structure of repository:
Root Project directory contains the following the folders:
e.g.
 - __config__: contains configuration files in `yaml` or `json` format

 - __data__: contains datasets used for testing the implementation, place datasets used for model development in this folder 
 
 - __reccurent_transactions__: the python package repository

 - __logs__: all logs related to project are automatically stored in this folder

 - __models__: contains trained models, use for storing trained models

 - __results__: all results such as metrics, images, etc. are stored here 
 
 - __tests__: all the tests that can be run using the pytest library 



### 1. Set up
Make sure you have miniconda and python 3.6.5.
Create virtual environment: 

`conda create --name reccurent_transactions python=3.6`

`source activate reccurent_transactions`

#### 1.1 Use install script
1. check the execute rights for executing the 'install.sh' script, 
change the rights if needed:
 `sudo chmod 755 install.sh`
2. run installation script:
 `./install.sh`


### 2. Test:
- test the installation by running `python example.py`
- there is a series of implementation tests available in `reccurent_transactions/tests`
- these can all be run by `python -m pytest tests`


### 3. Configuration Files:
1. `config/logging.yaml`: configuration of the logger
2. `config/config.json`: main configuration file that can be used for forecasting model
3. `config/models/`: config file with model architecture for the forecasting algorithms, each model architecture has separate config file 


### 4.  `reccurent_transactions` package:
3. `recurrent_transactions/metrics`: module containing useful functions for evaluating the performance of predictions
4. `recurrent_transactions/models`:  module containing different model architectures
5. `recurrent_transactions/dataset_utils`: module for dataset manipulation 
6. `recurrent_transactions/labelling`: preprocessing utils 
8. `recurrent_transactions/tests`: 

### 5.  `labelling` config:
