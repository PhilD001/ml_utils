# Sample machine learning framework for both feature or signal based models

This repository presents a code framework that can be used as a starting point for machine learning projects. The 
code is demonstrated on two sample databases


## Environment 

### Pre-requisites
- environment managed by conda
- install miniconda (or the full anaconda) https://docs.conda.io/en/latest/miniconda.html
- update conda : ``conda update conda``

### Set up (first time)
- install environment using a conda environment file ``conda env create -f ml.yml``
- if you need to update the environment after adding/removing new packages: 
  ``conda env export --from-history > conda env create -f ml.yml``
- Note : to recreate a environemnt from scratch: <br>
``conda create -n ml python=3.9 tensorflow`` \
`` conda install -c anaconda scikit-learn``\
`` conda install -c anaconda pandas``\
``conda install -c anaconda matplotlib``

# Set up (Next times)
1. Activate environment: ``conda activate ml``

# Run (Overview)
1. CD to root of repository
2. Activate environment ``conda activate ml``
3. Run ``main.py`` with appropriate settings (see ``main.py`` for choices)
4. Results are stored in ``results`` folder named ``<data-set>_<model_name>_<time-stamp>``

# Example 1: HAR database via a deep learning model
Here we show how to train, validate, and test a deep learning model on the HAR dataset. We show how to 
run in terminal, but similar steps could be completed in your IDE of choice
1. Open terminal
2. CD to root of repository
3. Activate environment ``conda activate ml_env``
4. train baseline model: ``python main.py --data HAR_sample --epochs 5 --n_filts 4 --n_layers 1``, 
   model should achieve train/validation accuracy ~ 55-80%
5. train an overfit model: ``python main.py --data HAR_sample --epochs 20 --n_filts 64 --n_layers 10``, 
   model should achieve train/validation accuracy of 95 and 75%, respectively
6. Train a "good" model via regularization and dropout (see Chollet, Deep learning with Python sec 4.4)
   ``python main.py --data HAR_sample --epochs 500 --n_filts 64 --n_layers 10 --regularizer L1 --dropout --dropout_amt 0.2 --early_stop``, 
   model achives approximately 94% for the train and validation

Note: For more details about the HAR database, see:
https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/

# Example 2: HAR database via a feature-based machine learning model
Here we show how to train, validate, and test a feature-based machine learning model on the HAR dataset. We show how to 
run in terminal, but similar steps could be completed in your IDE of choice. Note that the HAR dataset contains 
precomputed features, however, here, we will develop our own. 
1. Open terminal
2. CD to root of repository
3. Activate environment ``conda activate ml_env``
4. train baseline model:

## Advanced options
- tuning

