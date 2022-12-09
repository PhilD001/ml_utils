# Sample machine learning framework for both feature or signal based models

This repository presents a code framework that can be used as a starting point for machine learning projects. The 
code is demonstrated on two sample databases


## Environment 

### Pre-requisites
- environment managed by conda
- install anaconda/miniconda https://docs.conda.io/en/latest/miniconda.html
- update conda : ``conda update conda``

### Set up (first time)
- install environment using environment.yml ``conda env create -f sample_ml_env.yml``
- if you need to update the environment after adding/removing new packages: 
  ``conda env export --from-history > conda env create -f sample_ml_env.yml.yml``

# Set up (Next times)
1. Activate environment: ``conda activate sample_ml``
2. Choose which dataset you would like to explore: HAR or Keras

# Run (Overview)
1. CD to root of repository
2. Activate environment ``conda activate sample_ml_env``
3. Run ``main.py`` with appropriate settings (see ``main.py`` for choices)
4. Results are stored in ``results`` folder named ``<data-set>_<model_name>_<time-stamp>``

# Example 1: HAR database via a deep learning model
Here we show how to train, validate, and test a deep learning model on the HAR dataset. We show how to 
run in terminal, but similar steps could be completed in your IDE of choice
1. Open terminal
2. CD to root of repository
3. Activate environment ``conda activate sample_ml_env``
4. We start by deepython main.py --
# Results

### Signal-based (Deep learning) model 
- A sample CNN model can be tested a sample data set available from keras:
`` main.py --data keras_sample --model_name cnn_sample --epochs 200  --batch_size 32`` accuracy 

### Feature-based model
- A sample feature-based model with features computed from the tsfresh package:
`` main.py ---data keras_sample --model_name decision_tree --feature_set comprehensive --feature_selection`` accuracy 100%


