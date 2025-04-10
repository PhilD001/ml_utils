# Sample machine learning framework

Sample code repository for machine learning. 
User can test framework using two sample datasets (HAR and Ford) 

## Environment 

### Pre-requisites
- environment managed by conda
- install miniconda (or the full anaconda) https://docs.conda.io/en/latest/miniconda.html
- update your conda : ``conda update conda``

### Create (first time)

User should follow these steps: <br>
1. open terminal
2. cd to sample_framework directory
3. `conda create --name sample_ml python=3.9`
4. `conda activate sample_ml`
5. `conda install -c conda-forge tensorflow=2.11.1`
6. `conda install --y --file conda_requirements.txt`
7. `pip install -r pip_requirements.txt`

Note: 
- For installing an environment on a GPU based machine, replace step 5 with : 
``conda install -c anaconda tensorflow-gpu=2.6.0``

  
### Activate (next times)
1. Activate environment: ``conda activate sample_ml``

# Run (Overview)
1. CD to ``samle_framework``
2. Activate environment ``conda activate sample_ml``
3. Run ``main.py`` with appropriate settings (see ``main.py`` for choices)
4. Results are stored in ``results`` folder


# Experiments
Below we demonstrate some example models that have been tested. It may still be possible to obtain better results

## Feature-based models

### Support vector classifier on HAR database (untuned, no feature selection)
python main.py --database HAR_sample --model_name svc --evaluate_on_test_set --cross_val_num 3
* test set accuracy =  84.62 (3.29)%

### Random forest classifier on HAR database (body_acc channels only, tuned, feature selection)
main.py --database HAR_sample --model_name random_forest --evaluate_on_test_set --tune --cross_val_num 3
* test set accuracy = 95.8%

### Signal-based models 

### Baseline CNN (untuned)
python main.py --database HAR_sample --model_name cnn --evaluate_on_test_set --channel_names body_acc_x body_acc_y body_acc_z --epochs 100 --early_stop --cross_val          

### Baseline CNN (tuned)
# todo: test this
python main.py --database keras_sample --model_name cnn --epochs 500 --n_layers 3 --tune --early_stop

