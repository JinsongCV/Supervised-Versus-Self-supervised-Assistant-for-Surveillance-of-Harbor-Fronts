# README #

This repo lets you train a basic autoencoder and use it for anormaly detection.

### Overview ###
* experiments.py configure the fundamentals here and use this script to execute train.py, test.py and evaluate.py
* train.py trains a model using.
* test.py produces MSE loss, reconstruction etc. for each sample and saves as npy
* evaluate.py computes metrics, plots and find a decision threshold.
* src/autoencoder.py pytorch lightning style autoencoder model
* src/model.py encoder and decoder definition 
* data/harbour_datamodule.py loades .jpg from folders

### prepare data ###
Place jpg images in subfolders in the data/ directory and make sure to specify the folder path in the cfg dictionary in e.g. experiments.py

### Results for different configurations of training and test data ###

precision = tp / (tp+fp) 
recall = tp / (tp+fn)
F1 = 2 * (precision * recall) / (precision + recall)

% trained on normal, results on test (easy)
threshold: 0.0011759361950680614 (from training set)
tn 409, fp 0, fn 5, tp 74
F1: 0.967, recall: 0.937, precision: 1.000

% trained on normal and abnormal, results on test (easy)
threshold: 0.0006402939325198531 (from training set)
tn 385, fp 24, fn 1, tp 78
F1: 0.862, recall: 0.987, precision: 0.765

% trained on normal and cleaned abnormal, results on test (easy)
threshold: 0.0005544883315451443 (from training set)
tn 372, fp 37, fn 2, tp 77
F1: 0.798, recall: 0.975, precision: 0.675

% trained on normal, results on test (difficult)
threshold: 0.0005973399383947253 (from training set)
tn 388, fp 21, fn 9, tp 82
F1: 0.845, recall: 0.901, precision: 0.796

% trained on normal and abnormal, results on test (difficult)
threshold: 0.0005854243063367903 (from training set)
tn 369, fp 40, fn 9, tp 82
F1: 0.770, recall: 0.901, precision: 0.672

% trained on normal and cleaned abnormal, results on test (difficult)
threshold: 0.0005544883315451443 (from training set)
tn 372, fp 37, fn 10, tp 81
F1: 0.775, recall: 0.890, precision: 0.686
