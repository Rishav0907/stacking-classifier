# Heart Disease Prediction using Stacking Classifier

This project implements a stacking classifier to predict heart disease using various machine learning models. The stacking classifier combines the predictions of logistic regression, support vector machine (SVM), and a meta-classifier (neural network) to make final predictions.

## Project Structure

- `main.py`: The main script that orchestrates the entire process.
- `Logistic_regression_model.py`: Implementation of the logistic regression model.
- `svm.py`: Implementation of the support vector machine model.
- `meta_classifier.py`: Implementation of the meta-classifier using a neural network.
- `decision_tree.py`: Implementation of a decision tree (not used in the final stacking classifier).
- `config.py`: Configuration settings for various models.

## Dependencies

- numpy
- pandas
- scikit-learn
- torch
- cvxpy
- ucimlrepo

## Dataset

The project uses the Heart Disease dataset from the UCI Machine Learning Repository.

## Models

### Logistic Regression

A custom implementation of logistic regression is used.

### Support Vector Machine (SVM)

A custom implementation of SVM using CVXPY for optimization.

### Meta-Classifier

A neural network implemented using PyTorch serves as the meta-classifier.

## Stacking Process

1. The data is split into training and testing sets.
2. Logistic Regression and SVM models are trained on the training data.
3. Predictions from these models on both training and testing data are used to create a new dataset.
4. The meta-classifier is trained on this new dataset using the original training labels.
5. Final predictions are made using the trained meta-classifier.

## Usage

Run the `main.py` script to execute the entire process:

`python main.py`


The script will output the accuracy scores for Logistic Regression, SVM, and the Meta-Classifier.

## Configuration

Model parameters can be adjusted in the `config.py` file.
