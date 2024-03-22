# Project 1: Regression - Predicting Heart Disease from Personal Lifestyle Factors

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)

This repository contains our work in predicting heart disease in adults based on personal lifestyle metrics and factors. We have employed three primary machine learning algorithms to classify whether an individual is healthy or not. These models include (Ridge) Least Square, Least Square (Stochastic) Gradient Descent, and (Ridge) Logistic Regression. To provide a baseline comparison, a Random Guess algorithm is also included.

**Authors:** : [Luca Carroz](https://people.epfl.ch/emilie.carroz), 
[David Schroeter](https://people.epfl.ch/david.schroeter), 
[Paulo Ribeiro de Carvalho](https://people.epfl.ch/paulo.ribeirodecarvalho)

<hr style="clear:both">

## Description

The repository is structured as follows:
- `run.ipynb`: A Jupyter Notebook that documents the project's journey. Running it from start to finish without any changes will produce a `.csv` file containing our best submission score on AI crowd. Please note that running the notebook in its entirety may take up to half an hour, depending on your computational power.
- `implementations.py`: This file contains the various regression algorithms that need to be tested. It also includes helper functions.

The `src` folder contains hand-built functions organized into separate files for easy navigation:

- `src/`
  - `data_loading.py`: Functions for data loading.
  - `features_engineering.py`: Functions for data engineering and preprocessing, including risk score mapping, under/over-sampling, and more.
  - `hyperparameters_selection.py`: Functions for selecting hyperparameters of specific functions.
  - `model_performance.py`: Functions for computing performance metrics of the model, such as accuracy and F1-score.
  - `model_training.py`: Functions for the final model training using k-fold training.
  - `helper.py`: Useful functions needed across various libraries.
  - `helper_visualization.py`: Functions for plotting graphs.
  - `Exceptions.py`: Custom exceptions to raise in case of invalid function arguments.

## Libraries

The entire project is implemented using basic libraries. While the core computational steps rely on these libraries, we have also imported more advanced libraries for easier graph visualization, such as matplotlib.

List of libraries used:
- `numpy`
- `matplotlib`

## Results

You can find our best submission to AI crowd [here](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/teams/DLP).
