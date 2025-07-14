# Automated Title/Abstract-Screening
This study is, to our knowledge, the first direct comparison between supervised machine learning and large language models in title/abstract-screening.

Title/abstract-screening is a laborious process step of systematic literature reviews, which summarize the results of individual studies on the same topic. Traditionally, title/abstract-screening requires human reviewers to review the titles and abstracts of potentially thousands of articles derived from searches in scientific databases to include only those eligible by pre-defined criteria.

The study consists of a preprocessing and an analysis of the datasets, the classification into inclusion or exclusion by both supervised machine learning and large language models, and the calculation of performance metrics and final evaluation of the performance.

Most of the code is structured in ipython notebooks within the ``chapters`` directory, which can be run one after another. Besides the code, the ``data`` directory houses versions of the datasets after each of the study's steps.

## Installation

1. Install [Git](https://git-scm.com/downloads) and ``git clone`` this repository to your computer
2. Install [Miniforge](https://github.com/conda-forge/miniforge) or another conda or mamba package manager
3. With a conda or mamba comptabile command line interface, ``cd`` into the root directory of this project 
4. Run ``mamba create -f environment.yml`` to install an environment ``title_abstract_screening`` that installs python along with all required packages
5. Make sure to use the newly created environment when running code from this project.

## Structure

### Chapters

The ``chapters`` directory contains the code which was used within this study. The files are numbered and have been run in ascending order.
- ``dataset preparation`` contains the code to retrieve the titles and abstracts of the articles, add metadata such as text languages, perform general preprocessing of the texts, and analyze the datasets
- ``classification`` contains the code which was used to classify the datasets by both supervised machine learning and large language models.
    - Classification by supervised machine learning consists of the ``preprocess`` notebook which contains code with specific preprocessing required for the supervised machine learning models and the ``classify`` notebook which contains code for the actual classification.
    - The code for the classification using the Llama-3.1-8B-Instruct model is contained within ``llama classification``
        > Note that the llama classification requires additional packages such as ``pytorch`` and ``transformers`` which have been left out from the environment to reduce its size.
        >
        > You can always install additional packages by ``mamba activate title_abstract_screening`` followed by ``mamba install [package_name]``
- ``results`` contains two files:
    - Within ``calculcation`` we calculate performance metrics with confidence-intervals by evaluating and averaging 1000 bootstrap samples of each model's prediction on every dataset. These metrics are then saved, together with classification reports, the bootstrap samples, and individual scores for the bootstrap samples within a python dictionary which we save as ``results.pkl``
    - Finally ``evaluation`` uses the data within ``results.pkl`` to plot and interpret the performances of the different models to create the plots and tables used within out manuscript.

### Data
- The ``data`` directory  houses two sub-directories:
    - ``datasets`` contains copies of the datasets after each step of the preprocessing
    - ``predictions`` contains lists of predictions by the supervised machine learning models and Llama-3.1-8B-Instruct.

### Src
- ``src`` contains functions that get re-used throughout the project.