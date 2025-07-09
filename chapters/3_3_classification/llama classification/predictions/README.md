# LLAMA-Classification

This folder contains the code and files that were used to classify articles using the Llama3.1-8b-Instruct model together with inclusion and exclusion criteria that were manually extracted from the underlying review papers.

## Hardware
The code was developed on the [bwVisu platform](https://www.urz.uni-heidelberg.de/en/research-and-teaching/research-related-projects/bwvisu) and run on the [bwForClusterHelix](https://www.urz.uni-heidelberg.de/en/service-catalogue/high-performance-computing/bwforcluster-helix) supercomputer hosted at Heidelberg university. 

Helix uses the [Slurm workload manager](https://www.urz.uni-heidelberg.de/en/service-catalogue/high-performance-computing/bwforcluster-helix). The entry point for the code was therefore the ``classify.sh``-file, which upon start calls the ``classify.py`` script which contains the actual logic for inference. 

``classify.py`` only inferes one dataset at a time and expects the name of the dataset as a command line argument when being called.

The script loads the given dataset from the ``./datasets`` and subsequently eligibility criteria that were menually extracted from the originial review articles from the ``criteria`` directory.

Next, the script initialized the Llama model weights from local storage. Therefore download the weights using the huggingface command line interface and the command `` huggingface-cli download meta-llama/Llama-3.1-8B-Instruct``.

The script then initializes the model and processes the articles in batches of a given size, individually classifying each article on its own. 

Predictions are stored in directories named  after the dataset and sub-directories refering to a specific run. There are two types of outputs:
- ``[dataset]_pred.csv`` uses a structured format that features the model's decision, its reasoning in natural language and metadata split across several columns. This structure is provided as the model is instructed to output in a certain format
- ``[dataset]_pred_raw.csv`` contains the same data without splitting it into several columns. This file was created in case the model output should cause errors in the automatic splitting.

