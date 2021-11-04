# The model QLogicE

This repository is the implement codes for the model QLogicE. This model joints the translation embedding \emph{TransE} and the quantum logic \emph{E2R} for the sake of capture more features to improve the expressiveness of the knowledge graph completion. 
## Install 

Firstly, clone the code from the github repository by following command:
```
git clone https://github.com/gzupanda/QLogicE.git
```
The code mainly depends on PyTorch1.1.0 or later verions and Python 3. In this code, there are seven directories corresponding to each dataset. In other words, every directory is an independent one with the model codes. To better understand how to quick start to run the experiments, we take the dataset UMLS corresponding one for example.
## Run the experiments
There are seven directories in this repository. They are devided as three groups. Fortunately, the running is simple. For every dataset, we just run the following steps and take the dataset umls for example:
### 1. enter the directory
In every directory, 
```
cd QLogicE
cd umls
```
### 2. training
```
python reasonE.train.py
```
### 3. testing
```
python reasonE.test.py
```
The trained model save in the directory model.

## The triples data
In the datasets fb15k237, wn18rr and yago3-10, there are a text file name triples. This file including the all triples in training set, validing set and testing set. This is because our model is under the assumption of close world, the entity and relation out of knowledge graph is incapable of link predicting.

## License

This software comes under a non-commercial use license, please see the LICENSE file.
