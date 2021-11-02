# The model QLogicE
This repository contains the code of the main experiments presented in the papers:

## Install 

First clone the repository:
```
git clone https://github.com/gzupanda/QLogicE.git
```

The code depends on PyTorch1.1.0 or later verions and Python 3.

## Run the experiments
There are seven directories in this repository. They are devided as three groups. Fortunately, the running is simple. For every dataset, we just run the following steps and take the dataset umls for example:
### 1. enter the directory
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
