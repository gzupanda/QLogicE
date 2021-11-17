# purpose: top level script to evaluate an existing model on the test data and compute accuracy.

import os, sys, math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import time
import importlib
start = time.time()
train = importlib.machinery.SourceFileLoader('reasonE.train', 'Train.py').load_module()
modelEval = importlib.machinery.SourceFileLoader('model.eval', 'QLogicE_Eval.py').load_module()


###

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = train.getTrainParams()
  modelSaveNamePrefix = 'model1'

  batchSize = 359680

  print('Loading data...')
  sys.stdout.flush()
  dataObj = data.WnReasonData(dataPath, negSampleSizeRatio, modelSavePath, modelSaveNamePrefix)
  entityCount = dataObj.getEntityCount()
  uConceptCount = dataObj.getUConceptCount()
  bConceptCount = dataObj.getBConceptCount()
  sys.stdout.flush()

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'nIters','r') as f:
    lst = f.readlines()
    nIters = int(lst[0].strip())
  nIters = 1000

  print('Evaluation...')
  print(' ','with model after '+str(nIters)+' iters of training')

  evalObj = modelEval.ModelEval()
  evalObj.setParam(dataObj, entityCount, uConceptCount, bConceptCount, embedDim, batchSize)
  evalObj.loadModel(modelSavePath, modelSaveNamePrefix, str(nIters))
  evalObj.evalModel()

end = time.time()
print("Running time on YAGO3-10 is %s second"%(end - start))


