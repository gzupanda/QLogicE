# purpose: top level script to train a new model for the specified set of parameters.

import os, sys, math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import time
import importlib
start = time.time()
modelTrainer = importlib.machinery.SourceFileLoader('model.trainer', 'model.trainer.py').load_module()


def getTrainParams():
  return dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio


###

dataPath = 'data/'
modelSavePath = 'model/'
modelSaveNamePrefix = 'model0'
embedDim = 4
lossMargin = 2
negSampleSizeRatio = 1

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = getTrainParams()

  learningRate = 0.1
  nIters = 1000
  batchSize = 70721


  print('Training...')
  print(' ','fresh training')
  print(' ',str(nIters)+' iters to do now')

  dataObj = data.WnReasonData(dataPath, negSampleSizeRatio)
  sys.stdout.flush()

  trainer = modelTrainer.ModelTrainer(dataObj, dataObj.getEntityCount(), dataObj.getUConceptCount(), dataObj.getBConceptCount(), embedDim)
  logF = open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'log', 'w')
  logF.write('Train: nIters='+str(nIters)+'\n')
  trainer.init(logF)
  trainer.trainIters(batchSize, learningRate, nIters, lossMargin, logF)
  sys.stdout.flush()
  logF.close()

  trainer.saveModel(modelSavePath, modelSaveNamePrefix, str(nIters))
  dataObj.saveEntityConceptMaps(modelSavePath, modelSaveNamePrefix)

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'nIters', 'w') as f:
    f.write("%s\n" % str(nIters))
    f.close()

end = time.time()
print("Training time on WN18 is %s second"%(end - start))
