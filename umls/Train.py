# purpose: top level script to train a new model for the specified set of parameters.

import os, sys, math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import Data
import time
import importlib
start = time.time()
#modelTrainer = imp.load_source('model.trainer', 'QLogicE_Trainer.py')
modelTrainer = importlib.machinery.SourceFileLoader('QLogicE_Trainer', 'QLogicE_Trainer.py').load_module()


def getTrainParams():
  return dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio


###

dataPath = 'data/'
modelSavePath = 'model/'
modelSaveNamePrefix = 'model5'
embedDim = 4
lossMargin = 2
negSampleSizeRatio = 1

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = getTrainParams()

  learningRate = 0.2
  nIters = 1000
  batchSize = 2500

  print('Start to train...')
  print('It will run', str(nIters)+' epochs.')

  dataObj = Data.WnReasonData(dataPath, negSampleSizeRatio)
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
#print("Training time on UMLS is %s seconds"%(end - start))
