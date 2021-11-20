# purpose: top level script to evaluate an existing model on the test data and compute accuracy.

import sys
import Data
import time
import importlib
start = time.time()
#train = imp.load_source('Train', 'Train.py')
train = importlib.machinery.SourceFileLoader('Train', 'Train.py').load_module()
#modelEval = imp.load_source('QLogicE_Eval', 'QLogicE_Eval.py')
modelEval = importlib.machinery.SourceFileLoader('QLogicE_Eval', 'QLogicE_Eval.py').load_module()

###

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = train.getTrainParams()
  modelSaveNamePrefix = 'model1'

  batchSize = 90705

  print('Loading data...')
  sys.stdout.flush()
  dataObj = Data.WnReasonData(dataPath, negSampleSizeRatio, modelSavePath, modelSaveNamePrefix)
  entityCount = dataObj.getEntityCount()
  uConceptCount = dataObj.getUConceptCount()
  bConceptCount = dataObj.getBConceptCount()
  sys.stdout.flush()

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'epochs','r') as f:
    lst = f.readlines()
    nIters = int(lst[0].strip())
  nIters = 1000

  print('Evaluation...')
  print(' ','with model after '+str(nIters)+' epochs of training')

  evalObj = modelEval.ModelEval()
  evalObj.setParam(dataObj, entityCount, uConceptCount, bConceptCount, embedDim, batchSize)
  evalObj.loadModel(modelSavePath, modelSaveNamePrefix, str(nIters))
  evalObj.evalModel()

#end = time.time()
#print("Testing time on UMLS is %s seconds"%(end - start))

