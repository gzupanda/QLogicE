# purpose: top level script to retrain an existing model for the specified set of parameters.

import sys
import Data
import importlib

train = importlib.machinery.SourceFileLoader('Train', 'Train.py').load_module()
modelTrainer = importlib.machinery.SourceFileLoader('QLogicE_Trainer', 'QLogicE_Trainer.py').load_module()


###

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = train.getTrainParams()

  learningRate = 0.00001
  nIters = 3000
  batchSize = 141442

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'epochs','r') as f:
    lst = f.readlines()
    oldNIters = int(lst[0].strip())
    totalNIters = oldNIters + nIters

  print('Retraining...')
  print(' ',str(oldNIters)+' epochs so far')
  print(' ',str(nIters)+' epochs to do now')

  dataObj = Data.WnReasonData(dataPath, negSampleSizeRatio, modelSavePath, modelSaveNamePrefix)
  sys.stdout.flush()

  trainer = modelTrainer.ModelTrainer(dataObj, dataObj.getEntityCount(), dataObj.getUConceptCount(), dataObj.getBConceptCount(), embedDim)
  logF = open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'log', 'a')
  logF.write('Retrain: epochs='+str(nIters)+'\n')
  trainer.init(logF, True, modelSavePath, modelSaveNamePrefix, str(oldNIters))
  trainer.trainIters(batchSize, learningRate, nIters, lossMargin, logF)
  sys.stdout.flush()
  logF.close()

  trainer.saveModel(modelSavePath, modelSaveNamePrefix, str(totalNIters))

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'epochs', 'w') as f:
    f.write("%s\n" % str(totalNIters))
    f.close()


