# purpose: model trainer class - to train a new model or load and retrain an existing model on the training data for a specific number of iterations and store the resultant model.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import model


class ModelTrainer:
  def __init__(self, dataObj, entityCount, uConceptCount, bConceptCount, embedDim):
    self.dataObj = dataObj
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount
    self.embedDim = embedDim

  def init(self, logF, retrainFlag=False, modelPath=None, modelNamePrefix=None, modelNamePostfix=None):
    if retrainFlag==False:
      self.model = model.ReasonEModel(self.entityCount, self.uConceptCount, self.bConceptCount, self.embedDim)
    else:
      self.model = self.loadModel(modelPath, modelNamePrefix, modelNamePostfix)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', self.device)
    sys.stdout.flush()

    self.model = self.model.to(self.device)

  def trainIters(self, batchSize, learningRate, nIters, lossMargin, logF):
    print('Training iters...')
    sys.stdout.flush()

    modelOpt = torch.optim.Adam(self.model.parameters(), lr = learningRate)

    for it in range(0, nIters):
      self.dataObj.updateRandomNegAboxTripleList()
      self.dataObj.updateRandomTrainIndexList()
      bE2CMemberAccLoss = 0
      bE2CDiscMemberAccLoss = 0
      uniqENormAccLoss = 0
      uniqBCBasisAlignAccLoss = 0
      uniqBCBasisCountAccLoss = 0
      TransELoss =0
      accLoss = 0
      accCount = 0
      for tI in range(self.dataObj.getTrainDataLen(batchSize)):
        modelOpt.zero_grad()
        aBHET, aBTET, aBCT, aHeadET, aTailET, aRelationET,nABHET, nABTET, nABCT, nHeadET, nTailET, nRelationET, uniqET, uniqBCT = self.dataObj.getTrainDataTensor(tI, batchSize)
  
        aBHET = aBHET.to(self.device)
        aBTET = aBTET.to(self.device)
        aBCT = aBCT.to(self.device)
        # TransE_start
        aHeadET = aHeadET.to(self.device)
        aTailET = aTailET.to(self.device)
        aRelationET = aRelationET.to(self.device)
        # TransE_end
        nABHET = nABHET.to(self.device)
        nABTET = nABTET.to(self.device)
        nABCT = nABCT.to(self.device)
        #TransE_start
        nHeadET = nHeadET.to(self.device)
        nTailET = nTailET.to(self.device)
        nRelationET = nRelationET.to(self.device)
        #TransE_end
        uniqET = uniqET.to(self.device)
        uniqBCT = uniqBCT.to(self.device)

        bE2CMemberL, bE2CDiscMemberL, uniqENormL, uniqBCBasisAlignL, uniqBCBasisCountL, TransEL = self.model(aBHET, aBTET, aBCT,  aHeadET, aTailET, aRelationET, nABHET, nABTET, nABCT, nHeadET, nTailET, nRelationET, uniqET, uniqBCT, lossMargin, self.device)

        bE2CMemberLoss = torch.sum(bE2CMemberL)/len(aBHET)
        bE2CDiscMemberLoss = torch.sum(bE2CDiscMemberL)/len(nABHET)
        uniqENormLoss = torch.sum(uniqENormL)/len(uniqET)
        uniqBCBasisAlignLoss = torch.sum(uniqBCBasisAlignL)/len(uniqBCT)
        uniqBCBasisCountLoss = torch.sum(uniqBCBasisCountL)/len(uniqBCT)
        #TransE loss function
        TransELoss = torch.sum(TransEL)/(0.5 * (len(aHeadET) + len(aTailET)))

        qlLoss = bE2CMemberLoss + bE2CDiscMemberLoss + uniqENormLoss + uniqBCBasisAlignLoss + uniqBCBasisCountLoss
        # TransE
        tLoss = 2 * TransELoss
        #loss = 0.5 * (qlLoss + tLoss)
        loss = qlLoss + tLoss

        loss.backward()
        modelOpt.step()

        bE2CMemberAccLoss += bE2CMemberLoss.item()
        bE2CDiscMemberAccLoss += bE2CDiscMemberLoss.item()
        uniqENormAccLoss += uniqENormLoss.item()
        uniqBCBasisAlignAccLoss += uniqBCBasisAlignLoss.item()
        uniqBCBasisCountAccLoss += uniqBCBasisCountLoss.item()
        TransELoss += TransELoss.item()
        accLoss += loss.item()
        accCount += 1
        c = accCount
        print('iter='+str(it)+' :', 'overall loss='+'{:.4f}'.format(accLoss/c)+',', 'bE2CMember='+'{:.4f}'.format(bE2CMemberAccLoss/c)+',', 'bE2CDiscMember='+'{:.4f}'.format(bE2CDiscMemberAccLoss/c)+',', 'uniqENorm='+'{:.4f}'.format(uniqENormAccLoss/c)+',', 'uniqBCBasisAlign='+'{:.4f}'.format(uniqBCBasisAlignAccLoss/c)+',', 'uniqBCBasisCount='+'{:.4f}'.format(uniqBCBasisCountAccLoss/c),'TransELoss='+'{:.4f}'.format(TransELoss/c))

      accLoss /= accCount
      bE2CMemberAccLoss /= accCount
      bE2CDiscMemberAccLoss /= accCount
      uniqENormAccLoss /= accCount
      uniqBCBasisAlignAccLoss /= accCount
      uniqBCBasisCountAccLoss /= accCount
      TransELoss /= accCount
 
  def loadModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    model = torch.load(modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Loaded model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    return model

  def saveModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    torch.save(self.model, modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Saved model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)


