# purpose: model definition class - to define model and forward step.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ReasonEModel(nn.Module):
  def __init__(self, entityCount, uConceptCount, bConceptCount, embedDim):
    super(ReasonEModel, self).__init__()
    self.embedDim = embedDim
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount  # 也是关系的数量

    self.baseMat = Variable(torch.FloatTensor(torch.eye(embedDim)))
    self.entityEmbed = nn.Embedding(entityCount, embedDim)
    self.bConceptHEmbed = nn.Embedding(bConceptCount, embedDim)
    self.bConceptTEmbed = nn.Embedding(bConceptCount, embedDim)
    #TransE的三个量的表示
    self.headEmbed = nn.Embedding(entityCount, embedDim)
    self.tailEmbed = nn.Embedding(entityCount, embedDim)
    self.relationEmbed = nn.Embedding(bConceptCount, embedDim)

    nn.init.xavier_uniform_(self.entityEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptHEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptTEmbed.weight)
    #TransE三个量的初始化
    nn.init.xavier_uniform_(self.headEmbed.weight)
    nn.init.xavier_uniform_(self.tailEmbed.weight)
    nn.init.xavier_uniform_(self.relationEmbed.weight)

    self.entityEmbed.weight.data = F.normalize(self.entityEmbed.weight.data, p=2, dim=1)
    self.bConceptHEmbed.weight.data = F.normalize(self.bConceptHEmbed.weight.data, p=2, dim=1)
    self.bConceptTEmbed.weight.data = F.normalize(self.bConceptTEmbed.weight.data, p=2, dim=1)
    #获取初始化数据
    self.headEmbed.weight.data = F.normalize(self.headEmbed.weight.data, p=2, dim=1)
    self.tailEmbed.weight.data = F.normalize(self.tailEmbed.weight.data, p=2, dim=1)
    self.relationEmbed.weight.data = F.normalize(self.relationEmbed.weight.data, p=2, dim=1)


  def forward(self, aBHE, aBTE, aBC, aHead, aTail, aRelation, nABHE, nABTE, nABC, nHead, nTail, nRelation, uniqE, uniqBC, lossMargin, device):
    aBHEE = self.entityEmbed(aBHE)
    aBTEE = self.entityEmbed(aBTE)
    aBCHE = self.bConceptHEmbed(aBC)
    aBCTE = self.bConceptTEmbed(aBC)
    #TransE量的正例
    aHeadE = self.headEmbed(aHead)
    aTailE = self.tailEmbed(aTail)
    aRelationE = self.relationEmbed(aRelation)

    nABHEE = self.entityEmbed(nABHE)
    nABTEE = self.entityEmbed(nABTE)
    nABCHE = self.bConceptHEmbed(nABC)
    nABCTE = self.bConceptTEmbed(nABC)
    #TransE的负例
    nHeadE = self.headEmbed(nHead)
    nTailE = self.tailEmbed(nTail)
    nRelationE = self.relationEmbed(nRelation)

    uniqEE = self.entityEmbed(uniqE)
    uniqBCHE = self.bConceptHEmbed(uniqBC)
    uniqBCTE = self.bConceptTEmbed(uniqBC)

    zero = Variable(torch.FloatTensor([0.0]))
    zero = zero.to(device)
    one = Variable(torch.FloatTensor([1.0]))
    one = one.to(device)
    halfDim = Variable(torch.FloatTensor([self.embedDim/2.0]))
    halfDim = halfDim.to(device)
    margin = Variable(torch.FloatTensor([lossMargin]))
    margin = margin.to(device)

    tmpBE2CH = (one-aBCHE)*aBHEE
    tmpBE2CT = (one-aBCTE)*aBTEE
    tmpTransE = torch.sum(torch.abs(aHeadE + aRelationE - aTailE), dim = 1)
    bE2CMemberL = torch.sum(tmpBE2CH*tmpBE2CH, dim =1) + torch.sum(tmpBE2CT*tmpBE2CT, dim=1) + tmpTransE

    tmpNBE2CH = (one-nABCHE)*nABHEE
    tmpNBE2CT = (one-nABCTE)*nABTEE
    tmpNTransE = torch.sum(torch.abs(nHeadE + nRelationE - nTailE), dim = 1)
    tmpNBL = torch.sum(tmpNBE2CH* tmpNBE2CH, dim=1) + torch.sum(tmpNBE2CT*tmpNBE2CT, dim=1) + tmpNTransE
    bE2CDiscMemberL = torch.max(margin - tmpNBL, zero)

    tmpE = torch.sum(uniqEE*uniqEE, dim=1) - one
    uniqENormL = tmpE*tmpE

    tmpBCH = uniqBCHE*(one-uniqBCHE)
    tmpBCT = uniqBCTE*(one-uniqBCTE)
    uniqBCBasisAlignL = torch.sum(tmpBCH*tmpBCH, dim=1) + torch.sum(tmpBCT*tmpBCT, dim=1)

    tmpBCHDim = torch.sum(torch.abs(uniqBCHE), dim=1)
    tmpBCHL = torch.max(one-tmpBCHDim, zero)
    tmpBCTDim = torch.sum(torch.abs(uniqBCTE), dim=1)
    tmpBCTL = torch.max(one-tmpBCTDim, zero)
    uniqBCBasisCountL = tmpBCHL + tmpBCTL

    # TransE的计算
    atmpTransE = torch.sum(torch.abs(aHeadE + aRelationE - aTailE), dim = 1)
    ntmpTransE = torch.sum(torch.abs(nHeadE + nRelationE - nTailE), dim = 1)
    tmpTransE = atmpTransE - ntmpTransE
    TransEL = torch.max(margin + tmpTransE, zero)

    
    return bE2CMemberL, bE2CDiscMemberL, uniqENormL, uniqBCBasisAlignL, uniqBCBasisCountL, TransEL
    
  def getEntityEmbedding(self, e):
    return self.entityEmbed(e)
# get relation embedding
  def getRelationEmbedding(self, r):
    return self.relationEmbed(r)

  def getBConceptHEmbedding(self, c):
    return self.bConceptHEmbed(c)

  def getBConceptTEmbedding(self, c):
    return self.bConceptTEmbed(c)

  def getBaseMat(self):
    return self.baseMat


