# purpose: model evaluation class - to load and evaluate an existing model on the test data and compute accuracy.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import Accuracy


class ModelEval:
  def setParam(self, dataObj, entityCount, uConceptCount, bConceptCount, embedDim, batchSize):
    self.dataObj = dataObj
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount
    self.embedDim = embedDim
    self.batchSize = batchSize

  def setModel(self, model, device):
    self.model = model
    self.device = device

  def loadModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    self.model = torch.load(modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Loaded model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', self.device)
    sys.stdout.flush()
    self.model = self.model.to(self.device)

  def evalModel(self):
    self.one = Variable(torch.FloatTensor([1.0]))
    self.one = self.one.to(self.device)
    self.accObj = Accuracy.Accuracy()
    self.computeEmbeddingQuality()

  def getUClassSpaceMembershipScore(self, uCE, eLst):
    uE = [e for e in eLst]
    uE = Variable(torch.LongTensor(uE))
    uE = uE.to(self.device)
    uEE = self.model.getEntityEmbedding(uE)
    uCE = uCE.repeat(len(eLst), 1)
    tmp = (self.one-uCE)*uEE
    s = torch.sum(tmp*tmp, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getBClassSpaceMembershipScore(self, bCE, eLst):
    bHE = []
    bTE = []
    HeadE = []
    TailE = []
    RelationE = []
    for e in eLst:
      h, t = e
      bHE.append(h)
      bTE.append(t)
      HeadE.append(h)
      TailE.append(t)
      #RelationE.append(h)
    bHE = Variable(torch.LongTensor(bHE))
    bTE = Variable(torch.LongTensor(bTE))
    HeadE = Variable(torch.LongTensor(HeadE))
    TailE = Variable(torch.LongTensor(TailE))
    RelationE = Variable(torch.LongTensor(RelationE))
    bHE = bHE.to(self.device)
    bTE = bTE.to(self.device)
    HeadE = HeadE.to(self.device)
    TailE = TailE.to(self.device)
    RelationE = RelationE.to(self.device)
    bHEE = self.model.getEntityEmbedding(bHE)
    bHTE = self.model.getEntityEmbedding(bTE)
    HeadEE = self.model.getEntityEmbedding(HeadE)
    TailEE = self.model.getEntityEmbedding(TailE)
    RelationEE = self.model.getRelationEmbedding(RelationE)
    bCHE, bCTE = bCE
    bCHE = bCHE.repeat(len(eLst), 1)
    bCTE = bCTE.repeat(len(eLst), 1)

    tmpH = (self.one-bCHE)*bHEE
    tmpT = (self.one-bCTE)*bHTE
    #tmpTransE = torch.sum(torch.abs(HeadEE + RelationEE -TailEE),dim =1)
    s = torch.sum(tmpH*tmpH, dim=1) + torch.sum(tmpT*tmpT, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getClassSpaceMembershipScore(self, cE, rLst, eLst):
    uHE = []
    bHE = []
    bTE = []
    HeadE = []
    TailE = []
    RelationE = []
    uCount = 0
    bCount = 0
    rCount = 0
    eLstMap = []
    rLstMap = []
    for e in eLst:
      h, t = e
      if t==None:
       uHE.append(h)
       eLstMap.append((0,uCount))
       uCount+=1
      else:
       bHE.append(h)
       bTE.append(t)
       HeadE.append(h)
       TailE.append(t)
       eLstMap.append((1,bCount))
       bCount+=1
    for rC in rLst:
      r = rC
      RelationE.append(r)
      rLstMap.append((1,rCount))
    uHE = Variable(torch.LongTensor(uHE))
    bHE = Variable(torch.LongTensor(bHE))
    bTE = Variable(torch.LongTensor(bTE))
    HeadE = Variable(torch.LongTensor(HeadE))
    TailE = Variable(torch.LongTensor(TailE))
    RelationE = Variable(torch.LongTensor(RelationE))
    uHE = uHE.to(self.device)
    bHE = bHE.to(self.device)
    bTE = bTE.to(self.device)
    HeadE = HeadE.to(self.device)
    TailE = TailE.to(self.device)
    RelationE = RelationE.to(self.device)
    uHEE = self.model.getEntityEmbedding(uHE)
    uTEE = Variable(torch.FloatTensor(torch.zeros(len(uHE), self.embedDim)))
    uTEE = uTEE.to(self.device)
    uEE = torch.cat((uHEE, uTEE), 1)
    bHEE = self.model.getEntityEmbedding(bHE)
    bTEE = self.model.getEntityEmbedding(bTE)
    HeadEE = self.model.getEntityEmbedding(HeadE)
    TailEE = self.model.getEntityEmbedding(TailE)
    RelationEE = self.model.getRelationEmbedding(RelationE)
    bEE = torch.cat((bHEE, bTEE), 1)
    eE = torch.cat((uEE, bEE), 0)
    cE = cE.repeat(len(eLst),1)
    one = Variable(torch.FloatTensor([1.0]))
    one = one.to(self.device)    
    tmp = (one-cE)*eE
    tmpTransE = 0.2 * torch.abs(HeadEE + RelationEE - TailEE)
    s = torch.sum(tmp*tmp, dim=1) + torch.sum(tmpTransE, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getSortedKeyList(self, keyValueMap):
    keyLst = list(keyValueMap.keys())
    i=0
    while i<len(keyLst):
      j=i+1
      while j<len(keyLst):
        if keyValueMap[keyLst[i]]>keyValueMap[keyLst[j]]:
          tmp = keyLst[i]
          keyLst[i] = keyLst[j]
          keyLst[j] = tmp
        j+=1
      i+=1
    return keyLst

  def computeEmbeddingQuality(self):
    print('Embedding Quality:')
    allBRanks = self.getTestBClassEmbeddingQuality(list(self.dataObj.testBCMemberMap.keys()))
    result = self.accObj.computeMetrics(allBRanks)
    print('Performance Examples:')
    print('      ',self.getAccuracyPrintText(result))
    # allRanks = []
    # for r in allBRanks:
    #   allRanks.append(r)
    # result = self.accObj.computeMetrics(allRanks)
    # print(' > All Abox Classes')
    # print('      ',self.getAccuracyPrintText(result))

  def getUClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    uc2id = self.dataObj.getUConceptMap()
    entityLst = list(e2id.values())
    relationLst = list(uc2id.values())

    allRanks = []
    allCandidateLstLen = 0
    allTrueMembersCount = 0
    for c in classLst:
      trueMembers = set(self.dataObj.getUClassMembers(c))
      print(self.dataObj.id2uc[c], len(trueMembers))
      print("class:",self.dataObj.id2uc[c])
      ranks = []
      candidateLstLen = 0
      for trueMember in trueMembers:
        candidateLst = self.getUClassMembershipCandidateList(trueMember, trueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getUClassSpaceMembershipScore(self.getUClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList(scoreLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        ranks.append(rank)
        allRanks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += len(trueMembers)
      print('   ', self.accObj.computeMetrics(ranks), candidateLstLen/len(trueMembers))
    print(allCandidateLstLen/allTrueMembersCount)
    return allRanks

  def getBClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    bc2id = self.dataObj.getBConceptMap()
    entityLst = list(e2id.values())
    relationLst = list(bc2id.values())
    
    allRanks = []
    allCandidateLstLen = 0 
    allTrueMembersCount = 0
    for c in classLst:
      trueMembers = set(self.dataObj.getBClassMembers(c))
      print(self.dataObj.id2bc[c], len(trueMembers))
      print("class:",self.dataObj.id2uc[c])
      ranks = []
      candidateLstLen = 0
      for trueMember in trueMembers:
        candidateLst = self.getBClassMembershipCandidateList(trueMember, trueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getBClassSpaceMembershipScore(self.getBClassSpace(c), relationLst, candidateLst)
        rankLst = self.accObj.getRankList(scoreLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        ranks.append(rank)
        allRanks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += len(trueMembers)
      print('   ',self.accObj.computeMetrics(ranks), round(candidateLstLen/len(trueMembers),4))
    print(round(allCandidateLstLen/allTrueMembersCount,4))
    return allRanks

  def getTestBClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    bc2id = self.dataObj.getBConceptMap()
    entityLst = list(e2id.values())
    relationLst = classLst

    HRanks = []
    TRanks = []
    allRanks = []
    allCandidateLstLen = 0
    allTrueMembersCount = 0
    for c in classLst:
      testTrueMembers = set(self.dataObj.getTestBClassMembers(c))
      allTrueMembers = set(self.dataObj.getAllBClassMembers(c))
      print('Relation:\n', self.dataObj.id2bc[c], len(testTrueMembers))
      ranks = []
      candidateLstLen = 0
      for testTrueMember in testTrueMembers:
        candidateLst = self.getBClassMembershipHCandidateList(testTrueMember, allTrueMembers, entityLst)
        print(candidateLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getBClassSpaceMembershipScore(self.getBClassSpace(c), candidateLst)
        print(scoreLst)
        rankLst = self.accObj.getRankList(scoreLst)
        print(rankLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        print('Head Ranking:',rank)
        ranks.append(rank)
        allRanks.append(rank)
        candidateLst = self.getBClassMembershipTCandidateList(testTrueMember, allTrueMembers, entityLst)
        print(candidateLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getBClassSpaceMembershipScore(self.getBClassSpace(c), candidateLst)
        print(scoreLst)
        rankLst = self.accObj.getRankList(scoreLst)
        print(rankLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        print('Tail Ranking:', rank)
        ranks.append(rank)
        allRanks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += 2*len(testTrueMembers)
      print('Metrics:\n',self.accObj.computeMetrics(ranks), round(candidateLstLen/(2*len(testTrueMembers)),2))
    print('---------------Overall------------------')
    print('Average Candidates\n',round(allCandidateLstLen/allTrueMembersCount,2))
    print('---------------Overall------------------')
    return allRanks

  def getUClassMembershipCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    for e in entityLst:
      if e in trueMembers:
        continue
      candidateLst.append(e)
    return candidateLst

  def getBClassMembershipCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (h, e) in trueMembers:
        candidateLst.append((h, e))
      if not (e, t) in trueMembers:
        candidateLst.append((e, t))
    return candidateLst

  def getBClassMembershipHCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (e, t) in trueMembers:
        candidateLst.append((e, t))
    return candidateLst

  def getBClassMembershipTCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (h, e) in trueMembers:
        candidateLst.append((h, e))
    return candidateLst


  def getClassEmbeddingQualityOld(self, classLst):
    e2id = self.dataObj.getEntityMap()
    c2id = self.dataObj.getConceptMap()
    entityLst = list(e2id.values())

    gRanks = []
    for c in classLst:
      scoreLst = self.getClassSpaceMembershipScore(self.getClassSpace(c), entityLst)
      memberLst = self.dataObj.getClassMembers(c)
      rankMap = self.accObj.getMemberRank(entityLst, scoreLst, memberLst)
      for key in rankMap.keys():
        gRanks.append(rankMap[key])
    return self.accObj.computeMetrics(gRanks)

  def getUClassSpace(self, c):
    cT = Variable(torch.LongTensor([c]))
    cT = cT.to(self.device)
    uCE = self.model.getUConceptEmbedding(cT)
    return uCE

  def getBClassSpace(self, c):
    cT = Variable(torch.LongTensor([c]))
    #rT = Variable(torch.LongTensor([r]))
    cT = cT.to(self.device)
    #rT = rT.to(self.device)
    bCHE = self.model.getBConceptHEmbedding(cT)
    bCTE = self.model.getBConceptTEmbedding(cT)
    #rTE = self.model.getBConceptTEmbedding(rT)
    return bCHE, bCTE

  def getEntityEmbedding(self, eName):
    eT = Variable(torch.LongTensor([self.dataObj.getEntityId(eName)]))
    eT = eT.to(self.device)
    eE = self.model.getEntityEmbedding(eT)
    return eE

  def getClassEmbedding(self, cName):
    return self.getClassSpace(self.dataObj.getClassId(cName))

  # get relation embedding
  def getRelationEmbedding(self, rName):
    return self.getRelationSpace(self.dataObj.getClassId(rName))
  
  def getRelationSpace(self, r):
    rT = Variable(torch.LongTensor([r]))
    rT = rT.to(self.device)
    rE = self.model.getRelationEmbedding(rT)
    return rE

  def getAccuracyPrintText(self, resObj):
    retVal = 'MR='+'{:.4f}'.format(resObj['MR'])
    retVal += ', MRR='+'{:.6f}'.format(resObj['MRR'])
    retVal += ', Hit@1%='+'{:.4f}'.format(resObj['Hit@1%'])
    #retVal += ', Hit@2%='+'{:.2f}'.format(resObj['Hit@2%'])
    retVal += ', Hit@3%='+'{:.4f}'.format(resObj['Hit@3%'])
    #retVal += ', Hit@5%='+'{:.2f}'.format(resObj['Hit@5%'])
    retVal += ', Hit@10%='+'{:.4f}'.format(resObj['Hit@10%'])
    return retVal


