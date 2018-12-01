# -----------------------------
# Written by Guiying Li
# Copyright@UBRI, 2016
# -----------------------------

"""Python version of Negatively Correlated Search"""

import numpy as np
import pdb

class NCS:
   'This class contain the alogirhtm of NCS, and its API for invoking.'

   def __init__(self, parameters):
      '''Init an instance of NCS.'''
      self.init_value = parameters.init_value
      self.stepsize = parameters.stepsize
      self.bounds = parameters.bounds
      self.ftarget = parameters.ftarget
      self.popsize = parameters.popsize
      self.Tmax = parameters.tmax
      self.n = np.shape(parameters.init_value)[0]
      self.xl = self.bounds[0]*np.ones([parameters.popsize, self.n])
      self.xu = self.bounds[1]*np.ones([parameters.popsize, self.n])
      self.best_k = parameters.best_k
      self.k_min_f = np.zeros([self.best_k, 1])
      self.k_bestpop = np.zeros([self.best_k, self.n])
      #self.pop = np.random.rand(parameters.popsize, self.n)*0.1#(self.bounds[1] - self.bounds[0])
      #self.pop[0,:] = parameters.init_value
      #the same init values
      #self.pop = np.ones([parameters.popsize, self.n])*0.1
      if parameters.has_key('init_pop'):
        self.pop = np.tile(parameters.init_pop, (parameters.popsize,1))[:parameters.popsize,:]
      else:
        self.pop = np.tile(self.init_value, (parameters.popsize,1))
      if parameters.reset_xl_to_pop:
        self.xl = self.pop

   def set_initFitness(self, fitness, sigma=None):
      arg_min = np.argmin(fitness)
      self.min_f = fitness[arg_min]
      self.bestpop = self.pop[arg_min, :]
      if sigma==None:
        #self.sigma = np.ones([self.popsize, self.n]) * ((np.array(self.bounds[1]) - np.array(self.bounds[0]))*1./self.popsize)
        self.sigma = np.ones([self.popsize, self.n]) * self.stepsize
      else:
        self.sigma = np.tile(sigma, (self.popsize, 1))
      self.r = 0.99
      self.fit = np.array(fitness)
      self.flag = np.zeros([self.popsize, 1])
      self.epoch = self.popsize
      self.lambda_ = np.ones([self.popsize, 1])
      self.lambda_sigma = 0.1
      self.lambda_range = self.lambda_sigma
      self.FES = self.popsize
      self.Gen = 0
      # record best
      self.k_min_f[0,0] = self.min_f
      self.k_bestpop[0,:] = self.bestpop
   
   def set_lowerBound(self, lowerBound):
      '''Set the lower bound for each individual, so that no extra search will happen'''
      self.xl = lowerBound

   def stop(self):
      '''Return the finishing state of algorithm'''
      return self.FES > self.Tmax

   def result(self):
      '''Return the results'''
      return (self.bestpop, self.min_f, self.k_bestpop, self.k_min_f)

   def disp(self, count):
      if self.Gen % count == 0:
        print "%-----------------Best so far-----------------------%"
        print "[{}]best fitness: {}".format(self.Gen/count, self.min_f)
        print "k best records"
        for i in range(self.best_k):
          print "fitness of record[{}]:{}".format(i, self.k_min_f[i])
        print "%---------------------------------------------------%"
        return False
      else:
        return False


   def ask(self):
      '''Return the next population'''
      uSet = self.pop + self.sigma * np.random.randn(self.popsize, self.n)
      #check the boundary
      #pos = np.where((uSet < self.xl) + (uSet > self.xu))
      pos = np.where(uSet < self.xl)
      uSet[pos] = self.xl[pos]+0.0001
      pos = np.where(uSet > self.xu)
      uSet[pos] = self.xu[pos]-0.0001
      #while (pos[0].size > 0):
      #  uSet[pos] = (self.pop + self.sigma * np.random.randn(self.popsize, self.n))[pos]
      #  pos = np.where((uSet < self.xl) + (uSet > self.xu))
      #uSet[pos] = 2*self.xl[pos] - uSet[pos]
      #bound_condition = (uSet[pos] > self.xu[pos])
      #uSet[pos] = bound_condition*self.xu[pos] + np.logical_not(bound_condition)*uSet[pos]

      #uSet[pos] = 2*self.xu[pos] - uSet[pos]
      #bound_condition = (uSet[pos] < self.xl[pos])
      #uSet[pos] = bound_condition*self.xl[pos] + np.logical_not(bound_condition)*uSet[pos]

      listResult = []
      for i in range(self.popsize):
        listResult.append(uSet[i,:])
      return listResult

   def tell(self, uSet, fitSet):
      '''Tell the algorithm about the pair of population and fitness.'''
      #record once evaluation
      self.FES = self.FES + self.popsize
      self.Gen = self.Gen + 1

      uSet = np.array(uSet)
      fitSet = np.array(fitSet)

      # normalize fitness values
      arg_min = np.argmin(fitSet)
      if fitSet[arg_min] < self.min_f:
        self.min_f = fitSet[arg_min]
        self.bestpop = uSet[arg_min]
        #record the k best
        record_tag = True
        # records should be identical
        for i_k in range(self.best_k):
           if self.k_min_f[i_k] == self.min_f:
              record_tag = False
        if record_tag:
          tmp_max_ind = np.argmax(self.k_min_f)
          self.k_min_f[tmp_max_ind, 0] = self.min_f
          self.k_bestpop[tmp_max_ind, :] = self.bestpop

      tempFit = self.fit - self.min_f
      tempTrialFit = fitSet - self.min_f
      normFit = tempFit / (tempFit + tempTrialFit)
      normTrialFit = tempTrialFit / (tempFit + tempTrialFit)

      # calculate the BHattacharyya distance
      pCorr = 1e300*np.ones([self.popsize, self.popsize])
      trialCorr = 1e300*np.ones([self.popsize, self.popsize])

      for i in range(self.popsize):
        for j in range(self.popsize):
         if j != i:
            # BD
            m1 = self.pop[i,:] - self.pop[j,:]
            c1 = (np.power(self.sigma[i,:],2) + np.power(self.sigma[j,:],2))/2.
            tempD = 0
            for k in range(self.n):
               tempD = tempD + np.log(c1[k]) - 0.5*(np.log(np.power(self.sigma[i,k],2)) + np.log(np.power(self.sigma[j,k],2)))
            pCorr[i,j] = (1./8) * m1.dot(np.diag(1./c1)).dot(np.transpose(m1)) + 0.5*tempD
            # BD
            m2 = uSet[i,:] - self.pop[j,:]
            trialCorr[i,j] = (1./8) * m2.dot(np.diag(1./c1)).dot(np.transpose(m2)) + 0.5*tempD
      pMinCorr = pCorr.min(1)
      trialMinCorr = trialCorr.min(1)

      # normalize correlation values
      normCorr = pMinCorr / (pMinCorr + trialMinCorr)
      normTrialCorr = trialMinCorr / (pMinCorr + trialMinCorr)
      self.lambda_ = 1 + self.lambda_sigma*np.random.randn(self.popsize)
      self.lambda_sigma = self.lambda_range - self.lambda_range*self.Gen/(self.Tmax*1./self.popsize)
      pos = np.where(((self.lambda_ * normTrialCorr) > normTrialFit)*(fitSet < 0))
      pos = pos[0]
      self.pop[pos, :] = uSet[pos, :]
      self.fit[pos] = fitSet[pos]
      self.flag[pos] = self.flag[pos] + 1
      # i/5 successful rule
      if self.Gen % self.epoch == 0:
        for i in range(self.popsize):
           if self.flag[i]*1./self.epoch > 0.2:
             self.sigma[i,:] = self.sigma[i,:]/self.r
           elif self.flag[i]*1./self.epoch < 0.2:
             self.sigma[i,:] = self.sigma[i,:]*self.r
        self.flag = np.zeros([self.popsize,1])

