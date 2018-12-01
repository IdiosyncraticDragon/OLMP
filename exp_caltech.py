import sys
sys.path.insert(0, './python/')
import caffe
import numpy as np
from lcg_random import lcg_rand
import ncs
from easydict import EasyDict as edict
import time

start_time = time.time()
ncs_time = 0.
adjusting_time = 0.
retraining_time = 0.
# model files
proto='./models/caltech_caffenet/train_val_caltech.prototxt'
weights='/home/deepModels/caffe_models/bvlc_reference_caffenet/scratch_caltech_caffenet_train_iter_10000.caffemodel'
solver_path='./models/caltech_caffenet/caltech_solver.prototxt'
es_method='ncs'
# cpu/gpu
caffe.set_mode_gpu()
caffe.set_device(0)
# init solver
solver = caffe.SGDSolver(solver_path)
# basic parameters
#   accuracy constraint for pruning
acc_constrain=0.08
#   stop iteration count
niter = 15001
#   stop pruning iteration count
prune_stop_iter = 10000
# interval for 
prune_interval = 250
# interval for std variate
std_interval = 7000
#   the list of layer names
layer_name = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8*']
#   the dict of layer names to its arrary indices
layer_inds = {'conv1':0, 'conv2':1, 'conv3':2,'conv4':3,'conv5':4,'fc6':5,'fc7':6,'fc8*':7}
#   the dict of crates for each layer
crates = {'conv1':0.001, 'conv2':0.001, 'conv3':0.001,'conv4':0.001,'conv5':0.001,'fc6':0.001,'fc7':0.001,'fc8*':0.001}
#   the list of the crates
crates_list = 8*[0.001]
#   the gamma for each layer
gamma = {'conv1':0.00002, 'conv2':0.00002, 'conv3':0.00002,'conv4':0.00002,'conv5':0.00002,'fc6':0.0002,'fc7':0.0002,'fc8*':0.0002}
gamma_star = 0.0002
ncs_stepsize = 50
#   random see for numpy.random
seed=np.random.randint(1000000) 
#seed = 217750
np.random.seed([seed])
#   the dict to store intermedia results
es_cache = {}
#retrieval_tag=[]
r_count=0
# load the pretrained caffe model
if weights:
  solver.net.copy_from(weights)

# definition of many axuliliary methods
#   run the network on its dataset
def test_net(thenet, _start='data', _count=1):
   '''
    thenet: the object of network
    _start: the layer to start from
    _count: the number of batches to run
   '''
   scores = 0
   for i in range(_count):
      thenet.forward(start=_start)
      scores += thenet.blobs['accuracy'].data
   return scores/_count

#   Set the crates of each layer, the pruning will happen in the next forward action
def apply_prune(thenet, _crates):
   '''
      thenet: the model to be pruned
      _crates: the list of crates for layers
   '''
   for _id in range(len(layer_name)):
         if _crates[_id] < 0:
           continue
         layer_id = layer_name[_id]
         mask0 = thenet.params[layer_id][2].data.ravel()[0]
         if mask0 == 0:
           thenet.params[layer_id][2].data.ravel()[0] = -_crates[_id]
         elif mask0 == 1:
           thenet.params[layer_id][2].data.ravel()[0] = 1+_crates[_id]
         else:
           pdb.set_trace()

#  calcuate the sparsity of a network model
def get_sparsity(thenet):
   '''
     thenet: the network for checking
   '''
   remain = 0
   total = 0
   for layer_id in layer_name:
      remain += len(np.where(thenet.params[layer_id][2].data != 0)[0])
      remain += len(np.where(thenet.params[layer_id][3].data != 0)[0])
      total += thenet.params[layer_id][0].data.size
      total += thenet.params[layer_id][1].data.size
   #return total*1./(100.*remain)
   return remain*1./total

#  evaluate the accuracy of a network with a set of crates respect to a original accuracy
def evaluate(thenet, x_set, batchcount=1, accuracy_ontrain=0.9988):
   nofit=False
   fitness=[]
   X=[]
   for x in x_set:
     x_fit = 1.1
     apply_prune(thenet,x)
     acc = test_net(thenet, _start='conv1', _count=batchcount)
     if acc >= accuracy_ontrain - acc_constrain:
       x_fit = get_sparsity(thenet)
       nofit=True
     fitness.append(x_fit)
     X.append(x)
   return (X, fitness, nofit)
#------mian--------------
solver.step(1)
#  Adaptive dynamic surgery
for itr in range(niter):
   #r = np.random.rand()
   #if itr%500==0 and solver.test_nets[0].blobs['accuracy'].data >= 0.9909:
   #  retrieval_tag.append(itr)
   tmp_crates=[]
   tmp_ind = []
   for ii in layer_name:
      #tmp_crates.append(crates[ii]*(np.power(1+gamma[ii]*itr, -1)>np.random.rand()))
      tmp_tag = np.power(1+gamma[ii]*itr, -1)>np.random.rand()
      if tmp_tag:
        tmp_ind.append(ii)
        tmp_crates.append(tmp_tag*crates[ii])
   if itr < prune_stop_iter and itr%std_interval == 0:
      ncs_stepsize = ncs_stepsize/10.
   if itr%500 == 0:
        print "Compression:{}, Accuracy:{}".format(1./get_sparsity(solver.net), test_net(solver.net, _count=1, _start="conv1"))
   if len(tmp_ind)>0 and itr < prune_stop_iter:
         _tmp_c = np.array(len(crates_list)*[-1.])
         for t_name in tmp_ind:
            _tmp_c[layer_inds[t_name]] = crates[t_name]
         apply_prune(solver.net, _tmp_c)
   #if len(tmp_ind)>1 and itr < prune_stop_iter:
   if itr%prune_interval==0 and len(tmp_ind)>1 and itr < prune_stop_iter:
         ncs_start_t = time.time()
         accuracy_ = test_net(solver.net, _count=1, _start="conv1")

         # make sure a worable son x
         es = {}
         if es_method == 'ncs':
           __C = edict()
           __C.parameters = {'reset_xl_to_pop':False,'init_value':tmp_crates, 'stepsize':ncs_stepsize, 'bounds':[0.0, 10.], 'ftarget':0, 'tmax':1600, 'popsize':8, 'best_k':1}
           es = ncs.NCS(__C.parameters)
           print '***************NCS initialization***************'
           tmp_x_ = np.array(crates_list)
           tmp_input_x = tmp_crates
           for _ii in range(len(tmp_ind)):
             tmp_x_[layer_inds[tmp_ind[_ii]]] = tmp_input_x[_ii]
           _,tmp_fit,_= evaluate(solver.net, [tmp_x_], 1, accuracy_)
           es.set_initFitness(es.popsize*tmp_fit)
           print 'fit:{}'.format(tmp_fit)
           print '***************NCS initialization***************'

         # evolution loop
         while not es.stop():
           x = es.ask()
           X = []
           for x_ in x:
            tmp_x_ = np.array(crates_list)
            for _ii in range(len(tmp_ind)):
             tmp_x_[layer_inds[tmp_ind[_ii]]] = x_[_ii]
            X.append(tmp_x_)

           X_arrange,fit,has_fit_x = evaluate(solver.net, X, 1, accuracy_)

           X = []
           for x_ in X_arrange:
            tmp_x_ = np.array(len(tmp_ind)*[0.])
            for _ii in range(len(tmp_ind)):
             tmp_x_[_ii]= x_[layer_inds[tmp_ind[_ii]]] 
            X.append(tmp_x_)
           #print X,fit
           es.tell(X, fit)
           #es.disp(100)
           for _ii in range(len(tmp_ind)):
             crates_list[layer_inds[tmp_ind[_ii]]] = es.result()[0][_ii]
         for c_i in range(len(crates_list)):
            crates[layer_name[c_i]] = crates_list[c_i]
         es_cache[itr]={'compression':-es.result()[1], 'crates':crates_list[:]}
         _tmp_c = np.array(len(crates_list)*[-1.])
         for t_name in tmp_ind:
            _tmp_c[layer_inds[t_name]] = crates[t_name]
         apply_prune(solver.net, crates_list)

         ncs_end_t = time.time()
         ncs_time += (ncs_end_t - ncs_start_t)

   loop_start_t = time.time()
   # adjusting or retraining
   solver.step(1)

   loop_end_t = time.time()
   if itr < prune_stop_iter:
     adjusting_time += (loop_end_t - loop_start_t)
   else:
     retraining_time += (loop_end_t - loop_start_t)

# record 
import datetime
now = datetime.datetime.now()
time_styled = now.strftime("%Y-%m-%d %H:%M:%S")
out_ = open('record_{}.txt'.format(time_styled), 'w')
for key,value in es_cache.items():
   out_.write("Iteration[{}]:\t{}x\t{}\n".format(key,value['compression'],value['crates']))
out_.close()
print 'random seed:{}'.format(seed)
end_time = time.time()
#print(ncs_time)
#print(adjusting_time)
#print(retraining_time)
print('NCS time: %.4f mins' % (ncs_time/60.))
print('adjusting time: %.4f mins' % (adjusting_time/60.))
print('retraining time: %.4f mins' % (retraining_time/60.))
print('Total time: %.4f mins' % ((end_time - start_time)/60.))
#print "Retrieval accuracy @ iteration {}".format(retrieval_tag)
# save final model
#solver.net.save('./models/letnet5/9_letnet5_iter_{}.caffemodel'.format(itr+1))
