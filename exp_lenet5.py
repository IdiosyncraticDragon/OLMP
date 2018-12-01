import sys
sys.path.insert(0, './python/')
import caffe
import numpy as np
from lcg_random import lcg_rand
import ncs
from easydict import EasyDict as edict

# model files
proto='./models/lenet5/lenet_train_test.prototxt'
weights='./models/lenet5/caffe_lenet5_original.caffemodel'
solver_path='./models/lenet5/lenet_solver.prototxt'
es_method='ncs'
# cpu/gpu
caffe.set_mode_gpu()
caffe.set_device(0)
# init solver
solver = caffe.SGDSolver(solver_path)
# basic parameters
#   accuracy constraint for pruning
acc_constrain=0.05
#   stop iteration count
#niter = 20501
niter = 30001
#   stop pruning iteration count
prune_stop_iter = 15000
#   the list of layer names
layer_name = ['conv1','conv2','ip1', 'ip2']
#   the dict of layer names to its arrary indices
layer_inds = {'conv1':0, 'conv2':1, 'ip1':2, 'ip2':3}
#   the dict of crates for each layer
#crates = {'conv1':1.95, 'conv2':3.35, 'ip1':3.7, 'ip2':2.8}
#crates = {'conv1':-1.54, 'conv2':-1.40, 'ip1':-1.63, 'ip2':-1.37}
crates = {'conv1':0.001, 'conv2':0.001, 'ip1':0.001, 'ip2':0.001}
#   the list of the crates
#crates_list = [1.95, 3.35, 3.7, 2.8]
#crates_list = [-1.54, -1.40, -1.63, -1.37]
crates_list = [0.001, 0.001, 0.001, 0.001]
#   the gamma for each layer
gamma = {'conv1':0.00002, 'conv2':0.00002, 'ip1':0.0002, 'ip2':0.0002}
gamma_star = 0.0002
ncs_stepsize = 50
#   random see for numpy.random
seed=np.random.randint(1000000) 
seed=981118#seed 93306,124x,0.04;78011, 127x, 0.05,430000,150x, 515769,185x
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
def test_net(thenet, _start='mnist', _count=1):
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
   fitness=[]
   X=[]
   for x in x_set:
     x_fit = 1.1
     apply_prune(thenet,x)
     acc = test_net(thenet, _start='conv1', _count=batchcount)
     if acc >= accuracy_ontrain - acc_constrain:
       x_fit = get_sparsity(thenet)
     fitness.append(x_fit)
     X.append(x)
   return (X, fitness)
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
   if itr < 20000 and itr%10000 == 0:
      ncs_stepsize = ncs_stepsize/10.
   if itr%500 == 0:
        print "Compression:{}, Accuracy:{}".format(1./get_sparsity(solver.net), test_net(solver.net, _count=1, _start="conv1"))
   if len(tmp_ind)>0 and itr < prune_stop_iter:# run at window @6
         _tmp_c = np.array(len(crates_list)*[-1.])
         for t_name in tmp_ind:
            _tmp_c[layer_inds[t_name]] = crates[t_name]
         apply_prune(solver.net, _tmp_c)
   #if len(tmp_ind)>1 and itr < prune_stop_iter:
   if itr%1000==0 and len(tmp_ind)>1 and itr < prune_stop_iter:# run at window @3
         accuracy_ = test_net(solver.net, _count=1, _start="conv1")
         es = {}
         if es_method == 'ncs':
           __C = edict()
           __C.parameters = {'reset_xl_to_pop':False,'init_value':tmp_crates, 'stepsize':ncs_stepsize, 'bounds':[0.0, 20.], 'ftarget':0, 'tmax':1600, 'popsize':10, 'best_k':1}
           es = ncs.NCS(__C.parameters)
           print '***************NCS initialization***************'
           tmp_x_ = np.array(crates_list)
           tmp_input_x = tmp_crates
           for _ii in range(len(tmp_ind)):
             tmp_x_[layer_inds[tmp_ind[_ii]]] = tmp_input_x[_ii]
           _,tmp_fit = evaluate(solver.net, [tmp_x_], 1, accuracy_)
           es.set_initFitness(es.popsize*tmp_fit)
           print 'fit:{}'.format(tmp_fit)
           print '***************NCS initialization***************'
         while not es.stop():
           x = es.ask()
           X = []
           for x_ in x:
            tmp_x_ = np.array(crates_list)
            for _ii in range(len(tmp_ind)):
             tmp_x_[layer_inds[tmp_ind[_ii]]] = x_[_ii]
            X.append(tmp_x_)

           X_arrange,fit = evaluate(solver.net, X, 1, accuracy_)

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
   solver.step(1)
# record 
import datetime
now = datetime.datetime.now()
time_styled = now.strftime("%Y-%m-%d %H:%M:%S")
out_ = open('record_{}.txt'.format(time_styled), 'w')
for key,value in es_cache.items():
   out_.write("Iteration[{}]:\t{}x\t{}\n".format(key,value['compression'],value['crates']))
out_.close()
print 'random seed:{}'.format(seed)
#print "Retrieval accuracy @ iteration {}".format(retrieval_tag)
# save final model
#solver.net.save('./models/letnet5/9_letnet5_iter_{}.caffemodel'.format(itr+1))
