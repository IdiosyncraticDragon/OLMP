import sys
sys.path.insert(0, './python/')
import caffe
import numpy as np
import pdb
#weights='./models/lenet300100/caffe_lenet300100_original.caffemodel'
weights='./models/lenet300100/compressed_lenet300100.caffemodel'
#weights='/home/gitProject/Dynamic-Network-Surgery/models/lenet300100/caffe_lenet300100_sparse.caffemodel'
proto='./models/lenet300100/lenet_train_test.prototxt'
net=caffe.Net(proto, weights, caffe.TEST)
total=0
aa=0
# for each layer, a mask is applied to the original weights and bias.
# here, for net.params['ip1'], net.params['ip1'][0] is the weights, net.params['ip1'][1] is the bias,
#       net.params['ip1'][2] is the mask for the weights, net.params['ip1'][3] is the mask for the bias.
#       if one of the element value in the mask is 0, the corresponding element in network is pruned.
w_m=2
b_m=3

a1=len(np.where(net.params['ip1'][b_m].data != 0)[0])
a2=len(np.where(net.params['ip1'][w_m].data != 0)[0])
a3=len(np.where(net.params['ip2'][w_m].data != 0)[0])
a4=len(np.where(net.params['ip2'][b_m].data != 0)[0])
a5=len(np.where(net.params['ip3'][b_m].data != 0)[0])
a6=len(np.where(net.params['ip3'][w_m].data != 0)[0])

b1=net.params['ip1'][0].data.size+net.params['ip1'][1].data.size
b2=net.params['ip2'][0].data.size+net.params['ip2'][1].data.size
b3=net.params['ip3'][0].data.size+net.params['ip3'][1].data.size

aa = a1+a2+a3+a4+a5+a6
total = b1+b2+b3

print 'Compression rate :{}% ({}x)'.format(1- aa*1./total,total*1./aa)
print 'ip1:{}%'.format((a1+a2)*100./b1)
print 'ip2:{}%'.format((a3+a4)*100./b2)
print 'ip3:{}%'.format((a5+a6)*100./b3)
