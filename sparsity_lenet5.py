import sys
sys.path.insert(0, './python/')
import caffe
import numpy as np
import pdb
weights='./models/lenet5/compressed_lenet5.caffemodel'
proto='./models/lenet5/lenet_train_test.prototxt'
net=caffe.Net(proto, weights, caffe.TEST)
total=0
aa=0
w_m=2
b_m=3

a1=len(np.where(net.params['conv1'][b_m].data != 0)[0])
a2=len(np.where(net.params['conv1'][w_m].data != 0)[0])
a3=len(np.where(net.params['conv2'][w_m].data != 0)[0])
a4=len(np.where(net.params['conv2'][b_m].data != 0)[0])
a5=len(np.where(net.params['ip1'][b_m].data != 0)[0])
a6=len(np.where(net.params['ip1'][w_m].data != 0)[0])
a7=len(np.where(net.params['ip2'][w_m].data != 0)[0])
a8=len(np.where(net.params['ip2'][b_m].data != 0)[0])

b1=net.params['conv1'][0].data.size+net.params['conv1'][1].data.size
b2=net.params['conv2'][0].data.size+net.params['conv2'][1].data.size
b3=net.params['ip1'][0].data.size+net.params['ip1'][1].data.size
b4=net.params['ip2'][0].data.size+net.params['ip2'][1].data.size

aa = a1+a2+a3+a4+a5+a6+a7+a8
total = b1+b2+b3+b4

print 'Compression rate :{}% ({}x)'.format(1- aa*1./total,total*1./aa)
print 'conv1:{}%'.format((a1+a2)*100./b1)
print 'conv2:{}%'.format((a3+a4)*100./b2)
print 'ip1:{}%'.format((a5+a6)*100./b3)
print 'ip2:{}%'.format((a7+a8)*100./b4)
