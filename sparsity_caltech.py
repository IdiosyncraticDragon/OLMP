import sys
sys.path.insert(0, './python/')
import caffe
import numpy as np
import pdb
#weights='/home/gitProject/Dynamic-Network-Surgery/models/lenet5/10_lenet_iter_28000.caffemodel'
weights='./models/caltech_caffenet/compressed_alexnet_caltech.caffemodel'
#weights='/home/gitProject/Dynamic-Network-Surgery/models/lenet5/caffe_lenet5_original.caffemodel'
#weights='/home/gitProject/Dynamic-Network-Surgery/models/lenet5/caffe_lenet5_sparse.caffemodel'
proto='./models/caltech_caffenet/train_val_caltech.prototxt'
net=caffe.Net(proto, weights, caffe.TEST)
total=0
aa=0
w_m=2
b_m=3

a1=len(np.where(net.params['conv1'][b_m].data != 0)[0])
a2=len(np.where(net.params['conv1'][w_m].data != 0)[0])
a3=len(np.where(net.params['conv2'][w_m].data != 0)[0])
a4=len(np.where(net.params['conv2'][b_m].data != 0)[0])
a5=len(np.where(net.params['conv3'][w_m].data != 0)[0])
a6=len(np.where(net.params['conv3'][b_m].data != 0)[0])
a7=len(np.where(net.params['conv4'][w_m].data != 0)[0])
a8=len(np.where(net.params['conv4'][b_m].data != 0)[0])
a9=len(np.where(net.params['conv5'][w_m].data != 0)[0])
a10=len(np.where(net.params['conv5'][b_m].data != 0)[0])
a11=len(np.where(net.params['fc6'][b_m].data != 0)[0])
a12=len(np.where(net.params['fc6'][w_m].data != 0)[0])
a13=len(np.where(net.params['fc7'][w_m].data != 0)[0])
a14=len(np.where(net.params['fc7'][b_m].data != 0)[0])
a15=len(np.where(net.params['fc8*'][w_m].data != 0)[0])
a16=len(np.where(net.params['fc8*'][b_m].data != 0)[0])

b1=net.params['conv1'][0].data.size+net.params['conv1'][1].data.size
b2=net.params['conv2'][0].data.size+net.params['conv2'][1].data.size
b3=net.params['conv3'][0].data.size+net.params['conv3'][1].data.size
b4=net.params['conv4'][0].data.size+net.params['conv4'][1].data.size
b5=net.params['conv5'][0].data.size+net.params['conv5'][1].data.size
b6=net.params['fc6'][0].data.size+net.params['fc6'][1].data.size
b7=net.params['fc7'][0].data.size+net.params['fc7'][1].data.size
b8=net.params['fc8*'][0].data.size+net.params['fc8*'][1].data.size

aa = a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15+a16
total = b1+b2+b3+b4+b5+b6+b7+b8

print 'Compression rate :{}% ({}x)'.format(100.- aa*100./total,total*1./aa)
print 'conv1:{}%'.format((a1+a2)*100./b1)
print 'conv2:{}%'.format((a3+a4)*100./b2)
print 'conv3:{}%'.format((a5+a6)*100./b3)
print 'conv4:{}%'.format((a7+a8)*100./b4)
print 'conv5:{}%'.format((a9+a10)*100./b5)
print 'fc6:{}%'.format((a11+a12)*100./b6)
print 'fc7:{}%'.format((a13+a14)*100./b7)
print 'fc8*:{}%'.format((a15+a16)*100./b8)
