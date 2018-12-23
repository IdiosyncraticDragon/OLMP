# Optimization based Layer-wise Magnitude-based Pruning for DNN Compression 
Thank you for everyone who is intereted in our work.  
This repository is the implementation of OLMP. In experiments of LeNet-5 and LeNet-300-100, we have fixed the random seeds in python scripts for the purpose of reproducting the results shown in our paper. For AlexNet-Caltech, unfortunately, it has the dropout layers with the random seed inside Caffe framework which is the random seed we did not recorded during our experiments. Instead, We provide the compressed model of AlexNet-Caltech whoes result is reported in our paper. Users can also run the script of AlexNet-Caltech several times to reproduce a similar result compared to the one in our paper.

This project is based on [Caffe](https://github.com/BVLC/caffe) and [Dynamic surgery](https://github.com/yiwenguo/Dynamic-Network-Surgery). Thanks to the authors of these two projects.

Note that for who want run the code, there is no need to pre-install Caffe! This project is an edited version of Caffe, and an specifical Caffe will be genered by building the code in the project directly.

## Testing enviroment
- Docker image: [kaixhin/cuda-caffe:8.0](https://hub.docker.com/r/kaixhin/cuda-caffe/)
	- Ubuntu 16.04.2 LTS
	- g++ 5.4.0
	- python 2.7.12
- 1 x NVIDIA TITANX pascal
- 2 x Intel(R) Xeon(R) CPU E5-2683 v3 @ 2.00GHz
- 64 GB Memory

## Requiremetns
- The requirements are the same as Caffe.
- easydict package for python

## Installation
- Install all the requirements of Caffe. You can all download a docker image of Caffe directly.
- Go into the first level of the project folder, namely ./OLMP.
- Check the file "Makefile.config" to insure all the settings are suitable for your own enviroment, like the version of g++.
- make all
- make pycaffe
- pip install easydict (version 1.9 was tested)

### Problems
Most of the problems in making are caused by the settings of enviroment. Please refer to https://github.com/BVLC/caffe/issues for help.

## Data
We upload all the processed data (the data file can be directed used in this project, users can also process data by themself following the details described in our paper) to Baidu Wangpan (Usage: past the link to the internet explorer and use the password to download the file).

For MNIST:

link: https://pan.baidu.com/s/17lem8wVV9nd_dZxVd8FsHA

password: 40fa

For Caltech-256:

link: https://pan.baidu.com/s/1eezA0uCKHy0OLCz34XBHUQ

password: v5s8

## Tutorial
To all the experiments below, the user should edit the address of the data in .prototxt.

### LeNet-300-100
To compress the model LeNet-300-100, it firstly needs to make sure the data address in ./models/lenet300100/lenet\_train\_test.prototxt is the one in correct.
Please Run:
```
python exp_lenet300100.py
```

### LeNet-5
To compress the model LeNet-5, please Run:
```
python exp_lenet5.py
```

### AlexNet-Caltech
To compress the model AlexNet-Caltech, please Run:
```
python exp_caltech.py
```
Note that the reference model for exp\_caltech.py is too large for uploading to github, so that we upload it to Baidu Wangpan:

Reference model of AlexNet-Caltech:

link: https://pan.baidu.com/s/1cWrgx29icUR680U1mm9YoA

password: 8r48

(Usage: past the link to the internet explorer and use the password to download the file)

### Check the compressed model

For Lenet-300-100 and Lenet-5, the user can find the compressional results are the same as that reported in our paper. Or, the user can run sparsity\_lenet5.py and sparsity\_lenet300100.py to check the sparsity of the model compressed by us.

For Lenet-300-100, the model compressed by use is provided at:
```
./models/lenet300100/compressed_lenet300100.caffemodel
```
Run sparsity\_lenet300100.py to check the sparsity.

For Lenet-5, the model compressed by us is provided at:
```
./models/lenet5/compressed_lenet5.caffemodel
```

For AlexNet-Caltech, since we do not fixed the random seed for droupout operation, it can not guarantee the result is the same as that in our paper. Consider about this, we provide the model compressed by us.

Compressed model of AlexNet-Caltech:

link: https://pan.baidu.com/s/1qdsAEsBYFe6zTnmX_yO8ZA

password: 3ygh

(Usage: past the link to the internet explorer and use the password to download the file)


### Output format
Take the output of exp\_lenet5.py as an example:
```
I1129 04:04:49.392139  6877 solver.cpp:226] Iteration 29600, loss = 0.152239
I1129 04:04:49.392174  6877 solver.cpp:242]     Train net output #0: accuracy = 0.96875
I1129 04:04:49.392191  6877 solver.cpp:242]     Train net output #1: loss = 0.152239 (\* 1 = 0.152239 loss)
I1129 04:04:49.392267  6877 solver.cpp:521] Iteration 29600, lr = 0.00356228
I1129 04:04:50.325364  6877 solver.cpp:226] Iteration 29700, loss = 0.00853293
I1129 04:04:50.325392  6877 solver.cpp:242]     Train net output #0: accuracy = 1
I1129 04:04:50.325405  6877 solver.cpp:242]     Train net output #1: loss = 0.00853293 (\* 1 = 0.00853293 loss)
I1129 04:04:50.325415  6877 solver.cpp:521] Iteration 29700, lr = 0.00355555
I1129 04:04:51.243219  6877 solver.cpp:226] Iteration 29800, loss = 0.0735124
I1129 04:04:51.243247  6877 solver.cpp:242]     Train net output #0: accuracy = 0.96875
I1129 04:04:51.243260  6877 solver.cpp:242]     Train net output #1: loss = 0.0735124 (\* 1 = 0.0735124 loss)
I1129 04:04:51.243270  6877 solver.cpp:521] Iteration 29800, lr = 0.00354885
I1129 04:04:52.162196  6877 solver.cpp:226] Iteration 29900, loss = 0.0591469
I1129 04:04:52.162223  6877 solver.cpp:242]     Train net output #0: accuracy = 0.984375
I1129 04:04:52.162238  6877 solver.cpp:242]     Train net output #1: loss = 0.0591469 (\* 1 = 0.0591469 loss)
I1129 04:04:52.162248  6877 solver.cpp:521] Iteration 29900, lr = 0.00354218
I1129 04:04:53.071841  6877 solver.cpp:399] Snapshotting to binary proto file models/lenet5/10_lenet_iter_30000.caffemodel
I1129 04:04:53.084738  6877 solver.cpp:684] Snapshotting solver state to binary proto filemodels/lenet5/10_lenet_iter_30000.solverstate
I1129 04:04:53.091256  6877 solver.cpp:314] Iteration 30000, Testing net (#0)
I1129 04:04:53.717361  6877 solver.cpp:363]     Test net output #0: accuracy = 0.9909
I1129 04:04:53.717402  6877 solver.cpp:363]     Test net output #1: loss = 0.0321025 (\* 1 = 0.0321025 loss)
I1129 04:04:53.724666  6877 solver.cpp:226] Iteration 30000, loss = 0.00549194
I1129 04:04:53.724690  6877 solver.cpp:242]     Train net output #0: accuracy = 1
I1129 04:04:53.724704  6877 solver.cpp:242]     Train net output #1: loss = 0.00549194 (\* 1 = 0.00549194 loss)
I1129 04:04:53.724714  6877 solver.cpp:521] Iteration 30000, lr = 0.00353553
Compression:297.70718232, Accuracy:1.0
random seed:981118
```
In the output text, "random seed" indicates the random seed used in python script. Note that the random seed in C++ code is not restricted, so that if the network contain random operations like dropout, the setting of "random seed" is useless.

"Compression: xxx, Accuracy: xxx" indicates the current Pruning Ratio and the accuracy of the pruned model in the current batch. For example 
```
I1129 04:04:53.724714  6877 solver.cpp:521] Iteration 30000, lr = 0.00353553
Compression:297.70718232, Accuracy:1.0
```
means the Pruning Ration is 297.7 and the accuracy of the pruned model on the batch of iteration 30000 is 100%.

```
I1129 04:04:53.091256  6877 solver.cpp:314] Iteration 30000, Testing net (#0)
I1129 04:04:53.717361  6877 solver.cpp:363]     Test net output #0: accuracy = 0.9909
I1129 04:04:53.717402  6877 solver.cpp:363]     Test net output #1: loss = 0.0321025 (\* 1 = 0.0321025 loss)
```
This indicates the accuracy of the pruned model on the whole testing set. Here the testing accuracy is 0.9909 which is the same to the accuracy of the reference model.

### How to customize
- The project is based on Dynamic Surgery, so that the framework is similar.
- Firstly, the user should edit the prototxt of model, and change the type of convlutional layers and innerproduct layers to "CConvolution" and "CInnerProduct". Note that the cinner\_product\_param and cconvolution\_param should also be specified, but the values can be arbitrary becuase these values do not affect the pruning actually. For this step, take models/lenet5/lenet\_train\_test.prototxt, models/lenet300100/lenet\_train\_test.prototxt and  models/caltech\_caffenet/train\_val\_caltech.prototxt. This is similar to Dynamic Surgey.
- Secondly, the user shold write a python file to compress the models. Take exp\_lenet300100.py, exp\_lenent5.py and exp\_caltech.py as examples. All the values of hyper parameters of pruning are specified in the python scripts.

## Explanation for code
For the python scirpts, we have already written detailed comments inside the scripts.

For the editing of the C++ code. We have edited ./src/caffe/layers/compress\_inner\_product\_layer.cu and ./src/caffe/layers/compress\_conv\_layer.cu. In the forwarding passing:
```
 template <typename Dtype>
 void CConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top) {
...
// added by Guiying Li
180   bool _update = false;
181   Dtype* tmp_weightMask = this->blobs_[2]->mutable_cpu_data();
182   if (tmp_weightMask[0] > 1){
183         _update = true;
184         this->crate = tmp_weightMask[0] - 1;
185         tmp_weightMask[0] = 1;
186   } else if (tmp_weightMask[0] < 0){
187         _update = true;
188         this->crate = -tmp_weightMask[0];
189         tmp_weightMask[0] = 0;
190   }
191   weightMask = this->blobs_[2]->mutable_gpu_data();//update data
192   // -------Guiying------
...
}
```
The first value of the mask is extracted, if the value is larger than 0, than means the value is composed of (the current crate of this layer) + (the mask element is 1); else if the value is smaller than 0, that means the value is composed of (current crate of this layer)\*(-1) + (the mask element is 0). Note that the first value of the mask can be edited by users using the python api, so that the user can use python code to control the pruning. Take the exp\_lenet300100.py as example:
```
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

```
Here, when the algorithm has chosen the crates for each layer (the pruning related hyper-parameters), the python script transfer these crates to pruning process by encode them into the first element of the mask in each layer.

## Citation
Please cite our work as:

```
@inproceedings{li2018olmp,		
  title = {Optimization based Layer-wise Magnitude-based Pruning for DNN Compression},
  author = {Guiying Li and Chao Qian and Chunhui Jiang and Xiaofen Lu and Ke Tang},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  address={Stockholm, Sweden},
  pages={2383--2389},
  year = {2018}
}
```
and the other citations may also be needed:

Caffe

```
@article{jia2014caffe,
	Author = {Yangqing Jia and  Evan Shelhamer and  Jeff Donahue and  Sergey Karayev and  Jonathan Long and Ross Girshick and Sergio Guadarrama and Trevor Darrell},
	Journal = {arXiv:1408.5093},
	Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
	Year = {2014}
}
```

Dynamic surgery:

```
@inproceedings{guo2016dynamic,		
  title = {Dynamic Network Surgery for Efficient DNNs},
  author = {Yiwen Guo and Anbang Yao and Yurong Chen},
  booktitle = {Advances in neural information processing systems (NIPS)},
  address={Barcelona, Spain},
  pages={1379--1387},
  year = {2016}
}
```

Negatively Correlated Search:
```
@article{tang2016negatively,
	author={Ke Tang and Peng Yang and Xin Yao},
	title={Negatively correlated search},
	journal={IEEE Journal on Selected Areas in Communications},
	volume={34},
	number={3},
	pages={542--550},
	year={2016}
}
```
