#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <cmath>

namespace caffe {

// The constant NUM_THREADS should be equal to the value in CCMomentCalc
template <typename Dtype>
__global__ void CCMomentCollect(const int n, const Dtype* wb, const Dtype* mask,
    Dtype* mu, Dtype* std, unsigned int* count ) {  
  const int NUM_THREADS = 512;  
  __shared__ Dtype param [4*NUM_THREADS]; 
  __shared__ unsigned int tcount [2*NUM_THREADS];   
  unsigned int t = threadIdx.x; 
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  if (s+t < n){
    param[t] = fabs(mask[s+t]*wb[s+t]);
    param[t+2*NUM_THREADS] = mask[s+t]*wb[s+t]*wb[s+t];
    if(mask[s+t]*wb[s+t]!=0) tcount[t] = 1;
    else tcount[t] = 0;
  }
  else{
    param[t] = 0;param[t+2*NUM_THREADS] = 0;tcount[t] = 0;
  }
  if (s+t+NUM_THREADS < n){
    param[t+NUM_THREADS] = fabs(mask[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS]);
    param[t+3*NUM_THREADS] = mask[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS];
    if(mask[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS]!=0) tcount[t+NUM_THREADS] = 1;
    else tcount[t+NUM_THREADS] = 0;
  }
  else{
    param[t+NUM_THREADS] = 0;param[t+3*NUM_THREADS] = 0;tcount[t+NUM_THREADS] = 0;  
  }
  __syncthreads(); 
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      param[t] += param[t+stride]; 
      param[t+2*NUM_THREADS] += param[t+2*NUM_THREADS+stride];
      tcount[t] += tcount[t+stride];
    }
    __syncthreads();  
  }
  if (t == 0){
    mu   [blockIdx.x] = param[0];
    std  [blockIdx.x] = param[2*NUM_THREADS];
    count[blockIdx.x] = tcount[0]; 
  }      
}

// The constant NUM_THREADS should be equal to the value in CCMomentCalc
template <typename Dtype>
__global__ void CCNzeroCollect(const int n, const Dtype* mask, unsigned int* count ) {  
  const int NUM_THREADS = 512;  
  __shared__ unsigned int tcount [2*NUM_THREADS];   
  unsigned int t = threadIdx.x; 
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  tcount[t] = 0;
  if (s+t < n && mask[s+t]!=0){
    tcount[t] = 1;
  }
  tcount[t+NUM_THREADS] = 0;
  if (s+t+NUM_THREADS < n && mask[s+t+NUM_THREADS]!=0){
    tcount[t+NUM_THREADS] = 1;
  }
  __syncthreads(); 
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      tcount[t] += tcount[t+stride];
    }
    __syncthreads();  
  }
  if (t == 0){
    count[blockIdx.x] = tcount[0]; 
  }     
}

template <typename Dtype>
__global__ void CCMaskCalc(const int n, const Dtype* wb,
    Dtype* mask, Dtype mu, Dtype std, Dtype r) {
  CUDA_KERNEL_LOOP(index, n) {
    //  added by Guiying Li
    //if (mask[index] > 1)
    //  mask[index] = 1;
    //else if (mask[index] <0)
    //  mask[index] = 0;
    // ----Guiying Li-----
    if (mask[index]==1 && fabs(wb[index])<=0.9*max(mu+r*std,Dtype(0))) 
      mask[index] = 0;
    else if (mask[index]==0 && fabs(wb[index])>1.1*max(mu+r*std,Dtype(0)))
      mask[index] = 1;
  }
}

template <typename Dtype>
__global__ void CCcRateCalc(const int n,
    Dtype* mask) {
  CUDA_KERNEL_LOOP(index, n) {
     if (index == 0){
      if (mask[0] > 1){
        mask[0] = 1;
       } else if (mask[0] < 0){
        mask[0] = 0;
       }
     }
  }
}

template <typename Dtype>
__global__ void CCMaskApply(const int n, const Dtype* wb,
    const Dtype* mask, Dtype* wb_t) {
  CUDA_KERNEL_LOOP(index, n) {
    wb_t[index] = wb[index] * mask[index];    
  }
}

template <typename Dtype>
void CCMomentCalc(const int n, const Dtype* wb, const Dtype* mask, Dtype* mu, Dtype* std, unsigned int* ncount){ 
  const unsigned int NUM_THREADS = 512;
  Dtype* pmu_g; Dtype* pstd_g; unsigned int* pncount_g;
  Dtype* pmu_c; Dtype* pstd_c; unsigned int* pncount_c;
  int num_p = (n+(NUM_THREADS<<1)-1)/(NUM_THREADS<<1);  
  cudaMalloc(&pmu_g, sizeof(Dtype)  * num_p);
  cudaMalloc(&pstd_g, sizeof(Dtype) * num_p);
  cudaMalloc(&pncount_g, sizeof(unsigned int) * num_p);
  pmu_c = (Dtype*) malloc(num_p * sizeof(Dtype));
  pstd_c = (Dtype*) malloc(num_p * sizeof(Dtype)); 
  pncount_c = (unsigned int*) malloc(num_p * sizeof(unsigned int));      
  CCMomentCollect<Dtype><<<num_p,NUM_THREADS>>>(n, wb, mask, pmu_g, pstd_g, pncount_g);
  CUDA_POST_KERNEL_CHECK; 
  cudaMemcpy(pmu_c, pmu_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pstd_c, pstd_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pncount_c, pncount_g, sizeof(unsigned int) * num_p, cudaMemcpyDeviceToHost);      
  for (int i = 0; i < num_p; i++) {
    *mu += pmu_c[i];*std += pstd_c[i];*ncount += pncount_c[i];
  }       
  cudaFree(pmu_g);cudaFree(pstd_g);cudaFree(pncount_g);
  free(pmu_c);free(pstd_c);free(pncount_c);
}

template <typename Dtype>
void CCNZeroCalc(const int n, const Dtype* mask, unsigned int* ncount ){   
  const unsigned int NUM_THREADS = 512;
  unsigned int* pncount_g;
  unsigned int* pncount_c;
  int num_p = (n+(NUM_THREADS<<1)-1)/(NUM_THREADS<<1);  
  cudaMalloc(&pncount_g, sizeof(unsigned int) * num_p);
  pncount_c = (unsigned int*) malloc(num_p * sizeof(unsigned int));      
  CCNzeroCollect<Dtype><<<num_p,NUM_THREADS>>>(n, mask, pncount_g);
  CUDA_POST_KERNEL_CHECK; 
  cudaMemcpy(pncount_c, pncount_g, sizeof(unsigned int) * num_p, cudaMemcpyDeviceToHost);      
  for (int i = 0; i < num_p; i++) {
    *ncount += pncount_c[i];
  }       
  cudaFree(pncount_g);
  free(pncount_c);
}

template <typename Dtype>
void CInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {    

  const Dtype* weight = this->blobs_[0]->mutable_gpu_data();  
  Dtype* weightMask = this->blobs_[2]->mutable_gpu_data();
  Dtype* weightTmp = this->weight_tmp_.mutable_gpu_data();  
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL;
  if (this->bias_term_) {  
    bias = this->blobs_[1]->mutable_gpu_data();   
    biasMask = this->blobs_[3]->mutable_gpu_data();
    biasTmp = this->bias_tmp_.mutable_gpu_data();
  }   
  //  added by Guiying Li
  bool _update = false;
  Dtype* tmp_weightMask = this->blobs_[2]->mutable_cpu_data();
  if (tmp_weightMask[0] > 1){
        _update = true;
        this->crate = tmp_weightMask[0] - 1;
        tmp_weightMask[0] = 1;
  } else if (tmp_weightMask[0] < 0){
        _update = true;
        this->crate = -tmp_weightMask[0];
        tmp_weightMask[0] = 0;
  }
  weightMask = this->blobs_[2]->mutable_gpu_data();// update data
  //  ------Guiying--------
    
  if (this->phase_ == TRAIN){
		// Calculate the mean and standard deviation of learnable parameters 		
    if (this->std==0 && this->iter_==0){
      unsigned int ncount = 0;
      CCMomentCalc(this->blobs_[0]->count(), weight, weightMask, &mu, &std, &ncount);
      if (this->bias_term_) {  
        CCMomentCalc(this->blobs_[1]->count(), bias, biasMask, &mu, &std, &ncount); 
      }     
      this->mu /= ncount; this->std -= ncount*mu*mu; 
      this->std /= ncount; this->std = sqrt(std);  
      LOG(INFO)<<mu<<"  "<<std<<"  "<<ncount<<"\n";                    
    }
		
		// Demonstrate the sparsity of compressed fully-connected layer
		/********************************************************/
		/*if(this->iter_%100==0){
			unsigned int ncount = 0;
			CCNZeroCalc(this->blobs_[0]->count(), weightMask, &ncount);
			if (this->bias_term_) {  
				CCNZeroCalc(this->blobs_[1]->count(), biasMask, &ncount);   
			}
			LOG(INFO)<<ncount<<"\n";  			
		}*/	
		/********************************************************/		
		
		// Calculate the weight mask and bias mask with probability
    //Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    //if (pow(1+(this->gamma)*(this->iter_),-(this->power))>r && (this->iter_)<(this->iter_stop_)) { 
//    CCcRateCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[0]->count(), weightMask);
//    CUDA_POST_KERNEL_CHECK;    
    if (_update) { 	
      CCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, this->mu, this->std, this->crate);
      CUDA_POST_KERNEL_CHECK;    
      if (this->bias_term_) {  
        CCMaskCalc<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
          CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, this->mu, this->std, this->crate);
        CUDA_POST_KERNEL_CHECK;  
      }    
      _update = false;
    }
  }  
  
  // Calculate the current (masked) weight and bias
  CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
    CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
  CUDA_POST_KERNEL_CHECK;
  if (this->bias_term_) {  
    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, biasTmp);
    CUDA_POST_KERNEL_CHECK;  
  } 
   
	// Forward calculation with (masked) weight and bias 
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weightTmp, bottom_data, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            biasTmp, top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weightTmp, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            biasTmp, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
		const Dtype* weightMask = this->blobs_[2]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
		CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[2]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[2]->count(), weight_diff, weightMask, weight_diff);
    CUDA_POST_KERNEL_CHECK; 
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., weight_diff);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
		const Dtype* biasMask = this->blobs_[3]->gpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    // Gradient with respect to bias
    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[3]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[3]->count(), bias_diff, biasMask, bias_diff);
    CUDA_POST_KERNEL_CHECK; 		
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,bias_diff);
  }	
  if (propagate_down[0]) {
		const Dtype* weightTmp = this->weight_tmp_.gpu_data();        
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weightTmp, (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CInnerProductLayer);

}  // namespace caffe
