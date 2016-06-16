#include <iostream>
#include <fstream>
#include <string>
#include "math.h"
#include <cub/cub.cuh>
#include "flexarray.h"

//#define __CUDA__ 1
//#define __DEBUG__ 1

#define MAX_DEPTH 8

cudaStream_t myStream;
cudaStream_t debugStream;

#define cudaCheckError() cudaChkError(__LINE__, __FILE__)
void inline cudaChkError(int line, const char* filename) {
   cudaError_t err = cudaGetLastError();
   if (err) std::cout << "Error on line " << line << " of " << filename << " : " << cudaGetErrorString(err) << std::endl;
}
template <unsigned int i>
__global__ void debugMark() {
}

size_t description_outer(int *descript1, const size_t size1, const size_t d1, 
                         int *descript2, const size_t size2, const size_t d2,
                         int *descript_out);
void two_list_freq_count(flexarray_bool *data1/* txi */, 
                         flexarray_bool *data2/* txj */,  int *count/* ixj */, 
                         flexarray_bool *outdata/* tx(i*j) */);
int threshold_count(int *descriptions/* (i*j)xd */, const size_t i, const size_t j, const int d, int *count/* j*i */, 
                    flexarray_bool *data/* tx(i*j) */, const size_t t, const int threshold); 
int threshold_count_gpu(int *descriptions/* (i*j)xd */, const size_t i, const size_t j, const int d, int *count/* j*i */, 
                        flexarray_bool *data/* tx(i*j) */, const size_t t, const int threshold); 


int main(int argc, char** argv) {

   int n_items;
   int n_trans;
   int min_support;
   flexarray_bool *data, *newdata, *first_data, *h_data;
   int **freqpattern;
   int *this_descriptions = NULL;
   int *next_descriptions = NULL;
   int *first_descriptions = NULL;
   int *freq=NULL, *new_freq;
   std::string** names;

   if (argc < 2) {
      std::cout << "Usage: " << std::endl << "     lb_apriori <input file name>" << std::endl;
      return 0;
   }
#if defined(__CUDA__)
   std::cout << "Processing on GPU" << std::endl;
#else
   std::cout << "Processing on CPU" << std::endl;
#endif
   std::cout << "Reading input data..." << std::endl;
   std::ifstream fin(argv[1], std::ifstream::in);
   fin >>min_support>>n_items >> n_trans;
   std::cout << n_trans << " transactions with " << n_items << " items" << std::endl;
   

   freqpattern = (int**)malloc(sizeof(int*)*(MAX_DEPTH+1));
   data = new flexarray_bool(n_trans, n_items, true);
   h_data = new flexarray_bool(n_trans, n_items);
   names = (std::string**)malloc(sizeof(std::string*)*n_items);
   std::cout << "Found items named "<<std::endl;
   for (int p=0;p<n_items;p++) {
      std::string tmp;
      fin>>tmp;
      names[p] = new std::string(tmp);
      std::cout<<*names[p]<<std::endl;
   }
   //  Set input data
   for (int q=0; q<n_trans;q++) {
      for (int p=0; p<n_items; p++) {
         int tmp;
         fin>>tmp;
         if (tmp>0) h_data->set(q,p,true);
         else h_data->set(q,p,false);
         //if (tmp>0) (*data)(q,p) = true;
         //else (*data)(q,p) = false;
      }
   }
   cudaMemcpy(data->data, h_data->data, sizeof(unsigned int)*h_data->real_c*h_data->r/32, cudaMemcpyHostToDevice);
   
   int this_size=n_items;
   first_descriptions = (int*)malloc(sizeof(int)*n_items);
   this_descriptions = (int*)malloc(sizeof(int)); //allocate something for freeing
   int last_size=1;
   first_data = new flexarray_bool(n_trans, 1, true);
   for (int p=0; p<n_items; p++) {
      first_descriptions[p] = p;
   }
   for (int q=0; q<n_trans; q++) {
      h_data->set(q,0,true);
   }
   cudaMemcpy(first_data->data, h_data->data, sizeof(unsigned int)*first_data->real_c*first_data->r/32, cudaMemcpyHostToDevice);
   delete(h_data);

   cudaStreamCreate(&myStream);
   cudaStreamCreate(&debugStream);

   for (int depth=1;depth<MAX_DEPTH;depth++) {
      std::cout << std::endl << "    ****  DEPTH = " << depth << "  **** " << std::endl;
      cudaCheckError();
      this_descriptions = next_descriptions;
      next_descriptions = (int*)malloc(sizeof(int)*depth*last_size*n_items);
      this_size = last_size * n_items;
      //next_size = cull_descriptions(next_descriptions, next_size, depth);
#if defined(__CUDA__)
      cudaMalloc(&freqpattern[depth],sizeof(int)*this_size);
      cudaMemsetAsync(freqpattern[depth],0 ,sizeof(int)*this_size, myStream);
      cudaCheckError();
#else
      freqpattern[depth] = (int*)malloc(sizeof(int)*this_size);
#endif
      newdata = new flexarray_bool(n_trans, this_size, true);
      two_list_freq_count(data, first_data, freqpattern[depth], newdata);
      cudaCheckError();
      debugMark<1><<<1,1,0,debugStream>>>();
      this_size = description_outer(this_descriptions, last_size, depth-1, first_descriptions, n_items, 1,
                                    next_descriptions);
      debugMark<2><<<1,1,0,debugStream>>>();
#if defined(__DEBUG__)
#if defined(__CUDA__)
      new_freq = (int*) realloc(freq, sizeof(int)*this_size);
      freq = new_freq;
      cudaMemcpy(freq, freqpattern[depth], sizeof(int)*this_size, cudaMemcpyDeviceToHost);
      cudaCheckError();
#else
      freq = freqpattern[depth];
#endif
      for (int p=0; p<this_size; p++) {
            std::cout << p << " : ";
            for (int d=0; d<depth; d++) {
               std::cout << next_descriptions[p*depth+d] <<", ";
            }
            std::cout << " ==> " << freq[p] << std::endl;
      }
      for (int q=0;q<n_trans;q++) {
         for (int p=0;p<this_size;p++) {
            std::cout << (*newdata)(q,p) << "  ";
         }
         std::cout << std::endl;
      }
      std::cout << "Threshold.  last size : " << last_size << " n_items: " << n_items << std::endl;
#endif
#if defined(__CUDA__)
      this_size = threshold_count_gpu(next_descriptions, last_size, n_items, depth, freqpattern[depth], newdata, n_trans, min_support);
      freqpattern[depth]+=last_size*n_items-this_size;
#else
      this_size = threshold_count(next_descriptions, last_size, n_items, depth, freqpattern[depth], newdata, n_trans, min_support);
#endif
      
      cudaCheckError();
#if defined(__DEBUG__)
#if defined(__CUDA__)
      new_freq = (int*) realloc(freq, sizeof(int)*this_size);
      freq = new_freq;
      cudaMemcpy(freq, freqpattern[depth], sizeof(int)*this_size, cudaMemcpyDeviceToHost);
      cudaCheckError();
#else
      freq = freqpattern[depth];
#endif
      for (int p=0; p<this_size; p++) {
            std::cout << p << " : ";
            for (int d=0; d<depth; d++) {
               std::cout << next_descriptions[p*depth+d] <<", ";
            }
            std::cout << " ==> " << freq[p] << std::endl;
      }
      for (int q=0;q<n_trans;q++) {
         for (int p=0;p<last_size*n_items;p++) {
            std::cout << (*newdata)(q,p) << "  ";
         }
         std::cout << std::endl;
      }
#endif

#if defined(__CUDA__)
      new_freq = (int*) realloc(freq, sizeof(int)*this_size);
      freq = new_freq;
      cudaMemcpy(freq, freqpattern[depth], sizeof(int)*this_size, cudaMemcpyDeviceToHost);
      cudaCheckError();
#else
      freq = freqpattern[depth];
#endif
      for (int p=0;p<this_size;p++) {
         //if (freqpattern[depth][p]>=min_support) {
            std::cout<<"{";
            for (int d=0; d<depth; d++) std::cout<< *names[next_descriptions[p*depth+d]] << ", ";
            std::cout<< "},";
            std::cout<<freq[p]<<std::endl;
         //}
      }
#if defined(__CUDA__)
   cudaFree(freqpattern[depth]-last_size*n_items+this_size);
#else
   free(freqpattern[depth]);
#endif
      last_size = this_size;
      
      if (depth==1) {
         n_items = last_size;
         first_descriptions = next_descriptions;
         first_data = newdata;
      } else {
         if (data != first_data) delete(data);
      }
      data = newdata;
      if (last_size < depth) break;
#if defined(__DEBUG__)
      std::cout << "Data: " << std::endl;
      display_flexarray(data);
      std::cout << "First_data: " << std::endl;
      display_flexarray(first_data);
#endif
   }
   free(this_descriptions);
   free(next_descriptions);
#if defined(__CUDA__)
   free(freq);
#endif
   if (first_data != data) delete(first_data);
   delete(data);
   cudaStreamDestroy(myStream);
   cudaStreamDestroy(debugStream);
}
size_t description_outer(int *descript1, const size_t size1, const size_t d1, 
                         int *descript2, const size_t size2, const size_t d2,
                         int *descript_out) {
   for (int q=0; q < size1; q++) {
      for (int p=0; p < size2; p++) {
         for (int dd=0; dd < d1; dd++) {
            descript_out[(d1+d2)*size2*q + (d1+d2)*p + dd] = descript1[q*d1 + dd];
         }
         for (int dd=0; dd < d2; dd++) {
            descript_out[(d1+d2)*size2*q + (d1+d2)*p + d1+ dd] = descript2[p*d2 + dd];
         }
      }
   }
   return size1 * size2;
}
__global__ void two_list_freq_count_kernel(unsigned int *data1, size_t rows, size_t cols1, size_t size1, 
                                           unsigned int *data2, size_t cols2, size_t size2,
                                           int *count, unsigned int *outdata, size_t outsize) {
   size_t ii = threadIdx.x + blockDim.x * blockIdx.x;
   size_t jj = (threadIdx.y + blockDim.y * blockIdx.y) * 32;
   size_t tt0 = (threadIdx.z + blockDim.z * blockIdx.z) * 100;
   int sum[32];
   unsigned int chtmp, chout;
   bool b2;
   if (ii < cols1 && jj < cols2) {
      for(size_t cnt=0;cnt<32;cnt++) {
         sum[cnt]=0;
      }
      for (size_t tt=tt0; tt < tt0 + 100 && tt<rows ; tt++) {
         chout = 0;
         chtmp = __ldg(&data1[(tt*size1 + ii)/32]);
         if ( (chtmp & (unsigned int)(ONEBIT>>(ii%32))) != 0) {
            chtmp = __ldg(&data2[(tt*size2 + jj)/32]);
            chout |= chtmp;
            size_t cnt;
            for (cnt=0; cnt<32; cnt++) {
               b2 = (chtmp & (unsigned int)(ONEBIT>>cnt)) != 0;
               if (b2) {
                  if (jj+cnt < cols2) {
                     sum[cnt]++;
                  } else {
                     //Don't write past the end of the data2 row
                     chout -=  chout & (ONEBIT>>cnt);
                  }
               }
            }
            atomicOr(&outdata[(tt*outsize + ii*cols2 + jj)/32], 
                        chout>>((tt*outsize + ii*cols2 + jj)%32));
         }
         
      }
      for(size_t cnt=0;cnt<32;cnt++) {
         if (jj+cnt < cols2) {
            atomicAdd(&count[cols2*ii + jj + cnt ], sum[cnt]);
         }
      }
   }
   
}
void two_list_freq_count(flexarray_bool *data1/* txi */, 
                         flexarray_bool *data2/* txj */, int *count/* ixj */, 
                         flexarray_bool *outdata/* tx(i*j) */) {
#if defined(__CUDA__) 
   cudaMemset(outdata->data, 0, sizeof(unsigned int)*outdata->real_c*outdata->r/32);
   cudaCheckError();
   two_list_freq_count_kernel<<<dim3((data1->c+31)/32,(data2->c+7)/8, (data1->r+99)/100), 
                                dim3(32,8,1), 0, myStream>>>
                                    (data1->data, data1->r, data1->c, data1->real_c, 
                                     data2->data, data2->c, data2->real_c, count, 
                                     outdata->data, outdata->real_c);
   cudaCheckError();
#else
   for (int ii=0;ii < data1->c; ii++ ) {
      for (int jj=0; jj < data2->c; jj++ ) {
         int sum = 0;
         for (int tt=0; tt < data1->r; tt++) {
            if ((*data1)(tt,ii) && (*data2)(tt,jj)) sum++;
            outdata->set(tt, ii*data2->c + jj, (*data1)(tt,ii) && (*data2)(tt,jj));
         }
         count[data2->c*ii + jj] = sum;
      }
   }
#endif
}
#if 1
// set count to zero for any sets not in ascending order
// since the first d-1 elements of a set are guaranteed to be in order (because we did this step
// last time) we only need to check each element against the last
// also, fills an int* with an ordered set of integers from 0 to length
__global__ void zero_bad_sets(int* count, size_t length, int *range, const int* descriptions, const int d) {
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   if (d > 1 && idx < length*(d-1)) {
      int newi = d-1 + d*(idx/(d-1));
      int ii = idx%(d-1) + d*(idx/(d-1));
      if (descriptions[newi] <= descriptions[ii]) {
         count[idx/(d-1)] = 0;
      }
   }
   if (idx < length) range[idx] = idx;
}
// Rearrange the flexarray_bool data according to "order"
__global__ void rearrange_data(unsigned int *outdata, const unsigned int *data, const size_t cols, const size_t real_c, const size_t rows, int* order ) {
   int ii = (threadIdx.x + blockDim.x * blockIdx.x)*32;
   int tt = threadIdx.y + blockDim.y * blockIdx.y;

   unsigned int iout = 0;
   unsigned int mask = (unsigned int)1<<31;
   unsigned int iin;
   int inidx;
   if (tt < rows && ii<cols) {
   for(int cnt = 0; ii+cnt < cols && cnt < 32; cnt++) {
      inidx = order[ii+cnt];   
      iin = data[(tt*real_c + inidx)/32];
      if ((iin & (ONEBIT>>(inidx%32))) != 0) {
         iout |= mask;
      }
      mask >>=1;
   }
   outdata[(tt*real_c + ii)/32]= iout;
   }
}
// In a set assumed to be ordered, find the first element greater than
// or equal to threshold
__global__ void find_size(int *result, int *count, size_t max, int threshold) {
   int ii = threadIdx.x + blockDim.x * blockIdx.x;
   if (ii==0) {
      if (count[ii] >= threshold) *result = max;
   } else if (ii < max) {
      if(count[ii] >=threshold && count[ii-1] < threshold) {
         *result = max-ii;
      }
   }
}

int threshold_count_gpu(int *descriptions/* (i*j)xd */, const size_t i, const size_t j, const int d, int *count/* j*i */, 
                     flexarray_bool *data/* tx(i*j) */, const size_t t, const int threshold) {

   int *range;
   int *count_buf, *range_buf;
   cudaMalloc(&range, sizeof(int)*i*j);
   int *d_descriptions;
   if (d >1) {
      cudaMalloc(&d_descriptions, sizeof(int)*d*i*j);
      cudaMemcpy(d_descriptions, descriptions, sizeof(int)*d*i*j, cudaMemcpyHostToDevice);
      zero_bad_sets<<<(i*j*(d-1)+127)/128, 128>>>(count, i*j, range, d_descriptions, d);
   } else {
      //just initialize range
      zero_bad_sets<<<(i*j+127)/128, 128>>>(count, i*j, range, d_descriptions, d);
   }
   cudaMalloc(&count_buf, sizeof(int)*i*j);
   cudaMalloc(&range_buf, sizeof(int)*i*j);
   cub::DoubleBuffer<int> d_count(count, count_buf);
   cub::DoubleBuffer<int> d_range(range, range_buf);
   void *d_temp_storage = NULL;
   size_t temp_storage_bytes = 0;
   cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_count, 
                                   d_range, i*j);
   cudaMalloc(&d_temp_storage, temp_storage_bytes);
   cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_count, 
                                   d_range, i*j);
   int new_size = i*j;
   //count_buf is just a place to put the result
   find_size<<<(i*j+127)/128,128>>>(count_buf, count, i*j, threshold);
   cudaMemcpy(&new_size, count_buf, sizeof(int), cudaMemcpyDeviceToHost);
   if (new_size == 0) {
      data->c = 0;
      return 0;
   }
   cub::DoubleBuffer<int> d_range_new(range+(i*j-new_size), range_buf);
   cub::DoubleBuffer<int> d_count_new(count+(i*j-new_size), count_buf);
   cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                   d_range_new, d_count_new, new_size);
   cudaFree(d_temp_storage);
   cudaFree(range_buf);
   cudaFree(count_buf);
   cudaCheckError();

   //copy range data to host
   int *h_range = (int*) malloc(sizeof(int)*new_size);
   cudaMemcpy(h_range, range+(i*j-new_size), sizeof(int)*new_size, cudaMemcpyDeviceToHost);

   //duplicate data on the device
   flexarray_bool *indata = new flexarray_bool(data->r, data->real_c, true);
   cudaMemcpyAsync(indata->data, data->data, sizeof(unsigned int) * data->r * data->real_c/32, cudaMemcpyDeviceToDevice, myStream);
   cudaCheckError();

   //simultaneously process descriptions (on the host) and data (on the device)
   rearrange_data<<<dim3((new_size+31)/32,(t+7)/8), dim3(32,8), 0, myStream>>>(data->data, indata->data, new_size, data->real_c, t, range+(i*j-new_size));
   cudaCheckError();
   for (int p=0; p< new_size; p++) {
      int inidx = h_range[p];
      for (int dd=0; dd<d; dd++) {
         descriptions[d*p+dd] = descriptions[d*inidx+dd];
      }
   }
   free(h_range);
   delete(indata);
   
   cudaFree(range);
   if (d >1) {
      cudaFree(d_descriptions);
   }
   cudaCheckError();
   data->c = new_size;
   return new_size;
}
#endif
int threshold_count(int *descriptions/* (i*j)xd */, const size_t i, const size_t j, const int d, int *count/* j*i */, 
                     flexarray_bool *data/* tx(i*j) */, const size_t t, const int threshold) {
  
   int *h_count;
#if defined(__CUDA__)
   cudaCheckError();
   h_count = (int*)malloc(sizeof(int)*i*j);
   cudaMemcpy(h_count, count, sizeof(int)*i*j, cudaMemcpyDeviceToHost);
   cudaCheckError();
#else
   h_count = count;
#endif
   size_t inx=0; //breaks parallelization
   for (int jj=0; jj < i*j; jj++) {
      if (h_count[jj] >= threshold) {
         bool dup = false;
         for (int dd=0; dd < d; dd++) {
            if (dd < d-1 && descriptions[d*jj + dd] >= descriptions[d*jj+d-1]) dup = true;
            descriptions[d*inx + dd] = descriptions[d*jj + dd];
         }
         if (dup) continue;
         for (int tt=0; tt < data->r; tt++) {
            data->set(tt,inx,(*data)(tt,jj));
            cudaCheckError();
         }
         h_count[inx] = h_count[jj];
         inx++;
      }
   }
   data->c = inx;
#if defined(__CUDA__)
   cudaCheckError();
   cudaMemcpy(count, h_count, sizeof(int)*i*j, cudaMemcpyHostToDevice);
   cudaCheckError();
   free(h_count);
#endif
   return inx;
}  
