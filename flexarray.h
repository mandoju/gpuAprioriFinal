#define ONEBIT (unsigned int)2147483648

class flexarray_bool {
   public:
   size_t r;
   size_t c;
   size_t real_c;
   bool device;
   unsigned int* data;
   unsigned int tmp;
   bool retval;
   bool trueval, falseval;
   flexarray_bool(size_t rows, size_t cols, bool gpu=false) {
      r = rows;
      c = cols;
      real_c = 32*((cols+31)/32);
      device = gpu;
      trueval = true;
      falseval = false;
      if (gpu)
         //Make sure to be divisible by 32-bits
         cudaMalloc(&data, sizeof(unsigned int)*rows*real_c/32);
      else
         data = (unsigned int*)malloc(sizeof(unsigned int)*rows*real_c/32);
   }
   bool& operator()(const size_t row, const size_t col) {
      if (device) {
         cudaMemcpy(&tmp, &data[(col + row*real_c)/32], sizeof(unsigned int), cudaMemcpyDeviceToHost);
      } else {
         tmp = data[(col + row*real_c)/32];
      }
      unsigned int foo = tmp & (unsigned int)(ONEBIT>>(col%32));
      retval = (foo != 0);
      return retval;
   }
   void set(const size_t row, const size_t col, bool value) {
      if (device) {
         cudaMemcpy(&tmp, &data[(col + row*real_c)/32], sizeof(unsigned int), cudaMemcpyDeviceToHost);
      } else {
         tmp = data[(col + row*real_c)/32];
      }
      if (value) tmp |= (unsigned int)(ONEBIT >> (col%32));
      else tmp -= tmp & (unsigned int)(ONEBIT >> (col%32));
      if (device) {
         cudaMemcpy(&data[(col + row*real_c)/32], &tmp, sizeof(unsigned int), cudaMemcpyHostToDevice);
      } else {
         data[(col + row*real_c)/32] = tmp;
      }
   }
   ~flexarray_bool() {
      if (device)
         cudaFree(data);
      else
         free(data);
   }
};
void display_flexarray(flexarray_bool* data) {

   for (int q=0;q<data->r;q++) {
      for (int p=0;p<data->c;p++) {
         std::cout << (*data)(q,p) << "  ";
      }
      std::cout << std::endl;
   }
}

void sizeof_flexarray(flexarray_bool* data) {

   int tamanho_total = 0;
   for (int q=0;q<data->r;q++) {
      for (int p=0;p<data->c;p++) {
         tamanho_total += sizeof((*data)(q,p));
      }
   }
   std::cout << "O tamanho total do flex array é :" << tamanho_total ;
}
