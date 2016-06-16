#include <iostream>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

int main(int argc, char** argv) {

   int n_items = 200;
   int n_trans = 50000;

   float occurance = 0.07;
   float support = 0.01;
   float occurance2 = 0.04;
   float occurance3 = 0.08;
   float occurance4 = 0.013;
   float occurance5 = 0.02;

   bool FerencBodon = false;

   if (argc > 0) {
      if (strcmp(argv[1], "FB") == 0) FerencBodon = true;
   }

   srand(2541617);

   std::cout << (int) (support*n_trans) << std::endl;
   std::cout << n_items << " " << n_trans << std::endl;
   for (int p=0; p < n_items; p++) {
      std::cout << p << std::endl; 
   }

   for (int q=0; q < n_trans; q++) {
      bool special2 = false;
      if (rand() < occurance2*INT_MAX) special2=true;
      bool special3 = false;
      if (rand() < occurance3*INT_MAX) special3=true;
      bool special4 = false;
      if (rand() < occurance4*INT_MAX) special4=true;
      bool special5 = false;
      if (rand() < occurance5*INT_MAX) special5=true;
      for (int p=0; p < n_items; p++) {
         if ( (rand() < occurance*INT_MAX)  ||
                 ( special2 && p==4 || p == 91)  ||
                 ( special3 && p==24 || p == 91)  ||
                 ( special4 && p==7 || p == 78 || p == 129)  ||
                 ( special5 && p==14 || p == 65 || p== 111 || p==157) ) {
            if (FerencBodon) std::cout << p << " ";
            else std::cout << "1 ";
         } else if (!FerencBodon) {
            std::cout << "0 "; 
         }
      }
      std::cout << std::endl;
   }
   return 0;
}
