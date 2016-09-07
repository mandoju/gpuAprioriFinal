#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include "math.h"
#include "b_plus_tree.cpp"

using namespace std;
vector<string> explode( const string &delimiter, const string &explodeme);


int main(int argc, char** argv) {

  int numero_de_colunas;
  char delimitador;
  string str;
  vector<string> str_vector;
  BPlusTree<string,string,3,3,3,3,3> b_plus_tree;

  if (argc < 2) {
     std::cout << "Usage: " << std::endl << "     lb_apriori <input file name>" << std::endl;
     return 0;
  }
  std::cout << "Reading input data..." << std::endl;
  std::ifstream fin(argv[1], std::ifstream::in);
  fin >> numero_de_colunas >> delimitador;

  cout << "O numero de colunas é " << numero_de_colunas << std::endl;
  cout << "O delimitador é " << delimitador << std:: endl;

  while(getline(fin,str))
  {
    str_vector = explode(",",str);
    if(! (b_plus_tree.find(str_vector[0],str_vector[0]) ) )
    {
      b_plus_tree.insert(&str_vector[0],str_vector[0]);
    }

    cout << str_vector[0] << endl;
  }



}


vector<string> explode( const string &delimiter, const string &str)
{
    vector<string> arr;

    int strleng = str.length();
    int delleng = delimiter.length();
    if (delleng==0)
        return arr;//no change

    int i=0;
    int k=0;
    while( i<strleng )
    {
        int j=0;
        while (i+j<strleng && j<delleng && str[i+j]==delimiter[j])
            j++;
        if (j==delleng)//found delimiter
        {
            arr.push_back(  str.substr(k, i-k) );
            i+=delleng;
            k=i;
        }
        else
        {
            i++;
        }
    }
    arr.push_back(  str.substr(k, i-k) );
    return arr;
}
