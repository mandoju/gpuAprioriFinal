#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include "math.h"
#include "b_plus_tree.cpp"

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

using namespace std;
vector<string> explode( const string &delimiter, const string &explodeme);


int main(int argc, char** argv) {

  int numero_de_colunas = 0;
  int numero_de_items = 0;
  int numero_de_linhas = 0;
  int resultado_do_find_int = 0;
  int i;
  int j;
  int k;
  char delimitador;
  string str;
  string resultado_do_find;
  vector<string> str_vector;
  BPlusTree<string,string,3,3> b_plus_tree;

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

    for(i = 0 ; i < str_vector.size();i++)
    {
      if(! (b_plus_tree.find(str_vector[i]) ) )
      {
          cout << str_vector[i] << endl;
          cout << numero_de_items << endl;
          b_plus_tree.insert(str_vector[i],SSTR(numero_de_linhas));
          numero_de_items++;
      }
    }
    numero_de_linhas++;

  }

  cout << "agora construindo a matrix" << endl;
  fin.clear();
  fin.seekg(0, ios::beg);
  fin >> numero_de_colunas >> delimitador;

  //bool matrix[numero_de_linhas][numero_de_items];

  int** matrix = new int*[numero_de_linhas];
  for(int i = 0; i < numero_de_linhas; ++i) matrix[i] = new int[numero_de_items];

  i = 0;
  while(getline(fin,str))
  {

      str_vector = explode(",",str);
      for(k = 0;k < str_vector.size();k++)
      {
        b_plus_tree.find(str_vector[k],&resultado_do_find);
        resultado_do_find_int = atoi(resultado_do_find.c_str());
        //cout << "K = " << k  << " " << str_vector[k] << " " << resultado_do_find_int << endl;
        for(j = 0; j < numero_de_items;j++)
        {

          if(j == resultado_do_find_int) {
            if(i == 0) cout << j << " entrado" << endl;
            matrix[i][j] = true;
          }
          else matrix[i][j] = false;
        }
      }
      i++;
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
