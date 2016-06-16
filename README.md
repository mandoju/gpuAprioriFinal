GPUApriori
==========

GPU-accelerated implementation of the apriori set mining algorithm

Requirements
============

CUDA Toolkit 3.0

CUB 1.0.2  (available at https://github.com/NVlabs/cub)


Build
=====
1. Edit Makefile to point to your CUB
2. make CUDA=1
3. (to make a very simplistic CPU-only version, use CUDA=0)


Run
===
> apriori input.little


Additional
==========
apriori_data_gen can help generate simple test data

