#!/bin/bash
#####  Constructed by HPC everywhere #####
######  Module commands #####

module load python/gpu/3.10.5

######  Job commands go below this line #####
python run_30_224_huge_patch14.py 20 30 30 3e-5 0.05 0 0
python run_30_224_huge_patch14.py 30 30 30 3e-5 0.05 0 0
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.05 0 0
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.1 0 0
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.1 0 2
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.1 0 3
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.1 0 4
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.1 0 5
python run_30_224_huge_patch14.py 20 30 30 3e-5 0.15 0 0 
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.15 0 0
python run_30_224_huge_patch14.py 30 30 30 3e-5 0.15 0 0
python run_30_224_huge_patch14.py 20 30 30 3e-5 0.20 0 0
python run_30_224_huge_patch14.py 25 30 30 3e-5 0.20 0 0
python run_30_224_huge_patch14.py 30 30 30 3e-5 0.20 0 0




