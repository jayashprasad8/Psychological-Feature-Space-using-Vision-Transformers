#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH -A r00329
#SBATCH --mail-user=janandan@iu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-0:0:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=rocks_360_384
#SBATCH --output=jupyter-log-%J.txt

######  Module commands #####

module load python/gpu/3.10.5

######  Job commands go below this line #####
python run_30_384_patch_32.py 25 30 30 3e-5 0.01 0 0
python run_30_384_patch_32.py 25 30 30 3e-5 0.01 0 2
python run_30_384_patch_32.py 25 30 30 3e-5 0.01 0 3
python run_30_384_patch_32.py 25 30 30 3e-5 0.01 0 4
python run_30_384_patch_32.py 25 30 30 3e-5 0.01 0 5
python run_30_384_patch_32.py 25 120 120 3e-5 0.01 0 0
python run_30_384_patch_32.py 25 120 120 3e-5 0.01 0 2
python run_30_384_patch_32.py 25 120 120 3e-5 0.01 0 3
python run_30_384_patch_32.py 25 120 120 3e-5 0.01 0 4
python run_30_384_patch_32.py 25 120 120 3e-5 0.01 0 5
python run_30_384_patch_32.py 25 60 60 3e-5 0.01 0 0
python run_30_384_patch_32.py 25 60 60 3e-5 0.01 0 2
python run_30_384_patch_32.py 25 60 60 3e-5 0.01 0 3
python run_30_384_patch_32.py 25 60 60 3e-5 0.01 0 4
python run_30_384_patch_32.py 25 60 60 3e-5 0.01 0 5
python run_30_384_patch_32.py 25 60 60 3e-5 0.01 10 0
python run_30_384_patch_32.py 25 60 60 3e-5 0.01 20 0
python run_30_384_patch_32.py 25 30 30 1e-5 0.01 0 0 
python run_30_384_patch_32.py 25 30 30 1e-4 0.01 0 0 
python run_30_384_patch_32.py 25 30 30 1e-3 0.01 0 0 
python run_30_384_patch_32.py 25 30 30 1e-2 0.01 0 0 
python run_30_384_patch_32.py 25 30 30 3e-5 0.05 0 0 
python run_30_384_patch_32.py 25 30 30 1e-4 0.05 0 0 
python run_30_384_patch_32.py 25 30 30 1e-3 0.05 0 0
python run_30_384_patch_32.py 25 120 120 1e-3 0.05 0 0
python run_30_384_patch_32.py 25 60 60 1e-3 0.05 0 0 
python run_30_384_patch_32.py 25 30 30 1e-2 0.05 0 0 

python run_30_384_patch16_large.py 25 30 30 3e-5 0.01 0 0
python run_30_384_patch16_large.py 25 30 30 3e-5 0.01 0 2
python run_30_384_patch16_large.py 25 30 30 3e-5 0.01 0 3
python run_30_384_patch16_large.py 25 30 30 3e-5 0.01 0 4
python run_30_384_patch16_large.py 25 30 30 3e-5 0.01 0 5
python run_30_384_patch16_large.py 25 120 120 3e-5 0.01 0 0
python run_30_384_patch16_large.py 25 120 120 3e-5 0.01 0 2
python run_30_384_patch16_large.py 25 120 120 3e-5 0.01 0 3
python run_30_384_patch16_large.py 25 120 120 3e-5 0.01 0 4
python run_30_384_patch16_large.py 25 120 120 3e-5 0.01 0 5
python run_30_384_patch16_large.py 25 60 60 3e-5 0.01 0 0
python run_30_384_patch16_large.py 25 60 60 3e-5 0.01 0 2
python run_30_384_patch16_large.py 25 60 60 3e-5 0.01 0 3
python run_30_384_patch16_large.py 25 60 60 3e-5 0.01 0 4
python run_30_384_patch16_large.py 25 60 60 3e-5 0.01 0 5
python run_30_384_patch16_large.py 25 60 60 3e-5 0.01 10 0
python run_30_384_patch16_large.py 25 60 60 3e-5 0.01 20 0
python run_30_384_patch16_large.py 25 30 30 1e-5 0.01 0 0 
python run_30_384_patch16_large.py 25 30 30 1e-4 0.01 0 0 
python run_30_384_patch16_large.py 25 30 30 1e-3 0.01 0 0 
python run_30_384_patch16_large.py 25 30 30 1e-2 0.01 0 0 
python run_30_384_patch16_large.py 25 30 30 3e-5 0.05 0 0 
python run_30_384_patch16_large.py 25 30 30 1e-4 0.05 0 0 
python run_30_384_patch16_large.py 25 30 30 1e-3 0.05 0 0
python run_30_384_patch16_large.py 25 120 120 1e-3 0.05 0 0
python run_30_384_patch16_large.py 25 60 60 1e-3 0.05 0 0 
python run_30_384_patch16_large.py 25 30 30 1e-2 0.05 0 0 

python run_30_384_patch32_large.py 25 30 30 3e-5 0.01 0 0
python run_30_384_patch32_large.py 25 30 30 3e-5 0.01 0 2
python run_30_384_patch32_large.py 25 30 30 3e-5 0.01 0 3
python run_30_384_patch32_large.py 25 30 30 3e-5 0.01 0 4
python run_30_384_patch32_large.py 25 30 30 3e-5 0.01 0 5
python run_30_384_patch32_large.py 25 120 120 3e-5 0.01 0 0
python run_30_384_patch32_large.py 25 120 120 3e-5 0.01 0 2
python run_30_384_patch32_large.py 25 120 120 3e-5 0.01 0 3
python run_30_384_patch32_large.py 25 120 120 3e-5 0.01 0 4
python run_30_384_patch32_large.py 25 120 120 3e-5 0.01 0 5
python run_30_384_patch32_large.py 25 60 60 3e-5 0.01 0 0
python run_30_384_patch32_large.py 25 60 60 3e-5 0.01 0 2
python run_30_384_patch32_large.py 25 60 60 3e-5 0.01 0 3
python run_30_384_patch32_large.py 25 60 60 3e-5 0.01 0 4
python run_30_384_patch32_large.py 25 60 60 3e-5 0.01 0 5
python run_30_384_patch32_large.py 25 60 60 3e-5 0.01 10 0
python run_30_384_patch32_large.py 25 60 60 3e-5 0.01 20 0
python run_30_384_patch32_large.py 25 30 30 1e-5 0.01 0 0 
python run_30_384_patch32_large.py 25 30 30 1e-4 0.01 0 0 
python run_30_384_patch32_large.py 25 30 30 1e-3 0.01 0 0 
python run_30_384_patch32_large.py 25 30 30 1e-2 0.01 0 0 
python run_30_384_patch32_large.py 25 30 30 3e-5 0.05 0 0 
python run_30_384_patch32_large.py 25 30 30 1e-4 0.05 0 0 
python run_30_384_patch32_large.py 25 30 30 1e-3 0.05 0 0
python run_30_384_patch32_large.py 25 120 120 1e-3 0.05 0 0
python run_30_384_patch32_large.py 25 60 60 1e-3 0.05 0 0 
python run_30_384_patch32_large.py 25 30 30 1e-2 0.05 0 0 


