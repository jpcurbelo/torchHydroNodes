#!/bin/bash
 
#SBATCH --account=hpc_c_giws_spiteri
# # #SBATCH --job-name=4_basins_7305b_euler1d
# #SBATCH --job-name=bosh3tol46_569_basins_7305b_100ep
# #SBATCH --job-name=euler1d_256b_carryoverYES
# #SBATCH --job-name=euler1d_256b_carryoverNO
# #SBATCH --job-name=euler05d_256b_150ep_569_basins
# #SBATCH --job-name=euler05d256b150ep_B2
#SBATCH --job-name=euler05d512b150ep_B2
# #SBATCH --job-name=rk4_1d_569_basins_7305b_150ep
# #SBATCH --job-name=rk4_05d_569_basins_7305b_150ep
# #SBATCH --job-name=rk4_1d_569_basins_2565b_150ep
# #SBATCH --job-name=euler1d_256b_50ep_lr4
# #SBATCH --job-name=euler1d_256b_50ep_lr34
# #SBATCH --job-name=euler1d_256b_50ep_lr44
# #SBATCH --job-name=euler1d_256b_100ep_lr4
# #SBATCH --job-name=euler1d_256b_100ep_lr45
# #SBATCH --job-name=bosh3tol33_569_basins_256b_150ep
# #SBATCH --job-name=bosh3tol33lr45_569_basins_256b_150ep
# #SBATCH --job-name=rk4_05d_569_basins_2565b_150ep_B2
# #SBATCH --job-name=euler05d_128b_150ep

# #SBATCH --time=1-00:00:00
#SBATCH --time=7-00:00:00

#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=32
# #SBATCH --mem=128G
#SBATCH --mem=256G
# #SBATCH --mem=300G
#SBATCH --output=out-%x.%j.out
#SBATCH --error=out-%x.%j.err
#SBATCH --mail-user=jpcurbelo.ml@gmail.com
#SBATCH --mail-type=ALL
 
module load python/3.10.2

# Path to your virtual environment's activation script
source /scratch/gwf/gwf_cmt/jcurbelo/torchHydroNodes/venv-hydronodes/bin/activate
 
python3 run_hybrid_trainer_single_all_mlp.py
