#!/bin/bash
 
#SBATCH --account=hpc_c_giws_spiteri
#SBATCH --job-name=4_basins_256b_euler1d



#SBATCH --time=0-08:00:00
# #SBATCH --time=7-00:00:00

#SBATCH --ntasks=1       

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# #SBATCH --cpus-per-task=32
# #SBATCH --mem=128G
# #SBATCH --mem=256G
# #SBATCH --mem=300G
#SBATCH --output=out-%x.%j.out
#SBATCH --error=out-%x.%j.err
#SBATCH --mail-user=jpcurbelo.ml@gmail.com
#SBATCH --mail-type=ALL
 
module load python/3.10.2

# Path to your virtual environment's activation script
source /scratch/gwf/gwf_cmt/jcurbelo/torchHydroNodes/venv-hydronodes/bin/activate
 
python3 run_hybrid_trainer_single_all_mlp.py
