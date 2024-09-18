#!/bin/bash
 
#SBATCH --account=hpc_c_giws_spiteri
#SBATCH --job-name=lstm_euler1d_569basins_256b_150ep

#SBATCH --time=7-00:00:00

#SBATCH --ntasks=1  

#SBATCH --cpus-per-task=16
#SBATCH --mem=300G

# #SBATCH --cpus-per-task=1
# #SBATCH --mem=64G

#SBATCH --output=out-%x.%j.out
#SBATCH --error=out-%x.%j.err
#SBATCH --mail-user=jpcurbelo.ml@gmail.com
#SBATCH --mail-type=ALL
 
module load python/3.10.2

# Path to your virtual environment's activation script
source /scratch/gwf/gwf_cmt/jcurbelo/torchHydroNodes/venv-hydronodes/bin/activate
 
python3 run_hybrid_trainer_single_all_lstm.py
