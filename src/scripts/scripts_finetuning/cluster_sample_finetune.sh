#!/bin/bash
 
#SBATCH --account=hpc_c_giws_spiteri
# #SBATCH --job-name=finetune_all_mlp
# #SBATCH --job-name=fract01_euler1d

# #SBATCH --job-name=fract01_euler1dseeds
# #SBATCH --job-name=fract01_euler1dseeds_32x3
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr4
# #SBATCH --job-name=euler1dseeds_32x3_lr4_300ep
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr3
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr4_5inp
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr4_5inpB
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr4_6inp
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr4_7inp
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr4_8inp
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr34
# #SBATCH --job-name=euler1dseeds_32x3_lr34_100ep
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr34_5inpA
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr34_5inpB
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr34_6inp
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr34_7inp
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr34_8inp
# #SBATCH --job-name=fract01_euler1dseeds_32x3_lr45
# #SBATCH --job-name=fract01_euler05d_seeds_32x3
# #SBATCH --job-name=euler05dseeds_32x3_lr34_100ep
# #SBATCH --job-name=euler05dseeds_32x4_lr34_100ep
# #SBATCH --job-name=euler05dseeds_32x5_lr34_100ep
#SBATCH --job-name=euler05dseeds_32x5_lr34_100ep_carryNO
# #SBATCH --job-name=fract02_euler05d_seeds_32x5_lr34_100ep
# #SBATCH --job-name=fract01_euler05d_seeds_32x3_m0rk4_1d
# #SBATCH --job-name=fract01_euler05d_seeds_32x3_m0rk4_1d_mse
# #SBATCH --job-name=fract01_euler05d_seeds_32x4_m0rk4_1d_mse
# #SBATCH --job-name=fract01_euler05d_seeds_32x4_5inpA
# #SBATCH --job-name=fract01_rk4_1d_seeds_32x3
# #SBATCH --job-name=fract01_rk4_05d_seeds
# #SBATCH --job-name=fract01_rk4_05d_seeds_3lrs
# #SBATCH --job-name=fract01_rk4_05d_seeds_32x3
# #SBATCH --job-name=fract01_rk4_05d_seeds_32x3_3lrs
# #SBATCH --job-name=fract01_euler02d_seeds
# #SBATCH --job-name=fract01_euler02d_seeds_32x3
# #SBATCH --job-name=fract01_euler02d_seeds
# #SBATCH --job-name=fract01_euler02d_seeds_32x3
# #SBATCH --job-name=fract01_euler01d_seeds
# #SBATCH --job-name=fract01_euler01d_seeds_32x3

# #SBATCH --job-name=4inp_euler05d_finetune
# #SBATCH --job-name=5inpA_euler05d_finetune
# #SBATCH --job-name=5inpB_euler05d_finetune
# #SBATCH --job-name=6inp_euler05d_finetune
# #SBATCH --job-name=7inp_euler05d_finetune
# #SBATCH --job-name=8inp_euler05d_finetune




#SBATCH --time=2-00:00:00
# #SBATCH --time=7-00:00:00
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=32
# #SBATCH --cpus-per-task=8
# # # #SBATCH --mem-per-cpu=4G

# #SBATCH --mem=128G
#SBATCH --mem=256G

#SBATCH --output=out-%x.%j.out
#SBATCH --error=out-%x.%j.err
#SBATCH --mail-user=jpcurbelo.ml@gmail.com
#SBATCH --mail-type=ALL
 
module load python/3.10.2

# Path to your virtual environment's activation script
source /scratch/gwf/gwf_cmt/jcurbelo/torchHydroNodes/venv-hydronodes/bin/activate
 
python3 cluster_sample_finetune.py
# python3 cluster_sample_finetune_fract02.py
