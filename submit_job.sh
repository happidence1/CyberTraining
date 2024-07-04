#! /bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=pytorch-gpu       #Set the job name to "pytorch-gpu"
#SBATCH --time=01:00:00              #Set the wall clock limit to 1hr
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=1          #Request 1 task per node
#SBATCH --cpus-per-task=8            #Request 8 cpus per task
#SBATCH --partition=gpu              #Specify partition as GPU if required to run the workload on GPU
#SBATCH --gres=gpu:a30:1             #Specify GPU architecture and number of GPU required
#SBATCH --mem=40G                    #Request 40G per node
#SBATCH --output=pytorch-gpu.%j      #Send stdout/err to "pytorch-gpu.[jobID]"
##SBATCH --reservation=cybertraining

ml purge
ml WebProxy
ml Anaconda3/2022.10
source activate /sw/hprc/sw/Anaconda3/2022.10/envs/cybertraining-env
python main.py
