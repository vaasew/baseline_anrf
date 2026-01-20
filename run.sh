#!/usr/bin/env bash
#PBS -N anrf_check
#PBS -P m3rg.spons
#PBS -m bea
#PBS -M $USER@scai.iitd.ac.in
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:centos=skylake
#PBS -l walltime=00:30:00
#PBS -q standard
#################  Environment #################################
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

##############################################################

#######################################################################################
source /home/scai/msr/aiy257585/.bashrc
conda activate /home/scai/msr/aiy257585/envs/env1
export PYTHONPATH=$(pwd)
#python -u ./scripts/prepare_dataset.py
#python -u ./scripts/train.py
python -u ./scripts/infer.py
python -u ./scripts/eval.py
