#!/bin/bash
#SBATCH --job-name mlperf-hpc-openfold
#SBATCH --output=slurm-%j.txt
#SBATCH -N 2
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 48
#SBATCH -p systest-pvc
#SBATCH --time=6:00:00
##SBATCH --reservation=mlcommons

# Initialization
SECONDS=0

cd /scratch/05231/aruhela/ml/hpc/openfold/amit

module reset
module use /scratch/projects/compilers/modulefiles
module list

#source /scratch/projects/compilers/intel24.2/oneapi/intelpython/bin/activate
#conda activate pytorch-gpu
source /scratch/05231/aruhela/share/pip-pytorch-gpu/bin/activate
conda list
#source /scratch/projects/compilers/intel24.2/oneapi/intelpython/envs/pytorch-gpu/lib/python3.9/site-packages/oneccl_bindings_for_pytorch/env/setvars.sh

source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/vars.sh

#ldd `which python`
#export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#ldd `which python`

# export TF_FORCE_UNIFIED_MEMORY=1
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=".10"
# export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
#export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7

which python3 
python -m torch.utils.collect_env
#python -c 'import torch ; print(torch.__version__)'
#python -c "import torch; print(torch.cuda.is_available())"
#pip list
#conda list
env | grep SLURM

datadir=/scratch/05231/aruhela/ml/dataset/openfold
outdir="/scratch/05231/aruhela/ml/hpc/openfold/amit/output-$SLURM_JOB_ID"

# Print current datetime:
echo "started at `date`"
echo "START" $(date +"%Y-%m-%d %H:%M:%S")

# Print node list:
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Print current datetime again:
echo "READY" $(date +"%Y-%m-%d %H:%M:%S")

# Set number of threads to use for parallel regions:
#export OMP_NUM_THREADS=1
unset OMP_NUM_THREADS

# Set MLPerf variables:
export DATESTAMP=$(date +"%y%m%d%H%M%S%N")
export EXP_ID=1

#ppn=$SLURM_NTASKS_PER_NODE
ppn=8
nodes=$SLURM_NNODES
ranks=$((ppn*nodes))
export IBRUN_TASKS_PER_NODE=$ppn
echo "PPN   = $ppn"
echo "IBRUN_TASKS_PER_NODE=$IBRUN_TASKS_PER_NODE"
echo "Ranks = $ranks"
echo "Nodes = $SLURM_NNODES"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
export MY_MPIRUN_OPTIONS="-env MASTER_ADDR=$MASTER_ADDR -env MASTER_PORT=$MASTER_PORT "

RANDOM=$(date +%s)
echo "RANDOM=$RANDOM"

# echo "Kill Python"
mpiexec -np $SLURM_NNODES -ppn 1 pkill python
mpiexec -np $SLURM_NNODES -ppn 1 ./solution.sh

# echo "kill Torchrun"
# mpiexec -np $SLURM_NNODES -ppn 1 pkill torchrun

#sleep 5

#mpiexec -np $SLURM_NNODES -ppn 1 "./get_memory.sh"

export TACC_IBRUN_DEBUG=1
export I_MPI_DEBUG=4

set -x
gen_hostfile.sh $ppn
myhostfile="hosts.$SLURM_JOBID"
cat hosts

/scratch/05231/aruhela/ml/hpc/openfold/amit/intel-smi

/usr/bin/time -f "real \t%e (seconds)" \
mpiexec -np $ranks -ppn $ppn \
-print-rank-map -env I_MPI_DEBUG=4 \
python \
../train.py \
--training_dirpath $outdir \
--pdb_mmcif_chains_filepath $datadir/pdb_data/pdb_mmcif/processed/chains.csv \
--pdb_mmcif_dicts_dirpath $datadir/pdb_data/pdb_mmcif/processed/dicts \
--pdb_obsolete_filepath $datadir/pdb_data/pdb_mmcif/processed/obsolete.dat \
--pdb_alignments_dirpath $datadir/pdb_data/open_protein_set/processed/pdb_alignments \
--initialize_parameters_from $datadir/mlperf_hpc_openfold_resumable_checkpoint_b518be46.pt \
--seed $RANDOM \
--num_train_iters 1 \
--val_every_iters 1 \
--local_batch_size 4 \
--base_lr 1e-3 \
--warmup_lr_init 1e-5 \
--warmup_lr_iters 0 \
--distributed \
--num_train_dataloader_workers 12 \
--num_val_dataloader_workers 8 \
--gradient_accumulation_iters 1 --log_every_iters 1
set +x

mpiexec -np $SLURM_NNODES -ppn 1 pkill python

echo -e "`date` : ----- FINISHED $me in $SECONDS Seconds-------"


#-genv I_MPI_PIN_DOMAIN [0000000000000000FFFFFFFFFFFFFFFF,FFFFFFFFFFFFFFFF0000000000000000,FFFFFFFFFFFFFFFF0000000000000000] \
# torchrun \
# --nnodes $SLURM_NNODES \
# --nproc_per_node "gpu" \
# --rdzv_id $SLURM_JOB_ID \
# --rdzv_backend "c10d" \
# --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
