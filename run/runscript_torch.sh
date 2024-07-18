#!/bin/bash
#SBATCH --job-name mlperf-hpc-openfold
#SBATCH --output=slurm-%j.txt
#SBATCH -N 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH -p systest-pvc
#SBATCH --time=6:00:00
##SBATCH --reservation=mlcommons

# Initialization
SECONDS=0

cd /scratch/05231/aruhela/ml/hpc/openfold/amit

module reset
module use /scratch/projects/compilers/modulefiles
source /scratch/05231/aruhela/share/torchgpu/bin/activate
source $MKLROOT/env/vars.sh

export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.1"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0,2,4,6,1,3,5,7

#export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE 

#export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
#export COMPUTE_BACKEND=dpcpp
#export CCL_BLOCKING_WAIT=1
#export TORCH_LLM_ALLREDUCE=0
#export TORCH_LLM_REDUCE=0
#export CCL_SAME_STREAM=1
#export I_MPI_OFFLOAD=1
#export I_MPI_OFFLOAD_TOPOLIB=level_zero
#export CCL_WORKER_COUNT=2
#export CCL_ATL_TRANSPORT=mpi

#export I_MPI_HBW_POLICY=hbw_bind
export I_MPI_HBW_POLICY=hbw_preferred

#export TACC_IBRUN_DEBUG=1
#export I_MPI_DEBUG=4
#export MLIR_ENABLE_DUMP=1
env | grep "TORCH\|CCL\CCL\I_MPI"

which python3 
python -m torch.utils.collect_env
python -c 'import torch ; print(torch.__version__)'
conda list
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

echo "Kill Python"
srun -n $SLURM_NNODES pkill python
echo "kill Torchrun"

set -x
./intel-smi

export TORCH_RUN=1
/usr/bin/time -f "real \t%e (seconds)" \
srun -n $SLURM_NNODES \
torchrun \
--nnodes=$SLURM_NNODES \
--nproc_per_node=$ppn \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
../train.py \
--training_dirpath $outdir \
--pdb_mmcif_chains_filepath $datadir/pdb_data/pdb_mmcif/processed/chains.csv \
--pdb_mmcif_dicts_dirpath $datadir/pdb_data/pdb_mmcif/processed/dicts \
--pdb_obsolete_filepath $datadir/pdb_data/pdb_mmcif/processed/obsolete.dat \
--pdb_alignments_dirpath $datadir/pdb_data/open_protein_set/processed/pdb_alignments \
--initialize_parameters_from $datadir/mlperf_hpc_openfold_resumable_checkpoint_b518be46.pt \
--seed $RANDOM \
--num_train_iters 10 \
--val_every_iters 40  \
--local_batch_size 1 \
--base_lr 1e-3 \
--warmup_lr_init 1e-5 \
--warmup_lr_iters 2 \
--distributed \
--num_train_dataloader_workers 16 \
--num_val_dataloader_workers 4 \
--log_every_iters 1 \
--gradient_accumulation_iters 1 

set +x

#--mpi=none \
#--precision tf32 \

srun -n $SLURM_NNODES pkill python
echo -e "`date` : ----- FINISHED $me in $SECONDS Seconds-------"
