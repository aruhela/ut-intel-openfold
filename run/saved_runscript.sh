#!/bin/bash
#SBATCH --job-name mlperf-hpc-openfold
#SBATCH --output=slurm-%j.txt
#SBATCH -N  12
#SBATCH --ntasks-per-node 3
#SBATCH -p gpu-a100
#SBATCH --time=12:00:00

# Initialization
SECONDS=0

ml reset
ml gcc/11.2.0 cuda/12.0 nccl cudnn
#module restore sine
module list

source scripts/activate_local_openfold_venv.sh /scratch/05231/aruhela/mlcommons/openfold-venv/
ldd `which python`
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ldd `which python`

which python 
which python3 
python -m torch.utils.collect_env
#python -c 'import torch ; print(torch.__version__)'
#python -c "import torch; print(torch.cuda.is_available())"
#pip list
#conda list
env | grep SLURM

cd /scratch/05231/aruhela/mlcommons/hpc/openfold
outdir=/scratch/05231/aruhela/mlcommons/hpc/openfold/output
datadir=/scratch/05231/aruhela/dataset/openfold


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
export OMP_NUM_THREADS=1

# Set MLPerf variables:
export DATESTAMP=$(date +"%y%m%d%H%M%S%N")
export EXP_ID=1

#ppn=$SLURM_NTASKS_PER_NODE
ppn=3
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

mpiexec -np $SLURM_NNODES -ppn 1 pkill python
mpiexec -np $SLURM_NNODES -ppn 1 pkill torchrun

mpiexec -np $SLURM_NNODES -ppn 1 "./get_memory.sh"


export TACC_IBRUN_DEBUG=1
export I_MPI_DEBUG=4

set -x
#ibrun -n $ranks \

gen_hostfile.sh $ppn
myhostfile="hosts.$SLURM_JOBID"
cat hosts

/usr/bin/time -f "real \t%e (seconds)" \
mpiexec -np $ranks -f $myhostfile -ppn $ppn \
-genv I_MPI_PIN_PROCESSOR_LIST=0,64,96 \
/scratch/05231/aruhela/mlcommons/openfold-venv/conda/envs/openfold-venv/bin/python \
train.py \
--training_dirpath $outdir \
--pdb_mmcif_chains_filepath $datadir/pdb_data/pdb_mmcif/processed/chains.csv \
--pdb_mmcif_dicts_dirpath $datadir/pdb_data/pdb_mmcif/processed/dicts \
--pdb_obsolete_filepath $datadir/pdb_data/pdb_mmcif/processed/obsolete.dat \
--pdb_alignments_dirpath $datadir/pdb_data/open_protein_set/processed/pdb_alignments \
--initialize_parameters_from $datadir/mlperf_hpc_openfold_resumable_checkpoint_b518be46.pt \
--seed $RANDOM \
--num_train_iters 2000 \
--val_every_iters 40 \
--local_batch_size 1 \
--base_lr 1e-3 \
--warmup_lr_init 1e-5 \
--warmup_lr_iters 0 \
--num_train_dataloader_workers 16 \
--num_val_dataloader_workers 2 \
--distributed

set +x

ibrun -n $SLURM_NNODES pkill python

echo -e "`date` : ----- FINISHED $me in $SECONDS Seconds-------"



# torchrun \
# --nnodes $SLURM_NNODES \
# --nproc_per_node "gpu" \
# --rdzv_id $SLURM_JOB_ID \
# --rdzv_backend "c10d" \
# --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
