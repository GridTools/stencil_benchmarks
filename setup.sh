export SLURM_RESERVATION=knl
module switch PrgEnv-cray PrgEnv-intel
module load craype-mic-knl
module load cray-memkind
#command to run
#salloc -C <cache,quad|flat,quad> 
#in flat mode:
#srun numactl --membind=1 ./a.out <args>
#in cache mode:
#srun ./a.out <args>
