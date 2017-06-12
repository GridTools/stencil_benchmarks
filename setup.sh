export SLURM_RESERVATION=knl
module switch PrgEnv-cray PrgEnv-intel
module load craype-mic-knl
#command to run
#srun -C <cache,quad|flat,quad> ./openmp_hello
