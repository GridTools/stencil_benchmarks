module load perftools-base
module load perftools

export PAT_RT_EXPFILE_NAME=report
export PAT_RT_PERFCTR=PAPI_L2_TCM,PAPI_TLB_DM
export PAT_RT_EXPFILE_REPLACE=1

alias rep='pat_report -T -s aggr_th=sum,sort_by_fu=yes -d P -b fu,th=HIDE report.xf | less'
