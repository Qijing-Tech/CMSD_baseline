#method=('kmeans' 'gmms' 'ac')
method=('dbscan' 'optics' 'louvain')
data=('CSKB' 'OMaha' 'ex_HIT_syn' 'sm_HIT_syn')
embed=('combined.embed' 'Tencent_combined.embed')
# echo $#
if [ $# == 0 ]
then
    for d in ${data[@]}
    do
        for e in ${embed[@]}
        do
           for m in ${method[@]}
           do
              python run.py --data=$d --method=$m --embed=$e
              # echo $d+$m
           done
        done
    done
else
    for d in "$@"
    do
        for e in ${embed[@]}
        do
           for m in ${method[@]}
           do
              python run.py --data=$d --method=$m --embed=$e
              # echo $d+$m
           done
        done
    done
fi