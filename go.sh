method=('kmeans' 'gmms' 'ac' 'dbscan')
data=('CSKB' 'OMaha' 'ex_HIT_syn' 'sm_HIT_syn')

# echo $#
if [ $# == 0 ]
then
    for m in ${method[@]}   
    do
        for d in ${data[@]}
        do
            python run.py --data=$d --method=$m
            # echo $d+$m
        done
    done
else
    for m in ${method[@]}
    do
        for d in "$@"
        do
            python run.py --data=$d --method=$m
            # echo $d+$m
        done
    done
fi