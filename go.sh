method=('kmeans' 'gmms' 'ac' )
data=('CSKB' 'OMaha' 'exHITSyn' 'smHITSyn')

# echo $#
if [ $# == 0 ]
then
    for d in ${data[@]}
    do
        for m in ${method[@]}
        do
            python run.py --data=$d --method=$m
            # echo $d+$m
        done
    done
else
    for d in "$@"
    do
        for m in ${method[@]}
        do
            python run.py --data=$d --method=$m
            # echo $d+$m
        done
    done
fi