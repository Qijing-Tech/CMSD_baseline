#method=('kmeans' 'gmms' 'ac')
method=('dbscan' 'optics' 'louvain' 'l2c')
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
              if [ $m == 'l2c' ];then
                 # if use gpu, set gpuid
                 python run_l2c.py --gpuid 0 --data=$d --method=$m --embed=$e --model_type='l2c' --model_name='L2C_v0'
                 #only-cpu
                 #python run_l2c.py --data=$d --method=$m --embed=$e --model_type='l2c' --model_name='L2C_v0'
              else
                python run.py --data=$d --method=$m --embed=$e
              fi
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
              if [ $m == 'l2c' ];then
                # if use gpu, set gpuid
                python run_l2c.py --gpuid 0 --data=$d --method=$m --embed=$e --model_type='l2c' --model_name='L2C_v0'
                #only-cpu
                #python run_l2c.py --data=$d --method=$m --embed=$e --model_type='l2c' --model_name='L2C_v0'
              else
                python run.py --data=$d --method=$m --embed=$e
              fi
           done
        done
    done
fi