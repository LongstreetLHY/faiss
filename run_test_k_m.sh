#！bin/bash
for ((m=4;m<=16;m=m*2)) do
    for ((i=4;i<14;i=i+1)); do
        echo -e "\033[31m Calculate MSE \033[0m"
        ./test_k_m $m $i
    done
done