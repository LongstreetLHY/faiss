#！bin/bash
make test_recall

for ((m=4;m<=16;m=m*2)) do
    for ((i=9;i<14;i=i+1)); do
        echo -e "\033[31m Calculate Recall \033[0m"
        ./test_recall $m $i
    done
done