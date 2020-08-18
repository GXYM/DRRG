for ((i=490000;i<=510000;i+=1000)); do echo $i;sh eval.sh $1/submit-${i}_1.5|grep Calc;done
