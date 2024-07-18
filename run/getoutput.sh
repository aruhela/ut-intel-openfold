
cat $1 | grep eval_accuracy | awk '{printf "%2.4f, %6d\n", $11, $18}'
