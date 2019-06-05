for m in "drcf2"
do
	for d in 10 20 50
	do
		for ml in 5
		do
			for ns in "city"
			do
				qsub script.sh $m $d $ml $ns
			done
		done
	done
done
