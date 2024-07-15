num_uniq_headers=`head -q -n 1 results/param.export.0.* | uniq | wc -l`

if [ ! ${num_uniq_headers} -eq 1 ]; then
	echo "error"
	return ${num_uniq_headers}
fi
head -q -n 1 results/param.export.0.* | uniq > iblt_params.csv
tail -n +2 -q results/param.export.0.* >> iblt_params.csv
echo "success"

