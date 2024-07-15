echo n,k,m,d,code,rank
\grep best best_fpfz_*.txt | \sed -r 's@best_fpfz_n([0-9]+)_k([0-9]+)_m([0-9]+).txt:([A-Za-z]+): best d=([0-9]+), rank=([0-9]+)@\1,\2,\3,\5,\4,\6@g' | sort -n

