if [ $# -lt 1 ]; then
  echo 1>&2 "$0: not enough arguments"
  exit 2
fi
if [ $# -gt 2 ]; then
  echo 1>&2 "$0: too many arguments"
  exit 2
fi

prefix=small
n=$1
if [ $# -eq 2 ]; then
  prefix=$2
fi

events=mempool_history.events.csv
# txs=mempool_history.txs.csv

head -n 1 ${events} > ${prefix}_${events}
# head -n 1 ${txs}    > ${prefix}_${txs}

tail -n $n ${events} >> ${prefix}_${events}
# cat ${prefix}_${events} | tail -n+2 | cut -d, -f2 | sort -u > small.txs
# grep -E `cat small.txs | xargs | tr ' ' '|'` ${txs} >> ${prefix}_${txs}
# rm small.txs
