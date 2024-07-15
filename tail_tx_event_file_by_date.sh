if [ $# -lt 3 ]; then
  echo 1>&2 "$0: not enough arguments"
  exit 2
fi
if [ $# -gt 3 ]; then
  echo 1>&2 "$0: too many arguments"
  exit 2
fi

yyyy=$1
mm=$2
dd=$3
epoch=`date --date="$mm/$dd/$yyyy 00:00:00" +"%s000000"`
echo "epoch: " $epoch


events=mempool_history.events.csv
tevents=mempool_history.$yyyy$mm$dd.events.csv
txdb=tx.list
txs=mempool_history.txs.csv
ttxs=mempool_history.$yyyy$mm$dd.txs.csv

cat $events | awk 'NR==1 { print } NR != 1 && $1 >= '$epoch' { print }' | tee $tevents | tail -n+2 | cut -d, -f2 | sort -u > $txdb

head -n 1 $txs > $ttxs
grep -F -f $txdb $txs >> $ttxs

echo "done"
ls -ltrh $txdb $ttxs $tevents
echo "to remove temporary files:"
echo "rm " $txdb
