#!/bin/sh
case $2 in
major)
    type=major
    ;;
incompatibilities)
    type=incompatibilities
    ;;
*)
    type=minor
    ;;
esac

filename=../doc/news/changes/$type/`date "+%Y%m%d"`LucaHeltai

echo Writing to $filename
message="$1
<br>
(Luca Heltai, `date "+%Y/%m/%d"`)"

echo "$message" > $filename
git add $filename
