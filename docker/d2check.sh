#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage:
$0 regexp

Will check the generated output with the expected output using
cwdiff, and give you the opportunity to accept new results.
"
    exit 0
fi

echo Processing test $1:

srcfile=$(mktemp -t d2check.XXXXX)
dstfile=$(mktemp -t d2check.XXXXX)

ctest --output-on-failure -R $1 | tee >(grep "DIFF failed. ------ Source:" | awk '{ print $NF }' > $srcfile) >(grep "DIFF failed. ------ Result:" | awk '{ print $NF }' > $dstfile)

SRC=$(cat $srcfile)
DST=$(cat $dstfile)
rm $srcfile $dstfile

if [ ! -z "$DST" ]; then
    echo Comparing 
    echo $SRC 
    echo with 
    echo $DST
    cwdiff $SRC $DST

    echo "Accept new version ($FILE)?"
    select yn in "Yes" "No" "Remove"; do
        case $yn in
            Yes ) echo "You said yes. Running again should pass."; cp $DST $SRC; break;;
            No ) echo "You said no. Fix the test and try again."; break;;
            Remove ) echo "Removing generated file."; rm $DST; break;;
        esac
    done
fi