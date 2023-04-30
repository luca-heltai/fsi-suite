#!/bin/bash
snippetdir=$SNIPPET_DIR
if [ -z "$snippetdir" ];  then
        snippetdir=~/c++/snippets/
fi
echo copying 
echo `pwd`/output
echo to 
echo `cat $snippetdir/.dealii_output_file`
sed  's/^JobId.*//g' `pwd`/output > `cat $snippetdir/.dealii_output_file`
