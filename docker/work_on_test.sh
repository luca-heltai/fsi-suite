#!/bin/bash
newtestrel=$1
newtest="$(cd "$(dirname "$newtestrel")"; pwd)/$(basename "$newtestrel")"
snippetdir=$SNIPPET_DIR
if [ -z "$snippetdir" ];  then
	snippetdir=~/c++/snippets/
fi
echo Using snippet dir: $snippetdir
if [ -f $newtest ]; then 
  ln -sf $newtest $snippetdir/snippet.cc
  echo Linked 
  echo $newtest 
  echo to 
  echo $snippetdir/snippet.cc
  echo 
  extension=""
  for option in "${@:2}"; do
      if [[ $option == *"mpirun"* ]]; then
  	extension=$extension.$option
      else
  	extension=$extension.with_$option=on
      fi
  done
  output=${newtest/\.cc/$extension.output}
  if [ -f $output ]; then
  	echo $output >  $snippetdir/.dealii_output_file
	echo Stored output name 
	echo $output 
	echo in  
	echo $snippetdir/.dealii_output_file
  else
	echo Could not find $output file! Did not store its name anywhere
  fi
else
    echo File 
    echo $newtest 
    echo does not exist. Make sure you specify an existing test.
fi
