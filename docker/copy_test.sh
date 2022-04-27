oldtestrel=$1
newtestrel=$2
oldtest="$(cd "$(dirname "$oldtestrel")"; pwd)/$(basename "$oldtestrel")"
newtest="$(cd "$(dirname "$newtestrel")"; pwd)/$(basename "$newtestrel")"
if [ -f $oldtest ]; then 
  extension=""
  for option in "${@:3}"; do
      if [[ $option == *"mpirun"* ]]; then
  	extension=$extension.$option
      else
  	extension=$extension.with_$option=on
      fi
  done
  output=${newtest/\.cc/$extension.output}
  echo Created:
  echo $newtest
  echo $output 
  cp $oldtest $newtest
  touch $output
  git add $newtest
  git add $output
  echo $output >  /workspace/snippets/.dealii_output_file
  ln -sf $newtest /workspace/snippets/snippet.cc
else
    echo File 
    echo $oldtest 
    echo does not exist. Make sure you don\'t specify the number and the extension.
fi
