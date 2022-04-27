echo copying 
echo ~/c++/snippets/build/output
echo to 
echo `cat /workspace/snippets/.dealii_output_file`
sed  's/^JobId.*//g' ~/c++/snippets/build/output > `cat /workspace/snippets/.dealii_output_file`
