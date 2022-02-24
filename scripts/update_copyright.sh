#!/bin/sh

for i in `find . -name "*.cc" -o -name "*.h" -o -name "*.dox"`;
do
  if ! grep -q "Copyright (C)" $i
  then
    cat ./doc/copyright.txt $i >$i.new && mv $i.new $i
  fi
done