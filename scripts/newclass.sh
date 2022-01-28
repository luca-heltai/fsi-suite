test -d ../include/$(dirname $1) || mkdir -p ../include/$(dirname $1)
test -d ../source/$(dirname $1) || mkdir -p ../source/$(dirname $1)
cp ../source/template.nocc ../source/$1.cc
cp ../include/template.noh ../include/$1.h
ESCAPED_H=$(printf '%s\n' "$1.h" | sed -e 's/[\/&]/\\&/g')
sed -i "s/XXX/$ESCAPED_H/"   ../source/$(basename $1).cc
def=${1/\//\_}
sed -i "s/XXX/$def/"   ../include/$1.h
cmake .
~/.vscode-server/bin/899d46d82c4c95423fb7e10e68eba52050e30ba3/bin/code ../source/$1.cc
~/.vscode-server/bin/899d46d82c4c95423fb7e10e68eba52050e30ba3/bin/code ../include/$1.h
