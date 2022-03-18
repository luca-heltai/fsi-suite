#!/bin/sh
git stash
git pull
git stash pop
docker pull heltai/dealii:vscode
docker pull heltai/fsi-suite
