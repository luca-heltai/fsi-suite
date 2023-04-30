#!/bin/sh
docker pull dealii/dealii:master-jammy
docker build -t heltai/dealii:vscode .
docker push heltai/dealii:vscode
