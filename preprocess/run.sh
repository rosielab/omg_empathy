#!/bin/bash
echo "input file name in format Subject_X_Story_X"
read filename
mkdir watson_output
mkdir opensmile_output
mkdir results 
echo "${filename}"  | python watson.py
echo "${filename}"  | python opensmile_valence.py
echo "${filename}"  | python parse.py
