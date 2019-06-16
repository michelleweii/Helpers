#!/usr/bin/env bash
#WORKSPACE=.
WORKSPACE=/Users/admin/Desktop/metric
find $WORKSPACE -name "*.zip" -exec rm -rf {} \;
python3 ${WORKSPACE}/generate_gt.py
python3 ${WORKSPACE}/generate_pred.py
zip -r ${WORKSPACE}/gt.zip ${WORKSPACE}/gt
zip -r ${WORKSPACE}/submit.zip ${WORKSPACE}/submit
#rm -rf ${WORKSPACE}/submit/
#rm -rf ${WORKSPACE}/gt/
#python3 ${WORKSPACE}/script.py –g=$1 –s=$2
python3 ${WORKSPACE}/script.py $1 $2
# python3 script.py -g=gt.zip -s=submit.zip

# bash -x run.sh -g=gt.zip -s=submit.zip