#!/bin/bash
IFILE=/st2/myung/data/diabetic-retinopathy-detection/kaggle/trainLabels.csv
mkdir train_processed
awk -F, '{system("mv train/"$1".jpeg train_processed/"$1"_"$2".jpeg ")}' $IFILE