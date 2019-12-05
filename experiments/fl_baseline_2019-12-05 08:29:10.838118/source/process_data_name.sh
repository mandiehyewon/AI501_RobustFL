#!/bin/bash
IFILE=/st2/myung/data/diabetic-retinopathy-detection/kaggle/trainLabels.csv
mkdir train_processed
mkdir val_processed
awk -F, '(NR<30000){system("cp train/"$1".jpeg train_processed/"$1"_"$2".jpeg ")} (NR>=30000){system("cp train/"$1".jpeg val_processed/"$1"_"$2".jpeg ")}' $IFILE