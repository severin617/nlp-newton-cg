#!/bin/sh
if [ -z "$(ls -A checkpoints2)" ]; then
   echo "checkpoints is Empty"
else
   echo "checkpoints is Not Empty, delete all sub folders"
   #rm -r checkpoints2/*
fi
#echo "1st, start to train the Adam pre-trained model"
#python adam_schedule.py 20 96
#echo "2nd, start to train the Adam pre-trained model with longer epochs"
#python adam_schedule.py 2 64
echo "2nd, start to other optimizers with Adam pre-trained model"
python main.py

