#!/bin/bash

python utils/augmentData.py
python utils/translation.py -task pivot -sf data/augmented.csv -of training
python utils/translation.py -task pivot -sf data/EXIST/training/EXIST2021_test.tsv -of test

python utils/translation.py -task back -sf train
python utils/translation.py -task back -sf train

python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l en -tf data/training_backTo_en.csv -mtl mtl
python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l es -tf data/training_backTo_en.csv -mtl mtl
python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l de -tf data/training_de.csv -mtl mtl
python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l fr -tf data/training_.csv -mtl mtl

# python main.py -phase eval -wm online -interm_layer 64 -bs 64 -l en -df data/EXIST/training/EXIST2021_test.tsv -mtl mtl