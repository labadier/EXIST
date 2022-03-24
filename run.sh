#!/bin/bash

# python utils/augmentData.py
# python utils/translation.py -task pivot -sf data/augmented.csv -of training
# python utils/translation.py -task pivot -sf data/EXIST/training/EXIST2021_test.tsv -of test

# python utils/translation.py -task back -sf training
# python utils/translation.py -task back -sf test

python main.py -phase train -lr 3e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l en -tf data/training_backTo_en.csv -mtl mtl
python main.py -phase train -lr 3e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l es -tf data/training_backTo_es.csv -mtl mtl
python main.py -phase train -lr 3e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l de -tf data/training_de.csv -mtl mtl
python main.py -phase train -lr 3e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l fr -tf data/training_fr.csv -mtl mtl


#evaluation with prediction aumentation by paraprhasis (major voting)
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l en -lp all -df data/test_backTo_en.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l es -lp all -df data/test_backTo_es.csv -mtl mtl

#evaluation with prediction individual by language model variation 
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l en -lp en -df data/test_en.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l es -lp es -df data/test_es.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l fr -lp fr -df data/test_fr.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l de -lp de -df data/test_de.csv -mtl mtl
