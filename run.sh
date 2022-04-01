#!/bin/bash

# python utils/augmentData.py
# python utils/translation.py -task pivot -sf data/augmented.csv -of training
# python utils/translation.py -task pivot -sf data/EXIST/training/EXIST2021_test.tsv -of test

# python utils/translation.py -task back -sf training
# python utils/translation.py -task back -sf test

python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l en -tf data/training_backTo_en.csv -mtl mtl -t 1
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l es -tf data/training_backTo_es.csv -mtl mtl -t 1
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l de -tf data/training_de.csv -mtl mtl -t 1
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l fr -tf data/training_fr.csv -mtl mtl -t 1
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l it -tf data/training_it.csv -mtl mtl -t 1
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l pt -tf data/training_pt.csv -mtl mtl -t 1

#evaluation with prediction aumentation by paraprhasis (major voting)
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l en -lp all -df data/test_backTo_en.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l es -lp all -df data/test_backTo_es.csv -mtl mtl

#evaluation with prediction individual by language model variation 
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l en -lp en -df data/test_en.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l es -lp es -df data/test_es.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l fr -lp fr -df data/test_fr.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l de -lp de -df data/test_de.csv -mtl mtl

python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l pt -lp pt -df data/test_pt.csv -mtl mtl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l it -lp it -df data/test_it.csv -mtl mtl

zip mtl_data.zip logs/*.csv

################## Single task for taks 1 ######################33

python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l en -tf data/training_backTo_en.csv -mtl stl
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l es -tf data/training_backTo_es.csv -mtl stl
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l de -tf data/training_de.csv -mtl stl
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l fr -tf data/training_fr.csv -mtl stl
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l it -tf data/training_it.csv -mtl stl
python main.py -phase train -lr 1e-5 -decay 1e-5 -wm online -interm_layer 64 -epoches 12 -bs 64 -l pt -tf data/training_pt.csv -mtl stl

#evaluation with prediction aumentation by paraprhasis (major voting)
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l en -lp all -df data/test_backTo_en.csv -mtl stl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l es -lp all -df data/test_backTo_es.csv -mtl stl

#evaluation with prediction individual by language model variation 
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l en -lp en -df data/test_en.csv -mtl stl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l es -lp es -df data/test_es.csv -mtl stl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l fr -lp fr -df data/test_fr.csv -mtl stl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l de -lp de -df data/test_de.csv -mtl stl

python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l pt -lp pt -df data/test_pt.csv -mtl stl
python main.py -phase eval -wm online -interm_layer 64 -bs 128 -l it -lp it -df data/test_it.csv -mtl stl


zip stl_t=1_data.zip logs/*.csv