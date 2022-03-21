#!/bin/bash

python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l en -tf data/back_to_en.csv -mtl mtl
python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l es -tf data/back_to_es.csv -mtl mtl
python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l de -tf data/de.csv -mtl mtl
python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l fr -tf data/fr.csv -mtl mtl

python main.py -phase eval -wm online -interm_layer 64 -bs 64 -l en -df data/EXIST/training/EXIST2021_test.tsv -mtl mtl