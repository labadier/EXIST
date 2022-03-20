#!/bin/bash

python main.py -phase train -lr 1e-5 -decay 2e-5 -wm online -interm_layer 64 -epoches 8 -bs 64 -l en -tf data/back_to_en.csv -mtl mtl