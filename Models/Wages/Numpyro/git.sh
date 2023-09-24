#!/bin/bash
cd ~/Documents/Thesis/LabourSIM/
git pull origin master
git add .
git commit -m "$1"
git push origin master
cd Models/Wages/Numpyro/