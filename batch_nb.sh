#!/bin/bash

#from http://stackoverflow.com/questions/17905350/running-an-ipython-notebook-non-interactively
cp Stat.ipynb Sav.ipynb
for mouse in 0216_4 0316_1 0316_3 0316ag_1 1215_1
do
    #batch_animal=$mouse jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True Stat.ipynb --output=${mouse}_report.html
	batch_animal=msa${mouse} jupyter nbconvert Sav.ipynb --to=html --execute --ExecutePreprocessor.timeout=-1 --output=${mouse}_report.html > out_${mouse}.txt 2> out_${mouse}.log & 
	#batch_animal=msa${mouse} jupyter nbconvert Sav.ipynb --to=html --execute --config ExecutePreprocessor.timeout=-1 --output=${mouse}_report.html &
done
wait
