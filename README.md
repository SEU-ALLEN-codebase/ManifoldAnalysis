# ManifoldAnalysis

This python package is used to find the optimal feature subspace. Currently it has been applied in analysis of single neuron morphologies and neuron type classification.

## pre-request python packages
os
copy
numpy
sklearn
scipy
matplotlib
multiprocessing
argparse

## command line
```
python -i input_feature_table -o outputfolder -v True/False 
```
i: csv format table. 
   row: data points
   column: features selected.

o: target folder you want to put all the results in.

v: visualization of all figures involved in feature screening process. If False, then only final result will be generated.
   default is True.
