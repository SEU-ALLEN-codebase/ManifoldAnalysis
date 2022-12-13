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
python OptimalSubspace.py -i input_feature_table -o outputfolder -v True/False 
```
i: csv format table. 
   row: data points
   column: features selected.

o: target folder you want to put all the results in.

v: visualization of all figures involved in feature screening process. If False, then only final result will be generated.
   default is True.

## Output
1. v==True: ScreeningResult Folder containing all figures related in the screening process of features
2. FeatureSubspace.npy: statistics for each dimension reduction step, e.g feature deletion order; outlier, number of clusters and scores generated; DBSCAN parameters
3. OptimalSubspaceResult.jpg: 3D plot of the final optimal feature subspace
4. scoreplot.png: score changes vs. dimension
5. clusteringlabel.txt: clustering label result for each data point
