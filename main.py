#!/usr/bin/python3
"""
OptimalSubspace: algorithm to find the optimal feature subspace
KANNDBSCAN: an adaptive parameter dbscan algorithm
Created on Aug 27th, 2022
Last revision: Dec 16th, 2022

@author: Penghao Qian, Sujun Zhao
"""
import os,argparse
import numpy as np
import OptimalSubspace as OpSub
if __name__ == '__main__':
    print(__doc__)
    parser=argparse.ArgumentParser()
    parser.add_argument('--i',help='feature table',type=str)
    parser.add_argument('--o',help='output folder. if none, it will be current folder',type=str)
    parser.add_argument('--v',help='whether need to visualize all figures in the screening process',type=str)

    args=parser.parse_args()
    feature_table=np.loadtxt(args.i,delimiter=',',skiprows=1)

    OpSub.FeatureScreen(feature_table,args.o,visualization=args.v)
    label,num_outlier,num_class,finalScore,S,DB,CH = OpSub.bestSubspace(feature_table,args.o)

    if args.o == None:
        outtext=os.path.join(os.getcwd(),"clusteringlabel.txt")
    else:
        outtext=os.path.join(args.o,"clusteringlabel.txt")
    ids=np.arange(len(label))
    result=np.hstack((np.array(label),ids))
    np.savetxt(outtext,result,fmt='%d')