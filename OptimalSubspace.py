#!/usr/bin/python3
"""
OptimalSubspace: algorithm to find the optimal feature subspace 
Created on Aug 27th, 2022
Last revision: Dec 13th, 2022

@author: Penghao Qian, Sujun Zhao
"""

print(__doc__)

import os,copy
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist,cdist,squareform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics, preprocessing
from multiprocessing import Pool
import argparse


color_dict={0:'#FF0000',2:'#00FF00',1:'#0000FF',3:'#FFFF00',4:'#00FFFF',5:'#FF00FF',-1:'#000000'}


def distanceMatrix(data):
    D=pdist(data,metric='euclidean')
    DistM=squareform(D)
    return DistM


##### PCA transformation
def my_PCA(data):
    X=data.astype(np.float64)
    ### min-max standardization
    min_max_scaler=preprocessing.MinMaxScaler(copy=True)
    scaled_data=min_max_scaler.fit_transform(data)
    ### PCA
    pca=PCA(n_components=data.shape[1])
    newX=pca.fit_transform(scaled_data)
    data_components=pca.components_
    return newX, data_components  


def computeEpsCandidate(data):
    """
    Calculate Eps candidate values
    return: Eps candidate set
    """
    DistMatrix = distanceMatrix(data)
    tmp_matrix = copy.deepcopy(DistMatrix)
    for i in range(len(tmp_matrix)):
        tmp_matrix[i].sort()
    EpsCandidate = []
    for k in range(1,len(data)):
        Dk = tmp_matrix[:,k]
        DkAverage = np.mean(Dk)
        EpsCandidate.append(DkAverage)
    return EpsCandidate


def computeMinptsCandidate(data,DistMatrix,EpsCandidate):
    """
    Calculate MinPts candidate values
    """
    MinptsCandidate = []
    for k in range(len(EpsCandidate)):
        tmp_eps = EpsCandidate[k]
        tmp_count = np.count_nonzero(DistMatrix<=tmp_eps)
        MinptsCandidate.append(tmp_count/len(data))
    return MinptsCandidate


def mutiGetDBSCAN(par):
    """
    Try DBSCAN clustering
    """
    eps,min_samples,num,data=par[0],par[1],par[2],par[3]
    clustering = DBSCAN(eps=eps,min_samples=min_samples).fit(data)
    num_clustering = max(clustering.labels_)
    return [num, num_clustering]

def clusterNumber(data,EpsCandidate,MinPtsCandidate):
    """
    Compute cluster number list with different pairs of parameters
    """
    np_data=np.array(data)
    par_list=[]
    ClusterNumberList=[]
    for i in range(len(EpsCandidate)):
        par_list.append([EpsCandidate[i],MinPtsCandidate[i],i,np_data])
    cpu_worker_num = 18
    with Pool(cpu_worker_num) as p:
        num_clustering_result=p.map(mutiGetDBSCAN, par_list)
    num_clustering_result=np.array(num_clustering_result)    
    
    for i in range(len(EpsCandidate)):
        clustering_result=num_clustering_result[num_clustering_result[:,0]==i,1][0]
        ClusterNumberList.append(clustering_result)
    return ClusterNumberList


##### get best combination of eps and minpts
def optimalDBSCANParameter(data):
    EpsCandidate=computeEpsCandidate(data)
    DistMatrix=distanceMatrix(data)
    MinPtsCandidate=computeMinptsCandidate(data,DistMatrix,EpsCandidate)
    ClusterNumberList=clusterNumber(data,EpsCandidate,MinPtsCandidate)
    for i in range(len(ClusterNumberList)-3):
        if ClusterNumberList[i]==ClusterNumberList[i+1] and ClusterNumberList[i+1]==ClusterNumberList[i+2]:
            if ClusterNumberList[i]!=0:
                count=0
                while(ClusterNumberList[i+count]==ClusterNumberList[i+1+count]):
                    count+=1
                best_eps=EpsCandidate[i+count]
                best_min_samples=MinPtsCandidate[i+count]
                break
            else:
                print("Warning cluster==0")
                best_eps=EpsCandidate[i+2]
                best_min_samples=MinPtsCandidate[i+2]
                break
    return best_eps,best_min_samples
    

def computeStatistics(data, label, method="distance"):
    """
    compute clustering statistisc
    num_outlier: number of outliers defined by DBSCAN
    num_cluster: number of clusters detected except outliers
    sw: average distance score (2^-d) within clusters
    sb: average distance score (2^-d) between clusters
    S: Silhouette Coefficient
    CH: Calinski-Harabasz Index
    DB: Davies-Bouldin Index
    """

    num_outlier=len(np.where(label==-1)[0])
    num_feature=data.shape[1]
    n=data.shape[0]
    num_cluster=len(set(label))-1

    ### distance within
    intra_distance=0
    intra_num=0
    for i in range(num_cluster):
        sub_cluster = data[label==i]
        intra_matrix=pdist(sub_cluster,metric='euclidean')
        intra_matrix = squareform(intra_matrix)    
        if method=="similarity":
            intra_matrix=np.exp2(-intra_matrix)
        intra_distance+=np.sum(intra_matrix)
        intra_num+=len(sub_cluster)*len(sub_cluster)
    sw=intra_distance/intra_num

    ### distance between
    if num_cluster<2:
        sb=np.NaN
    else:
        inter_distance=np.zeros((num_cluster,num_cluster))
        for i in range(0,num_cluster):
            for j in range(i+1,num_cluster):
                sub_cluster1 = data[label==i]
                sub_cluster2 = data[label==j]
                inter_matrix=cdist(sub_cluster1,sub_cluster2,metric='euclidean')
                if method=="similarity":
                    inter_matrix=np.exp2(-inter_matrix)
                inter_distance[i,j]=np.mean(intra_matrix)
        sb=2*np.sum(inter_distance)/(num_cluster-1)/num_cluster
    
    ### other indicators, e.g. Silhouette Coefficient; Calinski-Harabasz Index; Davies-Bouldin Index
    if num_cluster<2:
        S,DB,CH=np.NaN,np.NaN,np.NaN
    else:
        X=data[label!=-1]
        label_d=label[label!=-1]
        S=metrics.silhouette_score(X,label_d)
        DB=metrics.davies_bouldin_score(X,label_d)
        CH=metrics.calinski_harabasz_score(X,label_d)

    return num_outlier, num_cluster, sw, sb, S, CH, DB


def FeatureScreen(dataset,outputFolder=None,visualization=True):
    """
    params: visualization, whether to show the whole screening process in 3D plot
    output:
           1. ScreeningResult: folder containing all figures involved in the screening process
                               subfolders: Dim + number(indicate number of PC left)
           2. FeatureSubspace.npy: contains all statistics
    """

    delete_list=[]
    statistic_list=[]
    parameter_list=[]
    n_feature=np.array(dataset).shape[1]
    origin_data=np.array(dataset,dtype=np.float64)
    pca_data,component=my_PCA(origin_data)
    fList=np.arange(n_feature)

    if visualization==True:
        if outputFolder==None:
            screenFolder=os.path.join(os.getcwd(),'ScreeningResult')
        else:
            screenFolder=os.path.join(outputFolder,'ScreeningResult')
        os.mkdir(screenFolder)
        
    for dim in range(n_feature-3):  
        if visualization ==True:
            ### save 3D visualization plot of each dimension reduction step
            dimFolder='Dim_'+str(n_feature-1-dim)
            if outputFolder==None:
                path=os.path.join(os.getcwd(),'ScreeningResult',dimFolder)
            else:
                path=os.path.join(outputFolder,'ScreeningResult',dimFolder)
            os.mkdir(path)
        
        if len(delete_list)!=0:
            currentFList=np.delete(fList,delete_list)
            current_data=np.delete(pca_data,delete_list,1)
        else:
            currentFList=fList
            current_data=pca_data

        scores=[]
        result=[]
        p=[]
        for col in range(current_data.shape[1]):
            tmp=np.delete(current_data,col,1)
            local_data=tmp[:,0:3]
            best_eps, best_minpts=optimalDBSCANParameter(tmp)
            p.append([best_eps,best_minpts])
            clustering=DBSCAN(eps=best_eps,min_samples=best_minpts).fit(tmp)
            label=clustering.labels_
            num_outlier, num_class, sw, sb, S, CH, DB=computeStatistics(tmp,label,"similarity")
            if num_class!=1:
                scores.append(sw/sb*(1-num_outlier/len(label)))
            else:
                scores.append(-1)
            result.append([num_outlier,num_class,sw,sb,S,DB,CH,len(label)])
            
            ### visualized in 3D (first 3 pc components) PC space
            if visualization==True:
                color_list=[color_dict[x] for x in label]
                plt.close()
                ax=plt.subplot(projection='3d')
                ax.scatter(local_data[:,0],local_data[:,1],local_data[:,2],c=color_list)
                ax.set_title('PC-'+str(currentFList[col]))
                plt.legend(["Outlier:"+str(num_outlier)+" Total Points:"+str(len(label))+" Clusters:"+str(num_class)+"\nsw:"+\
                            str(round(sw,4))+" sb:"+str(round(sb,4))+" Score:"+str(round(sw/sb*(1-num_outlier/len(label)),4))+"\nS:"+\
                            str(round(S,4))+" DB:"+str(round(DB,4))+" CH:"+str(round(CH,4))])
                plt.tight_layout()
                fig_path=os.path.join(path,str(currentFList[col])+'.jpg')
                plt.savefig(fig_path,dpi=300)
        bestCaseId=scores.index(max(scores))
        if scores[bestCaseId]==-1:
            break
        delete_list.append(currentFList[bestCaseId])
        statistic_list.append(result[bestCaseId])
        parameter_list.append(p[bestCaseId])

    if outputFolder==None:
        screenResultPath=os.path.join(os.getcwd(),'FeatureSubspace.npy')
    else:
        screenResultPath=os.path.join(outputFolder,'FeatureSubspace.npy')
    np.save(screenResultPath,np.array([delete_list,statistic_list,parameter_list],dtype=list))


def bestSubspace(data,outputFolder=None):
    """
    Find out the best feature subspace and compute corresponding statistics
    Output:
          1.scoreplot.png: plot showing the score changes during dimension reduction
          2.OptimalSubspaceResult.jpg: data distribution in optimal feature subspace, visualized in 3D plot
          3.return: clustering labels, nummber of outliers, number of classes(exclude outliers), 
                    finalScore, Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index
    """

    if outputFolder==None:
        path=os.path.join(os.getcwd(),'FeatureSubspace.npy')
        scoreFigPath=os.path.join(os.getcwd(),'scoreplot.png')
        fig_path=os.path.join(os.getcwd(),'OptimalSubspaceResult.jpg')
    else:
        path=os.path.join(outputFolder,'FeatureSubspace.npy')
        scoreFigPath=os.path.join(outputFolder,'scoreplot.png')
        fig_path=os.path.join(outputFolder,'OptimalSubspaceResult.jpg')

    res=np.load(path,allow_pickle=True)
    statistic=[x for x in res[1]]
    statistic=np.array(statistic,dtype=float)
    scores=statistic[:,2]/statistic[:,3]*(1-statistic[:,0]/statistic[:,-1])
    scores=scores.tolist()
    max_dim_id=scores.index(max(scores))

    ### score curve: score vs. dim
    X=[np.shape(data)[1]-1-i for i in range(len(statistic))]
    plt.close('all')
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Dim',fontsize=14)
    ax1.set_ylabel('Score', color=color,fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color,labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.invert_xaxis()
    ax1.plot(X,scores, color='tab:red',label='',linewidth=3)
    plt.title('Score plot during screening')
    plt.tight_layout()
    plt.savefig(scoreFigPath, dpi=300)
    
    ### compute information about the best optimal feature subspace
    delete_list=res[0][0:max_dim_id+1]
    delete_list=np.array(delete_list,dtype=int)
    parameterList=res[2]
    bestEps=parameterList[max_dim_id][0]
    bestMinPts=parameterList[max_dim_id][1]
    origin_data=np.array(data,dtype=np.float64)
    pca_data,component=my_PCA(origin_data)
    current_data=np.delete(pca_data,delete_list,1)
    local_data=current_data[:,0:3]
    clustering = DBSCAN(eps=bestEps,min_samples=bestMinPts).fit(current_data)
    label=clustering.labels_
    num_outlier=statistic[max_dim_id,0]
    num_class=statistic[max_dim_id,1]
    finalScore=max(scores)
    S=statistic[max_dim_id,4]
    CH=statistic[max_dim_id,6]
    DB=statistic[max_dim_id,5]

    ### best subspace visualization
    color_list=[color_dict[x] for x in label]
    plt.close()
    ax=plt.subplot(projection='3d')
    ax.scatter(local_data[:,0],local_data[:,1],local_data[:,2],c=color_list)
    ax.set_title('Data Distribution in Optimal Feature Subspace')
    plt.legend(["Outlier:"+str(num_outlier)+" Total Points:"+str(len(label))+" Clusters:"+str(num_class)+\
        "\nScore:"+str(round(finalScore,4))+" S:"+str(round(S,4))+" DB:"+str(round(DB,4))+" CH:"+str(round(CH,4))])
    plt.tight_layout()
    plt.savefig(fig_path,dpi=300)

    return label,num_outlier,num_class,finalScore,S,DB,CH

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--i',help='feature table',type=str)
    parser.add_argument('--o',help='output folder. if none, it will be current folder',type=str)
    parser.add_argument('--v',help='whether need to visualize all figures in the screening process',type=str)

    args=parser.parse_args()
    feature_table=np.loadtxt(args.i,delimiter=',',skiprows=1)

    FeatureScreen(feature_table,args.o,visualization=args.v)
    label,num_outlier,num_class,finalScore,S,DB,CH = bestSubspace(feature_table,args.o)

    if args.o == None:
        outtext=os.path.join(os.getcwd(),"clusteringlabel.txt")
    else:
        outtext=os.path.join(args.o,"clusteringlabel.txt")
    ids=np.arange(len(label))
    result=np.hstack((np.array(label),ids))
    np.savetxt(outtext,result,fmt='%d')
    