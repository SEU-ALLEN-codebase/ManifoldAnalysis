import copy
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist,squareform
from multiprocessing import Pool


def distanceMatrix(data):
    D=pdist(data,metric='euclidean')
    DistM=squareform(D)
    return DistM

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