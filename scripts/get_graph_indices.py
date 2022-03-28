## import packages
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import math
from numpy import random
import itertools
from scipy import stats
from scipy.stats import hmean 
import pandas as pd
import csv
import os
import subprocess
%matplotlib inline

## set the file path
cmd = subprocess.Popen('pwd', stdout=subprocess.PIPE)
cmd_out, cmd_err = cmd.communicate()
local_path = os.fsdecode(cmd_out).strip()


## get the region annotations
directory_in_str = local_path+'/data/ML_2020/'
directory = os.fsencode(directory_in_str)
files = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith('region_'):
        files.append(filename)
        
## load the annotated mice brain vasculature data        
mouse_brain_vasc_network = pd.read_pickle(local_path+'/annotated_mouse_brain_vasc_network_2020.pkl')
regional_groups = mouse_brain_vasc_network.groupby(by=['RL1','RL2'])


## get the sizes (number of vessel segments/edges) of different regions based on identification of annotations
sizes = {}
for file in files:
    with open(local_path+'/data/ML_2020/'+file,'r') as f:
        lines = f.readlines()
        
    labels=[[float(y) for y in line.strip().split(' ')] for line in lines]
    
    for label in labels:
        region = regional_groups.get_group((label[0],label[1]))
        n = len(region)
        df = region.fillna(region.dt.describe()['50%']) #fill in the missing values using mean 
        sizes[(label[0],label[1])] = (int(file.split('_')[2]),n,np.mean(1/df.dt))

## helper fuunctions

## compute the effective radius if an edge is a multiedge (more than one vessel segment betweeen two nodes)
def eff_radius(weights,combination):
    resistances = [l/r**2 for r,l in weights]
    if combination == 'parallel':
        eff_resistance = hmean(resistances)/len(resistances)
        eff_length = np.mean([l for r,l in weights])
    else:
        eff_resistance = sum(resistances)
        eff_length = sum([l for r,l in weights])
    return math.sqrt(eff_length/eff_resistance)

## extract a slice-like region that starts at lb along the axis ax and has a thickness of delta
## returns an empty array if there are no vessel segments within the specified region 
def extract_region(lb,delta,ax):
    d = {'X':['VX1','VX2'],'Y':['VY1','VY2'],'Z':['VZ1','VZ2']}     
    indices1 = list(region[(region[d[ax][0]] <= lb+delta) \
                                         & (region[d[ax][0]] >= lb)].index)
    indices2 = list(region[(region[d[ax][1]] <= lb+delta) \
                                         & (region[d[ax][1]] >= lb)].index)
    if indices1 and indices2:
        return np.intersect1d(np.array(indices1),np.array(indices2)).flatten()
    else:
        return np.array([])

## get box-like regions by getting indices of edges that are present in the regions of intersection
## of three slices each oriented in three orthogonal directions 
def extractgraph_indices(initx,initz):
    indicesx = extract_region(initx,500,'X')
    indicesz = extract_region(initz,500,'Z')
    if (indicesx.size>0) and (indicesz.size>0):
        print(f'non-empty-{0}')
        indices = np.intersect1d(indicesx,indicesz).flatten()
        indicesy = extract_region(inity,500,'Y')
        if (indices.size > 0) and (indicesy.size > 0):
            print(f'non-empty-{1}')
            indices = np.intersect1d(indices,indicesy).flatten()
            if indices.size > 0:
                print(f'non-empty-{2}')
                return indices.tolist()
            else:
                return None
        else:
            return None
        
## construct a undirected weighted graph, where the weights are adjusted as per the keyword arguments         
def graph(indices,descending=True,reweigh=False,not_shuffled=True):
    G = nx.MultiGraph()
    for k in indices:
        G.add_edge(int(region['V1'][k]), int(region['V2'][k]),\
                       dt = region['dt'][k], length = region['length'][k])
    
    edge_list = {}
    for edge in list(G.edges.data()):
        e = (edge[0],edge[1])
        if e in edge_list:
            edge_list[e].append((edge[2]['dt'],edge[2]['length']))
        else:
            edge_list[e] = [(edge[2]['dt'],edge[2]['length'])]
     
    x = []
    if descending:
        for edge in edge_list:
            if len(edge_list[(edge[0],edge[1])]) == 1:
                x.append(1/edge_list[(edge[0],edge[1])][0][0])
            else:
                x.append(1/eff_radius(edge_list[(edge[0],edge[1])],'parallel'))
    else:
        for edge in edge_list:
            if len(edge_list[(edge[0],edge[1])]) == 1:
                x.append(edge_list[(edge[0],edge[1])][0][0])
            else:
                x.append(eff_radius(edge_list[(edge[0],edge[1])],'parallel'))
    if reweigh:
        x = get_new_weights(x)
        
    if not not_shuffled:
        y = x.copy()
        np.random.shuffle(x)
        if x == y:
            print(f'number of nodes: {len(x)}')
    
    H = nx.Graph()           
    for k,edge in enumerate(edge_list):
        H.add_edge(edge[0], edge[1], weight = x[k])
        
    return nx.convert_node_labels_to_integers(H,first_label=0)

## get weights after normalizing using a reference distribution
def get_new_weights(w):
    return [np.percentile(reference_dist,stats.percentileofscore(w,x)) for x in w]

## compute persistence i.e. birth-death points for the one-dimensional PH of an undirected weighted 
## graph/network 
def create_simplicial_complex(initx,initz,sons):
    indices = extractgraph_indices(initx,initz)
    if indices:
        g = graph(indices,not_shuffled=sons)
        st = gd.SimplexTree()
        for edge in list(g.edges.data()):
            st.insert([edge[0],edge[1]],filtration=edge[2]['weight'])

        return [st.persistence(), len(g.nodes()), len(g.edges())]
    else:
        return None
    
## main function    
for label in list(sizes.keys()):
    region = pd.DataFrame(regional_groups.get_group((label[0],label[1])))
    region = region.fillna(region.dt.describe()['50%'])
    region.to_csv(local_path+'/data/annotated_PH_2020_500/region_'+str(sizes[label][0])+'_'+str(int(label[0]))+'.csv')
    print(label)
    coordinates_v1 = region[['V1','VX1','VY1','VZ1']]
    coordinates_v2 = region[['V2','VX2','VY2','VZ2']]

    coordinates = {}
    for k in range(coordinates_v1.shape[0]):
        coordinates[coordinates_v1.iloc[k]['V1']] = (coordinates_v1.iloc[k]['VX1'],coordinates_v1.iloc[k]['VY1'],coordinates_v1.iloc[k]['VZ1'])
    for k in range(coordinates_v2.shape[0]):
        if coordinates_v2.iloc[k]['V2'] in coordinates:
            pass
        else:
            coordinates[coordinates_v2.iloc[k]['V2']] = (coordinates_v2.iloc[k]['VX2'],coordinates_v2.iloc[k]['VY2'],coordinates_v2.iloc[k]['VZ2'])

    coordinates = np.array(list(coordinates.values()))

    maxy = np.max(coordinates[:,1])
    miny = np.min(coordinates[:,1])
    maxx = np.max(coordinates[:,0])
    minx = np.min(coordinates[:,0])
    maxz = np.max(coordinates[:,2])
    minz = np.min(coordinates[:,2])
    M = np.median(np.arange(miny,maxy,500))

    for p,q in [(0,maxy),(500,M)]:
        for k in np.arange(p,q,500):
            if p==0:
                inity = np.median(np.arange(miny,maxy,500))+k
            else:
                inity = np.median(np.arange(miny,maxy,500))-k
            print(inity)
            for item in list(itertools.product(np.arange(minx,maxx,500),np.arange(minz,maxz,500))):
                indices = extractgraph_indices(item[0],item[1])
                if indices:
                    n = len(graph(indices).nodes())
                    if n >= 500 and n <= 1500:
                        with open(local_path+'/data/annotated_PH_2020_500/'+\
                                  'PH_'+str(sizes[label][0])+'_'+str(int(label[0]))+\
                                  '/indices_'+str(int(item[0]))+'_'+str(int(inity))+\
                                  '_'+str(int(item[1]))+'.csv', 'w+') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(indices)
