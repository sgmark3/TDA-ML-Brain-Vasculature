## import packages
import numpy as np
import gudhi as gd
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import math
from numpy import random
import itertools
from scipy import stats
from scipy.stats import gamma
from scipy.stats import hmean 
import pandas as pd
import csv
import json
import os
import subprocess

## set the file path
cmd = subprocess.Popen('pwd', stdout=subprocess.PIPE)
cmd_out, cmd_err = cmd.communicate()
local_path = os.fsdecode(cmd_out).strip()


## get the region annotations
directory_in_str = local_path + '/data/ML_2018/'
directory = os.fsencode(directory_in_str)
files = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith('region_'):
        files.append(filename)

## load the samples for the experimentation        
samples = []
with open('2018_samples_reference_dist.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append([float(x) for x in row])
        
## load the edge weights        
with open('weights_2018.pickle', 'rb') as handle:
    weights = pickle.load(handle)
keys=list(weights.keys())
        
## load the annotated mice brain vasculature data        
mouse_brain_vasc_network = pd.read_pickle(local_path+'/annotated_mouse_brain_vasc_network_2018.pkl')
regional_groups = mouse_brain_vasc_network.groupby(by=['RL1','RL2'])


## get the sizes (number of vessel segments/edges) of different regions based on identification of annotations
sizes = {}
for file in files:
    with open(local_path+'/data/ML_2018/'+file,'r') as f:
        lines = f.readlines()
        
    labels=[[float(y) for y in line.strip().split(' ')] for line in lines]
    
    for label in labels:
        region = regional_groups.get_group((label[0],label[1]))
        n = len(region)
        df = region.fillna(region.dt.describe()['50%']) #fill in the missing values using mean 
        sizes[(label[0],label[1])] = (int(file.split('_')[2]),n,np.mean(1/df.dt))
        
## can get a region using .get_group
# region1 = regional_groups.get_group(list(sizes.keys())[0])
# region2 = regional_groups.get_group(list(sizes.keys())[-1])

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
    #if indices1 != [] and indices2 != []:
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
def graph(indices,descending=True,reweigh=True,not_shuffled=None):
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
        
    #return H
    return nx.convert_node_labels_to_integers(H,first_label=0)

## get weights after normalizing using a reference distribution
def get_new_weights(w):
    return [np.percentile(reference_dist,stats.percentileofscore(w,x)) for x in w]
    #mean = np.mean(w)
    #return [x/mean for x in w]

## compute persistence i.e. birth-death points for the one-dimensional PH of an undirected weighted 
## graph/network 
def create_simplicial_complex(initx,initz,sons):
    indices = extractgraph_indices(initx,initz)
    if indices:
        g = graph(indices,not_shuffled=sons)
        st = gd.SimplexTree()
        for edge in list(g.edges.data()):
            st.insert([edge[0],edge[1]],filtration=edge[2]['weight'])
            
        filtrations = {}
        for splx in st.get_filtration():
            if splx[1] in filtrations:
                filtrations[splx[1]].append(splx[0])
            else:
                filtrations[splx[1]] = [splx[0]]
        return [st.persistence(), len(g.nodes()), len(g.edges()), filtrations]
    else:
        return None
    
## main function    
# global region,inity
size_reference_dist = round(np.mean([len(val) for key,val in weights.items()]))
for entry in samples[34:35]:
    sample = int(entry[0])
    mu=0
    n=0
    ma=0
    mi=1000
    for x,k,y,z in [[sum(val),len(val),max(val),min(val)] for key,val in weights.items()]:
        mu += x
        n += k
        if y > ma:
            ma = y
        if z < mi:
            mi = z
    mu = mu/n
    smu = np.mean(weights[keys[sample]])-0.7
    ssdev = np.std(weights[keys[sample]],ddof=1)/2
    print(size_reference_dist,ma,mi,mu,smu,ssdev)  
    reference_dist = []
    k = 0
    while k < size_reference_dist:
        x = gamma.rvs(a=1.5,loc=smu,scale=ssdev,size=1)
        ## loc needs to be offset by a number in the interval from 0.3 to 0.6 
        ## scale needs to be divided by an integer [2,3,4,5] -- higher this number steeper the increase in counts
        ## set a to something less than 1, if a is closer to one the distribution has weaker peak at the max value
        ## of the resistance -- [0.1,0.2,..,0.5]
        # gamma.rvs(a=1.0,loc=mu-0.6,scale=0.3,size=1)
        # levy.rvs(loc=mu-0.6,scale=0.09,size=1)
        if x >= mi and x <= ma:
            reference_dist.append(1/x[0])
            k += 1
    pvalues = []
    for key,val in weights.items():
        statistic, p = stats.ks_2samp(np.array([1/x for x in val]),reference_dist)
        pvalues.append(p)
    pval=sum(np.array(pvalues)<0.01)/len(pvalues)
    print(pval)
    
    plt.clf()
    plt.hist([reference_dist,np.array([1/x for x in weights[keys[0]]]),
              np.array([1/x for x in weights[keys[1]]])],
             bins = 20,
             density=True,
             label=['x','y','z'])

    plt.legend(loc='upper left')
    plt.savefig('ref_dist_bef_'+str(int(entry[0]))+str(int(entry[1]))+'.jpg')
    reweights = {}
    k = 0
    for key,val in list(weights.items())[:2]:
        temp = [1/x for x in val]
        reweights[key] = [np.percentile(reference_dist,stats.percentileofscore(temp,x)) for x in temp]
        
    plt.clf()
    plt.hist([reference_dist,np.array([x for x in reweights[keys[0]]]),
              np.array([x for x in reweights[keys[1]]])],
             bins = 20,
             density=True,
             label=['x','y','z'])
    plt.legend(loc='upper left')
    plt.savefig('ref_dist_aft_'+str(int(entry[0]))+str(int(entry[1]))+'.jpg')    
    os.mkdir(local_path+'/Reweighed_PH_'+str(int(entry[0]))+'_'+str(int(entry[1])))
    os.mkdir(local_path+'/Reweighed_PH_'+str(int(entry[0]))+'_'+str(int(entry[1]))+'/PH_2018')

# os.mkdir(local_path+'/data/annotated_PH_2018_500/rescaled_PH_2018/')

    for label in list(sizes.keys()):
        region = pd.DataFrame(regional_groups.get_group((label[0],label[1])))
        region = region.fillna(region.dt.describe()['50%'])
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

        for s,flag in [('',True),('shuffled_',False)]:
            for p,q in [(0,maxy),(500,M)]:
                for k in np.arange(p,q,500):
                    if p==0:
                        inity = np.median(np.arange(miny,maxy,500))+k
                    else:
                        inity = np.median(np.arange(miny,maxy,500))-k
                    print(inity)
                    for item in list(itertools.product(np.arange(minx,maxx,500),np.arange(minz,maxz,500))):
                        results = create_simplicial_complex(item[0],item[1],flag)
                        if results:
                            if results[1] >= 500 and results[1] <= 1500:
                                try:
                                    os.mkdir(local_path+'/Reweighed_PH_'+str(int(entry[0]))+'_'+str(int(entry[1]))+'/PH_2018/PH_'+str(sizes[label][0])+'_'+str(int(label[0])))
                                except OSError as error:
                                    pass

                                with open(local_path+'/Reweighed_PH_'+str(int(entry[0]))+'_'+str(int(entry[1]))+'/PH_2018/PH_'+str(sizes[label][0])+'_'+str(int(label[0]))+'/reweighed_'+s+'ph_'+str(int(item[0]))+'_'+str(int(inity))+'_'+str(int(item[1]))+'.csv', 'w+') as csvfile:
                                    writer = csv.writer(csvfile)
                                    writer.writerow([results[1],results[2]])
                                    for row in results[0]:
                                        writer.writerow(list(row[1]))
                                    
#                             if flag:
#                                 json_object = json.dumps(results[-1])
#                                 with open(local_path + '/data/annotated_PH_2020_500/PH_'+str(sizes[label][0])+'_'+str(int(label[0]))+'/filtration_'+str(int(item[0]))+'_'+str(int(inity))+'_'+str(int(item[1]))+'.json', 'w') as f:
#                                     f.write(json_object)
