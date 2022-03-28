## basic packages and libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import persim
from numpy import random
import seaborn as sns
from persim.persistent_entropy import *
import gudhi.representations
import gudhi as gd
import csv
import os
import subprocess
sns.set_style("darkgrid")

## ML packages and libraries
from sklearn.svm import LinearSVC
from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

## set the local path
cmd = subprocess.Popen('pwd', stdout=subprocess.PIPE)
cmd_out, cmd_err = cmd.communicate()
local_path = os.fsdecode(cmd_out).strip()

## load the samples for the experimentation        
samples = []
with open(local_path+'/data/2020_samples_reference_dist.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append([float(x) for x in row])
        
"""
Returns persistence data
Inputs -- path: local_path
       -- starts_with: the string that the filenames start with if needed
       -- ends_with: the string that the filenames end with if needed
       -- threshold: number of edges threshold of the underlying vascular networks
"""

def get_data(path=None,starts_with=None,ends_with=None,threshold=0):
    files = [os.fsdecode(file) for file in os.listdir(path)]
    persistence_data={}
    for filename in files:
        if starts_with and ends_with:
            if filename.startswith(starts_with) and filename.endswith(ends_with):
                data = []
                with open(path+filename) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        data.append([float(x) for x in row])
                if data[0][1] >= threshold:
                    persistence_data[filename]=[item for item in data[1:] if item[1]!=np.inf]
        elif starts_with:
            if filename.startswith(starts_with):
                data = []
                with open(path+filename) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        data.append([float(x) for x in row])
                if data[0][1] >= threshold:
                    persistence_data[filename]=[item for item in data[1:] if item[1]!=np.inf]
        elif ends_with:
            if filename.endswith(ends_with):
                data = []
                with open(path+filename) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        data.append([float(x) for x in row])
                if data[0][1] >= threshold:
                    persistence_data[filename]=[item for item in data[1:] if item[1]!=np.inf]

    return persistence_data

"""
Returns average persistence lanscape
Input -- persistence_data: a dictionary of folders consisting of dictionaries 
         of files whose values are the lists of persistence (b,d) pairs dataset
"""

def get_pl(persistence_data):
    pl_data = []
    for key,value in persistence_data.items():
        if value:
            for subkey,subvalue in value.items():
                pl_data.append(subvalue)
                if subvalue == []:
                    print(subkey)
    return pl_data

## produce cross-validation plots and compare
for j,entry in enumerate(samples[62:]):
    path = local_path+'/normalized_weights_PH_2020/Reweighed_PH_'+str(int(entry[0]))+'_'+str(int(entry[1]))+'/PH_2020/'
    # +'/data/annotated_PH_2018_500/rescaled_PH_2018/'
    # '/data/annotated_PH_2020_500/'
    directory_in_str = path
    directory = os.fsencode(directory_in_str)

    folders_region1 = []

    for folder in os.listdir(directory):
        foldername = os.fsdecode(folder)
        if foldername.startswith('PH_640_'):
            folders_region1.append(foldername)
            foldername = path+foldername

    folders_region2 = []

    for folder in os.listdir(directory):
        foldername = os.fsdecode(folder)
        if foldername.startswith('PH_3_'):
            folders_region2.append(foldername)
            foldername = path+foldername

    for s in ['','shuffled_']:
        r1_PHs={}
        for folder in folders_region1:
            r1_PHs[folder] = get_data(path+folder+'/',starts_with='reweighed_'+s+'ph_',ends_with='.csv')

        r2_PHs={}
        for folder in folders_region2:
            r2_PHs[folder] = get_data(path+folder+'/',starts_with='reweighed_'+s+'ph_',ends_with='.csv')

        r1_ph_dataset = get_pl(r1_PHs)
        r2_ph_dataset = get_pl(r2_PHs)

        ## Downsampling
#         random.seed(j)
#         indices = np.random.randint(0,len(r2_ph_dataset),len(r1_ph_dataset))
#         r2_ph_dataset = [r2_ph_dataset[k] for k in indices]

        y=[]
        for k in range(len(r1_ph_dataset)+len(r2_ph_dataset)):
            if k < len(r1_ph_dataset):
                y.append(0)
            elif k >= len(r1_ph_dataset):
                y.append(1)

        dataset = [np.array(item) for item in r1_ph_dataset+r2_ph_dataset]
        print(sum(y)/len(y),len(y))

#         ph_train, ph_test, y_train, y_test = train_test_split(dataset, y,
#                                                           test_size = 0.1,
#                                                           random_state=j,
#                                                           shuffle=True,
#                                                           stratify=y)

        #     resolution = 10

        #     pipe = Pipeline(steps=[
        #         #('separator',gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
        #         ('scaler',gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
        #         ('tda',gd.representations.Landscape()),
        #         ('classifier',LinearSVC(max_iter=1000000))
        #     ])

        #     param_grid = {
        #         'scaler__use':[True],
        #         'tda__resolution':[resolution],
        #         'tda__num_landscapes':[10],
        #         'classifier__C':np.logspace(-2,3,10) 
        #     }

        #         if flag:
        #             model = GridSearchCV(pipe, param_grid, n_jobs=12, cv = 10, scoring='f1')
        #             model.fit(ph_train,y_train)
        #             print(model.best_score_,model.best_params_)

        final_model = Pipeline(steps=[
            #('separator',gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
            ('scaler',gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
            ('tda',gd.representations.Landscape(resolution=500,num_landscapes=5)),
            ('classifier',LinearSVC(max_iter=2000000))
        ])

        param_range = np.logspace(-2, 3, 10)

        train_scores, test_scores = validation_curve(
            final_model,
            dataset,
            y,
            param_name="classifier__C",
            param_range=param_range,
            scoring="f1",
            n_jobs=12,
            cv=10
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        print(np.mean(train_scores_mean),np.mean(test_scores_mean))

        plt.clf()
        plt.figure(figsize=(15,10))
        plt.title('Validation Curve with LinearSVC: Normalized'+s+' samples (BS) vs. Normalized'+s+' samples (CH)',fontsize=20)
        plt.xlabel("C: regularization parameter",fontsize=16)
        plt.ylabel("f1-score",fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(
            param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
        )
        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2,
            color="darkorange",
            lw=lw,
        )
        plt.semilogx(
            param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
        )
        plt.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.2,
            color="navy",
            lw=lw,
        )
        plt.legend(loc="upper left")
        #         plt.savefig(local_path+'/plots_2018_640_reweighed/balanced_dataset/2018_640_rescaled_CV.jpg')
        plt.savefig(local_path+'/hr_imbalanced_dataset_ml_anlys/2020_plots_640_3/imbalanced_2020_'+str(int(entry[0]))+'_'+str(int(entry[1]))+'_reweighed_'+s+'CV.jpg')

        with open(local_path+'/hr_imbalanced_dataset_ml_anlys/imbalanced_2020_640_3_reweighed_'+s+'eval_metrics.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow(test_scores_mean)
            writer.writerow(test_scores_std)
            writer.writerow(train_scores_mean)
            writer.writerow(train_scores_std)

#         from sklearn.manifold import TSNE
#         embedding = TSNE(2)

#         viz_pipe = Pipeline(steps=[
#             #('separator',gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
#             ('scaler',gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
#             ('tda',gd.representations.Landscape(resolution=500,num_landscapes=5)),
#             ('embedding',embedding)
#         ])

#         X_tsne = viz_pipe.fit_transform(dataset)
#         y = np.array(y)

#         plt.clf()
#         plt.figure(figsize=(16,10))

#         for i in range(2):
#             plt.scatter(X_tsne[y==i,0], X_tsne[y==i,1], label=i)

#         plt.legend(fontsize=14)
#         plt.title('TSNE embedding: Normalized'+s+' samples (BS) vs. Normalized'+s+' samples (CH)', fontsize=20)
#         plt.legend(['Normalized'+s+' samples (BS)','Normalized'+s+' samples (CH)'])
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
#         # plt.savefig(local_path+'/plots_2018_640_reweighed/balanced_dataset/2018_640_rescaled_tsne.jpg')
#         plt.savefig(local_path+'/hr_balanced_dataset_ml_anlys/2020_plots_640_3/balanced_2020_'+str(int(entry[0]))+'_'+str(int(entry[1]))+'_reweighed_tsne.jpg')

