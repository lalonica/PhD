from pandas import Series, DataFrame
import pandas as pd
from itertools import *
import itertools
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.signal import hilbert, chirp
import scipy
import networkx as nx


# Loading the dataset 0750-0805
# Description of the dataset is at: 
# D:/zzzLola/PhD/DataSet/US101/US101_time_series/US-101-Main-Data/vehicle-trajectory-data/trajectory-data-dictionary.htm

c_dataset = ['vID','fID', 'tF', 'Time', 'lX', 'lY', 'gX', 'gY', 'vLen', 'vWid', 'vType','vVel', 'vAcc', 'vLane', 'vPrec', 'vFoll', 'spac','headway' ]

dataset = pd.read_table('D:\\zzzLola\\PhD\\DataSet\\US101\\coding\\dataset_meters_sample.txt', sep=r"\s+", 
                        header=None, names=c_dataset)


#/ Num of different vehicles
numV = dataset['vID'].unique()
print 'The number of unique vehicles are: %i' %len(numV)

#Num of different timestamps
numTS = dataset['Time'].unique()
print 'The number of unique times are: %i' %len(numTS)

# Data set description. The the total number of Frames
# allows to know the time that the cars are in the segment. 
des_all = dataset.describe()
print 'Description of the DataSet'
print des_all

# How many cars change the lane
v_num_lanes = dataset.groupby('vID').vLane.nunique()

# Those do not change the lane
print 'The number of vehicles that do not change their lane are: %i' %v_num_lanes[v_num_lanes == 1].count()

# Those change at least once. 
print 'The number of vehicles that do  change their lane at least once are: %i' %v_num_lanes[v_num_lanes > 1].count()


# Function for saving graphs plots
def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.random_layout(graph) #spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    #cut = 1.00
    #xmax = cut * max(xx for xx, yy in pos.values())
    #ymax = cut * max(yy for xx, yy in pos.values())
    #plt.xlim(0, xmax)
    #plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig



# Compute the distances and the graph for each time. 
times = dataset['Time'].unique()

data = pd.DataFrame()
data = data.fillna(0) # with 0s rather than NaNs

dTime = pd.DataFrame()

for time in times:
    #print 'Time %i ' %time
    
    dataTime0 = dataset.loc[dataset['Time'] == time] 
    
    list_vIDs = dataTime0.vID.tolist()
    #print list_vIDs
    
    dataTime = dataTime0.set_index("vID")
    #index_dataTime = dataTime.index.values
    #print dataTime
    
    perm = list(permutations(list_vIDs,2))
    #print perm
    dist = [((((dataTime.loc[p[0],'gX'] - dataTime.loc[p[1],'gX']))**2) + 
            (((dataTime.loc[p[0],'gY'] - dataTime.loc[p[1],'gY']))**2))**0.5 for p in perm]
    dataDist = pd.DataFrame(dist , index=perm, columns = {'dist'}) 
    

    #Create the fields vID and To
    dataDist['FromTo'] = dataDist.index
    dataDist['From'] = dataDist.FromTo.str[0]
    dataDist['To'] = dataDist.FromTo.str[1]
    #I multiply by 100 in order to scale the number
    dataDist['weight'] = (1/dataDist.dist)*100
    
    #Delete the intermediate FromTo field
    dataDist = dataDist.drop('FromTo', 1)
    

    
    graph = nx.from_pandas_dataframe(dataDist, 'From','To',['weight'])
 

    save_graph(graph,'D:\\zzzLola\\PhD\\DataSet\\US101\\coding\\graphs\\000_my_graph+%i.png' %time)

    
