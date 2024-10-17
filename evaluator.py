
import numpy as np
import torch
from copy import deepcopy


class evaluator():
    def __init__(self,data_class,reco,args,N=[1,5,10,15,20],partition=[20,50]):
        print('*** evaluation phase ***')
        
        self.reco = reco
        self.data = data_class
        self.N = np.array(N)
        self.threshold = 4 # to generate ground truth set
        self.partition = partition
        
        all_items = set(np.arange(1,data_class.num_v + 1))
        tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
        no_items = all_items - tot_items;
        tot_items = torch.tensor(list(tot_items)) - 1
        self.no_item = (torch.tensor(list(no_items)) - 1).long().numpy()
        
        
        
        
        self.__gen_ground_truth_set()
        self.__group_partition()
        
        
    
    def __gen_ground_truth_set(self):
        print('*** ground truth set ***')
        self.GT = dict()
        temp = deepcopy(self.data.test)
        temp = temp[temp['rating']>=self.threshold].values[:,:-1]-1 # contain only user and item
        for j in range(self.data.num_u):
            # temp[:,0] is all user id in test set, who rates >= 4
            # temp[temp[:,0]==j][:,1] to extract item id given user id j
            if len(temp[temp[:,0]==j][:,1])>0:
                self.GT[j] = temp[temp[:,0]==j][:,1] # user id j rates items with score >= 4
            # if len(temp[temp[:,0]==j][:,1]>0) :  self.GT[j] = temp[temp[:,0]==j][:,1]
            # if len(np.setdiff1d(temp[temp[:,0]==j][:,1],self.no_item))>0 :  self.GT[j] = np.setdiff1d(temp[temp[:,0]==j][:,1],self.no_item)
    
    def __group_partition(self):
        print('*** ground partition ***')
        unique_u, counts_u = np.unique(self.data.train['userId'].values-1, return_counts=True)
        self.G = dict()
        # group 1 contains user id who rates < 20 times
        self.G['group1'] = unique_u[np.argwhere(counts_u<self.partition[0])].reshape(-1)
        # find which user in train set rates < 50 times
        temp = unique_u[np.argwhere(counts_u<self.partition[1])]
        # group 2 contains user id who rates between 20 times and 50 times
        self.G['group2'] = np.setdiff1d(temp,self.G['group1'])
        # group 3 contains user id who rates >= 50 times
        self.G['group3'] = np.setdiff1d(unique_u,temp)
        self.G['total'] = unique_u
        # group1 < 20 <= group2 < 50 <= group3
        
    def precision_and_recall(self):
        print('*** precision ***')
        self.p = dict(); self.r = dict(); leng = dict()
        maxn = max(self.N) # 20
        # set zeros array (20,) for each group
        for i in [j for j in self.G]:
            self.p[i] = np.zeros(maxn)
            self.r[i] = np.zeros(maxn)
            leng[i] = 0

        # loop for user id who rates >= 4
        for uid in [j for j in self.GT]:
            leng['total']+=1
            # import IPython; IPython.embed()
            # self.reco[uid][:maxn]: Get highest rated 20 item given uid, check if each is in GT

            hit_ = np.cumsum([1.0 if item in self.GT[uid] else 0.0 for idx, item in enumerate(self.reco[uid][:maxn])])
            # hit_: [0. 0. 0. 0. 0. 1. 1. 2. 2. 2. 2. 2. 2. 2. 2. 3. 3. 4. 4. 4.]
            self.p['total']+=hit_/ np.arange(1,maxn+1)
            self.r['total']+=hit_/len(self.GT[uid])
            leng['total']+=1
            if uid in self.G['group1']:
                self.p['group1']+=hit_/ np.arange(1,maxn+1)
                self.r['group1']+=hit_/len(self.GT[uid])
                leng['group1']+=1
            elif uid in self.G['group2']:
                self.p['group2']+=hit_/ np.arange(1,maxn+1)
                self.r['group2']+=hit_/len(self.GT[uid])
                leng['group2']+=1
            elif uid in self.G['group3']:
                self.p['group3']+=hit_/ np.arange(1,maxn+1)
                self.r['group3']+=hit_/len(self.GT[uid])
                leng['group3']+=1
        for i in [j for j in self.G]:
            self.p[i]/=leng[i]
            self.r[i]/=leng[i]; 

    def normalized_DCG(self):
        print('*** nDCG ***')
        self.nDCG = dict(); leng=dict()
        maxn = max(self.N)
        
        for i in [j for j in self.G]:
            self.nDCG[i] = np.zeros(maxn)
            leng[i] = 0
        for uid in [j for j in self.GT]:
            leng['total']+=1
            idcg_len = min(len(self.GT[uid]),maxn)
            temp_idcg = np.cumsum(1.0 / np.log2(np.arange(2,maxn+2)))
            temp_idcg[idcg_len:]=temp_idcg[idcg_len-1]
            temp_dcg = np.cumsum([1.0/np.log2(idx+2) if item in self.GT[uid] else 0.0 for idx, item in enumerate(self.reco[uid][:maxn])])
            self.nDCG['total']+=temp_dcg / temp_idcg
            if uid in self.G['group1']:
                self.nDCG['group1']+=temp_dcg / temp_idcg
                leng['group1']+=1
            elif uid in self.G['group2']:
                self.nDCG['group2']+=temp_dcg / temp_idcg
                leng['group2']+=1
            elif uid in self.G['group3']:
                self.nDCG['group3']+=temp_dcg / temp_idcg
                leng['group3']+=1
        for i in [j for j in self.G]:
            self.nDCG[i]/=leng[i];
    