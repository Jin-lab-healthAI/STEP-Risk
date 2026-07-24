import pandas as pd
import numpy as np
import warnings
from code_CTICU.MLfcn import train4RFEperAE,Hyperparams_train_uniqueFeature

warnings.filterwarnings("ignore")

# based on the RFE (Recursive Feature Elimination) approach, 
# identify the subset of features that are most important 
# for each AE using MDI ranking
# It takes around 20 minutes to run.
savepath_postsurgModel = 'postsurgModel_analysis'
import os
if not os.path.exists(savepath_postsurgModel):
    os.makedirs(savepath_postsurgModel)

# load the training and validation data
Xtrain = pd.read_csv('processed_MLreadydata/internal_Xtrain_train_val.csv',index_col=0)
Ytrain = pd.read_csv('processed_MLreadydata/internal_Ytrain_train_val.csv',index_col=0)
Xval = pd.read_csv('processed_MLreadydata/internal_Xval_train_val.csv',index_col=0)
Yval = pd.read_csv('processed_MLreadydata/internal_Yval_train_val.csv',index_col=0)
df_varname = pd.DataFrame(Xtrain.columns)
df_varname.columns = ['embed_name']
df_varname['var_name'] = df_varname['embed_name'].apply(lambda x: x.split('_', 1)[0] )


# number of features to select for each AE using RFE
topnfeatures = {'mortality': np.int64(100),
 'stroke': np.int64(371),
 'renal': np.int64(50),
 'cpvntlng': np.int64(100),
 'reop': np.int64(150),
 'major': np.int64(50),
 'PLOS': np.int64(50),
 'SLOS': np.int64(100)}


# obtain the selected features for each AE
clfs = {}
features_set = {}

for AEname in Ytrain.columns:
    features_set[AEname] = {}
    print(AEname)
    clf = train4RFEperAE(Xtrain,Ytrain,Xval,Yval,seed=42,AEname=AEname)

    df_MDI = pd.Series(clf.feature_importances_,index=clf.feature_names_in_)

    column_to_feature = dict(zip(df_varname['embed_name'], df_varname['var_name']))
    grouped_MDI = df_MDI.groupby(column_to_feature).sum()
    ranking = grouped_MDI.sort_values(ascending=False)
    top_n_feats = ranking.index[:topnfeatures[AEname]]
    sel_emb_features = df_varname.loc[df_varname['var_name'].isin(top_n_feats),'embed_name'].tolist()

    features_set[AEname] = (top_n_feats, sel_emb_features)

import pickle
with open(savepath_postsurgModel+'/features_set.pickle', 'wb') as f:
    pickle.dump(features_set,f)


# Train the models with the selected features
clfname = 'topn_features'
seed = 42
Hyperparams_train_uniqueFeature(Xtrain,Ytrain,Xval,Yval,savepath_postsurgModel,seed,clfname,features_set)


# Evaluate the trained models on internal test, temporal and external datasets
from code_CTICU.MLfcn import extval,exteval_saveprobs
from code_CTICU.MLfcn import val,val_saveprobs

with open(savepath_postsurgModel+'/topn_features.pkl','rb') as file:
    clfs = pickle.load(file)

Xtest = pd.read_csv('processed_MLreadydata/internal_Xtest.csv',index_col=0)
Ytest = pd.read_csv('processed_MLreadydata/internal_Ytest.csv',index_col=0)

Xpros = pd.read_csv('processed_MLreadydata/temporal_X.csv',index_col=0)
Ypros = pd.read_csv('processed_MLreadydata/temporal_Y.csv',index_col=0)

X_bwh = pd.read_csv('processed_MLreadydata/external_bwh_X.csv',index_col=0)
Y_bwh = pd.read_csv('processed_MLreadydata/external_bwh_Y.csv',index_col=0)
Y_bwh = Y_bwh.iloc[:,1:]
filesuffix = 'rf'

val(Xtest,Ytest,clfs,savepath_postsurgModel,clfname)
val_saveprobs(Xtest,Ytest,clfs,savepath_postsurgModel,clfname)

extval(Xpros,Ypros,clfs,savepath_postsurgModel,clfname)
exteval_saveprobs(Xpros,Ypros,clfs,savepath_postsurgModel,clfname)

extval(X_bwh,Y_bwh,clfs,savepath_postsurgModel,clfname+'_BWH')
exteval_saveprobs(X_bwh,Y_bwh,clfs,savepath_postsurgModel,clfname+'_BWH')