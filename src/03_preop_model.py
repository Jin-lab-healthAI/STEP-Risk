import pandas as pd
import numpy as np
import pickle

savepath_preopModel = 'preopModel_analysis'
import os
if not os.path.exists(savepath_preopModel):
    os.makedirs(savepath_preopModel)

feat371 = pd.read_excel('otherdata/Supplement eTable1.xlsx',index_col=0)
feat371 = feat371.iloc[1:,:4]

topnfeatures = {'mortality': np.int64(100),
 'stroke': np.int64(371),
 'renal': np.int64(50),
 'cpvntlng': np.int64(100),
 'reop': np.int64(150),
 'major': np.int64(50),
 'PLOS': np.int64(50),
 'SLOS': np.int64(100)}

with open('postsurgModel_analysis/features_set.pickle', 'rb') as f:
    features_set = pickle.load(f)


# properative correction for mixed variables
def preoperation_correction(df):
    df.loc[df['iabpwhen_Intraop']==1,'iabp_Yes'] = 0
    df.loc[df['iabpwhen_Intraop']==1,'iabp_No'] = 1
    df = df.drop(columns=['iabpwhen_Intraop','intubate_Yes, in the OR for this procedure'])
    return df

Xtrain = pd.read_csv('processed_MLreadydata/internal_Xtrain_train_val.csv',index_col=0)
Xval = pd.read_csv('processed_MLreadydata/internal_Xval_train_val.csv',index_col=0)
Xtrain = preoperation_correction(Xtrain)
Xval = preoperation_correction(Xval)

Ytrain = pd.read_csv('processed_MLreadydata/internal_Ytrain_train_val.csv',index_col=0)
Yval = pd.read_csv('processed_MLreadydata/internal_Yval_train_val.csv',index_col=0)

# got embedded vectors dictionary to full variables
df_varname = pd.DataFrame(Xtrain.columns)
df_varname.columns = ['embed_name']
df_varname['var_name'] = df_varname['embed_name'].apply(lambda x: x.split('_', 1)[0] )



features_preop_set = {}
for AEname in features_set.keys():
    selfeats = feat371[feat371['Short name Version 4.20.2'].isin(features_set[AEname][0].tolist())]
    selfeats = selfeats[selfeats['Type'].isin(['Preoperative','Mixed (Preoperative and Intraoperative)'])]
    sel_feats = df_varname.loc[df_varname['var_name'].isin(selfeats['Short name Version 4.20.2'].tolist()),'embed_name']
    features_preop_set[AEname] = Xtrain.columns.intersection(sel_feats).tolist()


from code_CTICU.MLfcn import Hyperparams_train_uniqueFeature_preop
clfname = 'preop_features'
savepath_preopModel = 'preopModel_analysis'
seed = 42
Hyperparams_train_uniqueFeature_preop(Xtrain,Ytrain,Xval,Yval,savepath_preopModel,seed,clfname,features_preop_set)



from code_CTICU.MLfcn import extval,exteval_saveprobs
from code_CTICU.MLfcn import val,val_saveprobs

import pandas as pd
import pickle
with open(savepath_preopModel+'/preop_features.pkl','rb') as file:
    clfs = pickle.load(file)


Xtest = pd.read_csv('processed_MLreadydata/internal_Xtest.csv',index_col=0)
Xtest = preoperation_correction(Xtest)
Ytest = pd.read_csv('processed_MLreadydata/internal_Ytest.csv',index_col=0)

Xpros = pd.read_csv('processed_MLreadydata/temporal_X.csv',index_col=0)
Xpros = preoperation_correction(Xpros)
Ypros = pd.read_csv('processed_MLreadydata/temporal_Y.csv',index_col=0)

X_bwh = pd.read_csv('processed_MLreadydata/external_bwh_X.csv',index_col=0)
X_bwh = preoperation_correction(X_bwh)
Y_bwh = pd.read_csv('processed_MLreadydata/external_bwh_Y.csv',index_col=0)
Y_bwh = Y_bwh.iloc[:,1:]

filesuffix = 'rf'

val(Xtest,Ytest,clfs,savepath_preopModel,clfname)
val_saveprobs(Xtest,Ytest,clfs,savepath_preopModel,clfname)

extval(Xpros,Ypros,clfs,savepath_preopModel,clfname)
exteval_saveprobs(Xpros,Ypros,clfs,savepath_preopModel,clfname)

extval(X_bwh,Y_bwh,clfs,savepath_preopModel,clfname+'_BWH')
exteval_saveprobs(X_bwh,Y_bwh,clfs,savepath_preopModel,clfname+'_BWH')