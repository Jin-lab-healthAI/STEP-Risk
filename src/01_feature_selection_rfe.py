import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import randint,uniform
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

def train4RFEperAE(Xtrain,Ytrain,Xval,Yval,seed,AEname):

    clfs = dict()
    aucs = dict()

    if (AEname=='PLOS') |(AEname=='SLOS'):
        Xtrain = Xtrain.loc[Ytrain[AEname].notna()]
        Ytrain = Ytrain.loc[Ytrain[AEname].notna()]

        Xval = Xval.loc[Yval[AEname].notna()]
        Yval = Yval.loc[Yval[AEname].notna()]
    
    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    rf.fit(Xtrain,Ytrain[AEname])

    Yval_probs = rf.predict_proba(Xval)[:, 1]
    auc = metrics.roc_auc_score(Yval[AEname], Yval_probs)
 
    return rf, auc


def RFE_auc_perAE(seed,AEname,Xtrain,Ytrain,Xval,Yval,df_varname):
    clf,auc = train4RFEperAE(Xtrain,Ytrain,Xval,Yval,seed,AEname)
    df_MDI = pd.Series(clf.feature_importances_,index=clf.feature_names_in_)

    column_to_feature = dict(zip(df_varname['embed_name'], df_varname['var_name']))
    grouped_MDI = df_MDI.groupby(column_to_feature).sum()


    topfeats = [371, 300, 250, 200, 150, 100, 50, 10, 5]
    aucs_all = {}
    clfs_all = {}
    top_n_feats = {}
    for n in topfeats:
        ranking = grouped_MDI.sort_values(ascending=False)
        top_n_feats[n] = ranking.index[:n]
        sel_emb_features = df_varname.loc[df_varname['var_name'].isin(top_n_feats[n]),'embed_name'].tolist()
        
        clf,auc = train4RFEperAE(Xtrain[sel_emb_features],Ytrain,Xval[sel_emb_features],Yval,seed,AEname)
        aucs_all[n] = auc
        clfs_all[n] = clf

        df_MDI = pd.Series(clf.feature_importances_,index=clf.feature_names_in_)
        column_to_feature = dict(zip(df_varname['embed_name'], df_varname['var_name']))
        grouped_MDI = df_MDI.groupby(column_to_feature).sum()
    return pd.Series(aucs_all),clfs_all


savepath_RFE = 'RFE_analysis'
import os
if not os.path.exists(savepath_RFE):
    os.makedirs(savepath_RFE)

Xtrain = pd.read_csv('processed_MLreadydata/internal_Xtrain_train_val.csv',index_col=0)
Ytrain = pd.read_csv('processed_MLreadydata/internal_Ytrain_train_val.csv',index_col=0)
Xval = pd.read_csv('processed_MLreadydata/internal_Xval_train_val.csv',index_col=0)
Yval = pd.read_csv('processed_MLreadydata/internal_Yval_train_val.csv',index_col=0)

# got embedded vectors dictionary to full variables
df_varname = pd.DataFrame(Xtrain.columns)
df_varname.columns = ['embed_name']
df_varname['var_name'] = df_varname['embed_name'].apply(lambda x: x.split('_', 1)[0] )

aucs_col = {}
clfs_col = {}
for AEname in Ytrain.columns:
    aucs_col[AEname] = {}
    clfs_col[AEname] = {}
    print('Processing AE:', AEname)
    for seed in range(1,31):
        print('Seed:', seed)
        aucs_col[AEname][seed], clfs_col[AEname][seed] = RFE_auc_perAE(seed*100,AEname,Xtrain,Ytrain,Xval,Yval,df_varname)
    aucs_col[AEname] = pd.DataFrame(aucs_col[AEname]).T

with open(savepath_RFE+'/aucs_col.pickle', 'wb') as f:
    pickle.dump(aucs_col, f)

with open(savepath_RFE+'/clfs_col.pickle', 'wb') as f:
    pickle.dump(clfs_col, f)

for AEname in Ytrain.columns:

    mean_aucs_seeds = aucs_col[AEname].T
    mean_perf = mean_aucs_seeds.mean(axis=1)  # mean performance across simulations
    std_perf = mean_aucs_seeds.std(axis=1)    # standard deviation
    n = mean_aucs_seeds.shape[1]              # number of simulations

    # 95% Confidence Interval = mean ± 1.96 * (std / sqrt(n))
    ci95 = 1.96 * (std_perf / np.sqrt(n))
    # ci95 = 1.96 * (std_perf)

    # Plot
    plt.subplots(figsize=(3,2.5))
    plt.rcParams['font.size'] = 9
    plt.plot(mean_aucs_seeds.index, mean_perf, label='Mean AUROC',marker='o',color='black')
    plt.fill_between(mean_aucs_seeds.index, mean_perf - ci95, mean_perf + ci95, alpha=0.3, label='95% CI',color='gray')
    plt.xlabel('Top $N$ Features')
    plt.title(AEname)
    plt.ylabel('AUROC')
    # plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath_RFE+'/RFE_analysis_'+AEname+'.jpg', format='jpg', dpi=300)
    
