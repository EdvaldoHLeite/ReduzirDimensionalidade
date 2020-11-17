import numpy as np
import pandas as pd
from numpy import linalg as eigen # autovetores e autovalores
from tqdm import tqdm

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2, SelectKBest # chi square
from skfeature.function.similarity_based import fisher_score

def PCA(X): # retorna matriz centralizada e covariancia
    media_matriz = X.mean(axis=0)

    X = X - media_matriz 
    XT = X.T #X.transpose() # transposta de X ou X.T

    covariancia = XT @ X #np.dot(XT, X) # X.T @ X
    covariancia = covariancia/len(X) 
    aut_val, aut_vet = eigen.eig(covariancia) # (np.float(_) for _ in )
    #aut_val, aut_vet = eigen.eig(covariancia.T) # uma matriz e sua transposta tem os mesmos autovetores
    # os autovetores são as colunas
    save = str(aut_val)
    ordenado = np.argsort(aut_val)[::-1]
    save2 = str(aut_val)
    if save != save2:
        print("Opa os autovetores foram modificados durante o PCA, não podem ser usados no MCEPCA")
    
    '''Ed = aut_vet[:, ordenado[:d]]
    aut_vet = Ed
    aut_val = aut_val[ordenado[:d]]'''
    #return Ed, aut_val, aut_vet

    return aut_vet, aut_val

def info_gain(X, y, dataframe_columns):
    # valor mutual entre as features
    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, dataframe_columns) # ganho de informação
    
    # indices features ordenados pelo maior ganho de informação
    ordenado = np.argsort(feat_importances)[::-1]
    return list(ordenado), feat_importances

def fishers_score(X, y, dataframe_columns):
    ranks = fisher_score.fisher_score(X, y)
    feat_importances = pd.Series(ranks, dataframe_columns)
    ordenado = np.argsort(feat_importances[::-1]) # ordenado em ordem decrescente
    return list(ordenado), feat_importances

def pearson_correlation_coefficient(data_frame, threshold):
    correlacao = data_frame.corr()
    a=abs(correlacao['petal_width'])
    result=a[a>threshold]   
    
    return result

def chi_square(X, y, k):
    X_cat = X.astype(int)
    chi2_features = SelectKBest(chi2, k)
    X_kbest_features = chi2_features.fit_transform(X_cat, y)
    
    return X_kbest_features

def MCEPCA(W, X, k, classes, Y, autovalores, autovetores):
    n = len(X)
    d = len(X[0])
         
    # media das features
    W_mean = [[0 for x in range(d)] for y in range(2)]
    for c in range(2): # classes
        for i in range(d): # features
            numerador = 0
            denominador = 0
            for j in range(n): # pontos
                numerador += W[j][i] * int(Y[j] == classes[c])  # wji eh o transposto de wij
                denominador += int(Y[j] == classes[c])
            W_mean[c][i] = numerador/denominador        
    
    # score das features
    score = []
    for i in range(d):
        if autovalores[i] != 0:
            score.append((W_mean[0][i] - W_mean[1][i])**2/autovalores[i])
        else:
            score.append(0)

    ordenado = np.argsort(score)[::-1] # vetor com os indices de score ordenados
    Sk = autovetores[:, ordenado[:k]] # seleciona as k colunas
    return Sk






        
    
