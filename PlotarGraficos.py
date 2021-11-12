from treino_teste import carregar, criar_pasta
from carregar_bases import *
from sklearn.feature_selection import mutual_info_classif
#from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def info_gain_grafico(base, repeticoes):
    nome = base[2]
    colunas = base[3]

    # data frame
    feat_importances = pd.Series(data=np.arange(len(colunas)), index=colunas)
    # calculando media
    for i in range(repeticoes):
        data = pd.read_csv("resultados/repeticoes-"+str(repeticoes)+
                            "/"+nome+"/reducoes_resultados/info_gain-"+
                            str(i+1)+".csv")

        data.columns = ['columns', 'values']
        dataSeries = pd.Series(data=np.array(data['values']), index=np.array(data['columns']).astype(str))
        #print(dataSeries.index)
        for coluna in list(colunas):
            #print(data[coluna])
            feat_importances[coluna] += dataSeries[coluna]/repeticoes
    # plotagem
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind='barh', color='teal')
    plt.title(nome+" - Info Gain")
    plt.show()
    plt.close()

def fisher_score_grafico(base, repeticoes):
    nome = base[2]
    colunas = base[3]

    # data frame
    feat_importances = pd.Series(data=np.arange(len(colunas)), index=colunas)
    # calculando media
    for i in range(repeticoes):
        data = pd.read_csv("resultados/repeticoes-"+str(repeticoes)+
                            "/"+nome+"/reducoes_resultados/fisher_score-"+
                            str(i+1)+".csv")

        data.columns = ['columns', 'values']
        dataSeries = pd.Series(data=np.array(data['values']), index=np.array(data['columns']).astype(str))
        #print(dataSeries.index)
        for coluna in list(colunas):
            #print(data[coluna])
            feat_importances[coluna] += dataSeries[coluna]/repeticoes
    # plotagem
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind='barh', color='teal')
    plt.title(nome+" - Fisher`s Score")
    plt.show()
    plt.close()
    
def fisher_info_gain():
    bases = [banknote(), climate(), debrecen(),
        pima(), vcolumn(), wdbc(), spambase(),
        occupancy()]
    #bases=[debrecen()]
    for base in bases:
        fisher_score_grafico(base, 100)
        info_gain_grafico(base, 100)

    
def graficos():
    ################ PCAs #################################################################
    nomes_reducao = ["Forward", "MCEPCA"]    
    #nomes_reducao = ["variance_threshold", "MCEPCA"]
    ############## configuracoes das imagens graficas ######################################
    config = {"PCA":['PCA', 'r', '.'],
              "MCEPCA":["MCEPCA", 'b', '.'],
              "info_gain":["info_gain", 'k', '.'],
              "fishers_score":["fishers_score", 'g', '.'],
              "correlation_coefficient":['correlation_coefficient', 'g', '.'],
              "chi2_square":["chi2_square", 'g', '.'],
              "RFE":["RFE", 'y', '.'],
              "Forward":["Forward", 'g', '.'],
              "variance_threshold":["Variance Threshold: 0.1", 'g', '.'],
              "LASSO":["LASSO", "r", "."]}
    
    classificadores = ['tree','knn', 'gnb', 'lda']
    bases = ["Banknote", "Climate", "Debrecen", 
             "Pima", "VColumn", "WDBC", "Occupancy"]
    for base in bases:
        carregar("resultados/repeticoes-100",
                 base,
                 classificadores,
                 nomes_reducao,
                 config)
#graficos()
#fisher_info_gain()       


