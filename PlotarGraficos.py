from treino_teste import carregar, criar_pasta
from carregar_bases import *
from sklearn.feature_selection import mutual_info_classif
from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt

def info_gain_grafico(base):
    X = base[0]
    y = base[1]
    nome = base[2]
    colunas = base[3]

    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, colunas)
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind='barh', color='teal')
    plt.title(nome+" - Info Gain")
    plt.show()
    plt.close()

def fisher_score_grafico(base):
    X = base[0]
    y = base[1]
    nome = base[2]
    colunas = base[3]
    
    ranks = fisher_score.fisher_score(X, y)
    feat_importances = pd.Series(ranks, colunas)
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind='barh', color='teal')
    plt.title(nome+" - Fisher`s Score")
    plt.show()
    plt.close()
    
def graficos():
    ################ PCAs #################################################################
    nomes_reducao = ["info_gain", "MCEPCA", "fishers_score"]    
    nomes_reducao = ['correlation_coefficient', "MCEPCA"]
    ############## configuracoes das imagens graficas ######################################
    config = {"PCA":['PCA', 'g', '.'],
              "MCEPCA":["MCEPCA", 'b', '.'],
              "info_gain":["info_gain", 'r', '.'],
              "fishers_score":["fishers_score", 'g', '.'],
              "correlation_coefficient":['correlation_coefficient', 'g', '.']}
    
    classificadores = ['tree','knn', 'gnb', 'lda']
    bases = ['Climate','VColumn', 'Debrecen',
             'WDBC', 'Banknote','Pima',
             'Spambase', 'Occupancy']
    bases = ["Climate", "Banknote", "Debrecen", "Pima"]
    for base in bases:
        carregar("resultados/repeticoes-1",
                 base,
                 classificadores,
                 nomes_reducao,
                 config)

graficos()
        


