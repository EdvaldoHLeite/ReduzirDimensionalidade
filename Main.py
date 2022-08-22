from carregar_bases import obs_network, letter, user_knowledge, mice, wine_quality_red, wine_quality_white, waveform
from carregar_bases import banknote, climate, debrecen, occupancy, pima, vcolumn, wdbc, spambase
from Projecao import projetar_bases
import warnings

from PlotarGraficos import info_gain_grafico, fisher_score_grafico, graficos

warnings.filterwarnings("ignore")

def main():
    bases3 = [
        #obs_network(),
        letter(),
        user_knowledge(),
        mice(),
        wine_quality_red(),
        wine_quality_white(),
        waveform()
    ]
    
    bases2 = [
        banknote(),
        climate(),
        debrecen(),
        occupancy(),
        pima(),
        vcolumn(),
        wdbc(),
        #spambase() Executar depois
    ]
    
    repeticoes = 100

    ### coeficiente de correlacao nao esta constando nos testes anteriores
    #nomes_reducao = ['RFE']
    nomes_reducao = ['MCEPCA',
                    #'PCA',
                     #'chi2_square',
                     #'LASSO',
                     #'fishers_score',
                     #'info_gain',
                     #'Forward',
                     #'RFE',
                     #'variance_threshold'
                     ]
    projetar_bases([*bases2], nomes_reducao, repeticoes)



    '''import numpy as np
    for b in bases2:
        print(str(b[2]) + ": " + str(len(np.unique(b[1])))) # quantidade de y'''
main()
    























