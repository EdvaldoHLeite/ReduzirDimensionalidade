from carregar_bases import *
from Projecao import projetar_bases
import warnings

from PlotarGraficos import info_gain_grafico, fisher_score_grafico, graficos

warnings.filterwarnings("ignore")

def main():
    bases = [banknote(), climate(), debrecen(),
        pima(), vcolumn(), wdbc(), spambase(),
        occupancy()]
    #bases = [climate()]#, banknote(), debrecen(), pima()]
    
    repeticoes = 100
    nomes_reducao = ['chi2_square']#'correlation_coefficient']#"fishers_score", "info_gain"]# nomes das reducoes       
    projetar_bases(bases, nomes_reducao, repeticoes)
    
    # pearson_correlation_coefficient
    
    '''for base in bases:
        info_gain_grafico(base)
        fisher_score_grafico(base)'''

main()
    
