from carregar_bases import obs_network, letter, user_knowledge, mice, wine_quality_red, wine_quality_white, waveform
from Projecao import projetar_bases
import warnings

from PlotarGraficos import info_gain_grafico, fisher_score_grafico, graficos

warnings.filterwarnings("ignore")

def main():
    bases = [
        #obs_network(),
        letter(),
        user_knowledge(),
        mice(),
        wine_quality_red(),
        wine_quality_white(),
        waveform()
    ]
    repeticoes = 1

    ### coeficiente de correlacao nao esta constando nos testes anteriores
    #nomes_reducao = ['RFE']
    nomes_reducao = ['PCA',
                     'chi2_square',
                     'LASSO',
                     'fishers_score',
                     'info_gain',
                     'Forward',
                     'RFE',
                     'variance_threshold'
                     ]
    projetar_bases(bases, nomes_reducao, repeticoes)
main()
    























