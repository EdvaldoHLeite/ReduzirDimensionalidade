from carregar_bases import obs_network
from Projecao import projetar_bases
import warnings

from PlotarGraficos import info_gain_grafico, fisher_score_grafico, graficos

warnings.filterwarnings("ignore")

def main():
    bases = [obs_network()]
    repeticoes = 2
    nomes_reducao = ["variance_threshold"] #, 'correlation_coefficient']#"fishers_score", "info_gain"]# nomes das reducoes       
    projetar_bases(bases, nomes_reducao, repeticoes)

main()
    























