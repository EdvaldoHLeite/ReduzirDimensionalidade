
""" Recebe os reducao, e as bases, para fazer a projecao """

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.model_selection import train_test_split

from treino_teste import treino_teste, salvar
from FeatureSelection import *

import os
from treino_teste import criar_pasta

# classificadores
knn_reducao = knn()
gnb_reducao = gnb()
tree_reducao = tree()
lda_reducao = lda()

def projetar_bases(bases, nomes_reducao, numero_repeticoes):
    
    # para cada base
    for indice in range(len(bases)):
        
        # dados da base5
        X = bases[indice][0]
        y = bases[indice][1]
        nome_base = bases[indice][2]
        nomes_colunas = bases[indice][3]
        
        # divisão das n iteracoes        
        df_treino_indices = {}
        df_teste_indices = {}
        pasta_indices = "bases/"+nome_base+"/dividir_treino_teste"
        criar_pasta(pasta_indices)
        if os.path.isfile(pasta_indices+"/treino-"+str(numero_repeticoes)+".csv") and os.path.isfile(pasta_indices+"/teste-"+str(numero_repeticoes)+".csv"):
            df_treino_indices = pd.read_csv(pasta_indices+"/treino-"+str(numero_repeticoes)+".csv")
            df_teste_indices = pd.read_csv(pasta_indices+"/teste-"+str(numero_repeticoes)+".csv")
        else:
            X_indices = np.arange(len(X))
            for iteracao in range(numero_repeticoes):
                treino_x, teste_x, treino_y, teste_y = train_test_split(X_indices, y, test_size=0.5, stratify=y)
                df_treino_indices[str(iteracao+1)] = treino_x
                df_teste_indices[str(iteracao+1)] = teste_x
            
            df_treino_indices = pd.DataFrame(df_treino_indices)
            df_teste_indices = pd.DataFrame(df_teste_indices) 
            df_treino_indices.to_csv(pasta_indices+"/treino-"+str(numero_repeticoes)+".csv")
            df_teste_indices.to_csv(pasta_indices+"/teste-"+str(numero_repeticoes)+".csv")
        
        
        print("BASE: ",nome_base)
        ## cataloga todas as classes do conjunto de dados
        classes = [] # classes encontradas
        # adiciona classes que nao foram adicionadas, que estejam apenas no teste
        for classe in y:
            if classe not in classes:
                classes.append(classe)

        maximo = len(X.T) # numero maximo de features
        
        # para cada tecnica de redução de dimensionalidade
        for nome_reducao in nomes_reducao: # percorre as chaves
            print("reducao: ",nome_reducao)

            # listas das medias para cada iteracao
            mediasTree = np.zeros(maximo)
            mediasNB = np.zeros(maximo)
            mediasKNN = np.zeros(maximo)
            mediasLD = np.zeros(maximo)
            
            # médias de todas as iteracoes pra cada base
            # cada feature tera uma lista de medias, para depois se obter o desvio padrão
            mediasIteracoesTree = []
            mediasIteracoesNB = []
            mediasIteracoesKNN = []
            mediasIteracoesLD = []
            
            pasta_salvar_pca = "bases/"+nome_base+"/autvetores_PCA_iteracoes/" 
            pasta_reducoes_resultados = "bases/"+nome_base+"/reducoes_resultados/"
            criar_pasta(pasta_salvar_pca) # cria a pasta onde ficara os pca de cada iteracao
            criar_pasta(pasta_reducoes_resultados)
            for iteracao in range(numero_repeticoes):
                treino_indices = df_treino_indices[str(iteracao+1)]
                teste_indices = df_teste_indices[str(iteracao+1)]
                treino_x, teste_x = X[treino_indices], X[teste_indices]                
                treino_y, teste_y = y[treino_indices], y[teste_indices]
                
                autovetores = None
                autovalores = None
                treino_reduzido_x = None
                teste_reduzido_x = None
                
                ######### PCA ###########
                pasta_iteracao_autvet = pasta_salvar_pca + "autvet-" + str(iteracao + 1)+".csv"
                pasta_iteracao_autval = pasta_salvar_pca + "autval-" + str(iteracao + 1)+".csv"
                if (not os.path.isfile(pasta_iteracao_autvet)): # se o pca não foi calculado
                    autovetores, autovalores = PCA(treino_x) # aplicacao do reducao normal
                    
                    np.savetxt(pasta_iteracao_autvet, autovetores, delimiter=',')
                    np.savetxt(pasta_iteracao_autval, autovalores, delimiter=',')
                else: # se existem os dados sao carregados
                    autovetores = np.loadtxt(pasta_iteracao_autvet, delimiter=',')
                    autovalores = np.loadtxt(pasta_iteracao_autval, delimiter=',')
                projecao_treino_x = treino_x @ autovetores
                projecao_teste_x = teste_x @ autovetores
                ##########################
                
                                    
                ########## Demais Tipos de redução de dimensionalidade ########
                if "info_gain" in nome_reducao:
                    ordenado, feature_importances = info_gain(projecao_treino_x, treino_y, nomes_colunas)
                    feature_importances.to_csv(pasta_reducoes_resultados+"info_gain-"+str(iteracao+1)+".csv")
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:, ordenado]
                elif "fishers_score" in nome_reducao:
                    # indices das features ordenadas em ordem decrescente de acordo com o fisher score
                    ordenado, feature_importances = fishers_score(projecao_treino_x, treino_y, nomes_colunas)
                    feature_importances.to_csv(pasta_reducoes_resultados+"fisher_score-"+str(iteracao+1)+".csv")
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:, ordenado]
                elif "MCEPCA" in nome_reducao:
                    Sk = MCEPCA(projecao_treino_x, treino_x, maximo, classes, treino_y, autovalores, autovetores)
                    treino_reduzido_x = treino_x @ Sk
                    teste_reduzido_x = teste_x @ Sk
                elif "chi_square" in nome_reducao:
                    #X_chi_square = chi_square(projecao_treino_x, y, )
                    # descobrir como ordenar matriz completa usando o chi square, ter acesso a esses dados 
                    # e usalos para comparar as features
                    # 
                #elif "correlation_coefficient" in nome_reducao:
                    
                ###############################################################
                                
                ############### Classificacao dos resultados ###############################
                resultado_tree = treino_teste(tree_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)
                resultado_gnb = treino_teste(gnb_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)
                resultado_knn = treino_teste(knn_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)
                resultado_lda = treino_teste(lda_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)

                ### adicionando resultados de cada feature
                mediasTree += resultado_tree
                mediasNB += resultado_gnb
                mediasKNN += resultado_knn
                mediasLD += resultado_lda

                ### matrizes usadas para desvio padrao
                mediasIteracoesTree.append(resultado_tree)
                mediasIteracoesNB.append(resultado_gnb)
                mediasIteracoesKNN.append(resultado_knn)
                mediasIteracoesLD.append(resultado_lda)

            # os vetores com as somas dos resultados em cada feature sao divididos para obter a media
            mediasTree = mediasTree/numero_repeticoes
            mediasNB = mediasNB/numero_repeticoes
            mediasKNN = mediasKNN/numero_repeticoes
            mediasLD = mediasLD/numero_repeticoes

            # desvio padrão

            desvioTree = []
            desvioNB=[]
            desvioKNN=[]
            desvioLD=[]
            # estraindo o desvio padrao
            for f in range(maximo):
                desvioTree.append(pd.Series(mediasTree).std())
                desvioNB.append(pd.Series(mediasNB).std())
                desvioKNN.append(pd.Series(mediasKNN).std())
                desvioLD.append(pd.Series(mediasLD).std())
        
            # salva os resultados em arquivos .txt
            salvar(mediasTree, mediasKNN, mediasNB, mediasLD, # resultados dos classificadores
                   ["resultados", "repeticoes-"+str(numero_repeticoes), nome_base], # nomes das pastas
                   nome_reducao, maximo) # nome do reducao

            # salva os desvios padroes em arquivos txt
            salvar(desvioTree, desvioKNN, desvioNB, desvioLD, # desvios padroes dos resultados anteriores
                   ["resultados", "repeticoes-"+str(numero_repeticoes), nome_base], # nomes das pastas
                   nome_reducao+"_desvio_padrao", maximo)

        
