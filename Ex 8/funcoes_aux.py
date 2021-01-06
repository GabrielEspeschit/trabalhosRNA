import numpy as np
from numpy import matmul
from numpy.linalg import inv, det
from sklearn.cluster import KMeans as kmeans

# Função para selecionar os pontos da RBF randomicamente
def seleciona_centros_rand(pontos, p):
    
    #assert (p<=N/n), "P menor que o número de amostras/número de dimensões"
    centros = []
    
    pnts_rand = pontos[(np.random.choice(pontos.shape[0], size=(2*p), replace=False)), : ]
    pnts_rand = np.split(pnts_rand, p)
    for pnt in pnts_rand:
        centros.append((pnt[0]+pnt[1])/2)
        #raios.append(np.linalg.norm(pontos[0]-pontos[1]))
    labels = []
    for ponto in pontos:
        aux = np.inf
        aux_label = -1
        for i, centro in enumerate(centros):
            if np.linalg.norm(centro-ponto) < aux:
                aux = np.linalg.norm(centro-ponto)
                aux_label = i
        labels.append(aux_label)
        class saida:
            def __init__ (self, centros, labels):
                self.cluster_centers_ = np.array(centros)
                self.labels_ = np.array(labels)
            
    return saida(centros, labels)


#### Função para treinar RBF
def treinaRBF(xin, yin, p, metodo = 'kmeans'):
    
    ######### Função Radial Gaussiana #########
    #Definindo função para calcular a PDF
    def pdf_mv(x, m, K, n):
        if n == 1:
            r = np.sqrt(K)
            px = ((1/(np.sqrt(2*np.pi*r*r)))*np.exp(-0.5 * ((x-m)/(r))**2))
        else:
            parte1 = 1/(((2* np.pi)**(n)*(det(K))))
            parte2 = -0.5 * matmul(matmul((x-m).T, (inv(K))), (x-m))
            px = parte1*np.exp(parte2)
        return(px)

    ##################################################

    N = xin.shape[0] # Número de amostras
    n = xin.shape[1] # Dimensão de entrada

    if metodo == 'kmeans':
        xclust = kmeans(n_clusters=p).fit(xin) # Fazendo o Clustering com a função kmeans do sklearn
       
    elif metodo == 'rand':
        xclust = seleciona_centros_rand(xin, p)
    
    else:
        print(f'Metodo {metodo} invalido')
        return

    # Armazena o centro dasd funções
    m = xclust.cluster_centers_
    
    
    covlist = []

    for i in range(p):
        xci = xin[xclust.labels_ == i]
        if n == 1:
            covi = np.var(xci.T)
        else:
            covi = np.cov(xci.T)
        covlist.append(covi)
    H = np.zeros((N, p))

    for j in range(N):
        for i in range(p):
            mi = np.array(m[i, :])
            cov = np.array(covlist[i])
            H[j, i] = pdf_mv(xin[j, :], mi, cov + 1e-3*np.identity(cov.shape[0]), n)

    Haug = np.append(np.ones((N,1)), H, axis = 1)
    W = matmul(np.linalg.pinv(Haug),(yin))

    return(m, covlist, W, H, xclust)

#### Função para encontrar o Y em dado um modelo RBF
def YRBF(xin, modRBF):

    ######### Função Radial Gaussiana #########
    #Definindo função para calcular a PDF
    def pdf_mv(x, m, K, n):
        if n == 1:
            r = np.sqrt(K)
            px = ((1/(np.sqrt(2*np.pi*r*r)))*np.exp(-0.5 * ((x-m)/(r))**2))
        else:
            parte1 = 1/(((2* np.pi)**(n)*(det(K))))
            parte2 = -0.5 * matmul(matmul((x-m).T, (inv(K))), (x-m))
            px = parte1*np.exp(parte2)
        return(px)

    ##################################################

    N = xin.shape[0] # Número de amostras
    n = xin.shape[1] # Dimensão de entrada
    m = modRBF[0]
    covlist = modRBF[1]
    p = len(covlist)
    W = modRBF[2]

    H = np.zeros((N, p))

    for j in range(N):
        for i in range(p):
            mi = m[i, :]
            cov = covlist[i]
            H[j, i] = pdf_mv(xin[j, :], mi, cov + 1e-3*np.identity(cov.shape[0]), n)
    
    Haug = np.append(np.ones((N,1)), H, axis = 1)
    Yhat = matmul(Haug, W)

    return Yhat

#### Função para treinar ELM
def ELM_train(X_data, Y_data, num_neuronios):
    p = num_neuronios
    X = X_data
    Y = Y_data
    n = X.shape[1]
    Z = np.random.uniform(low = -0.5, high = 0.5, size = (n+1, p))
    Xaug = np.append(X, np.ones((X.shape[0], 1)), 1)
    H = np.tanh(np.matmul(Xaug, Z))
    W = np.matmul(np.linalg.pinv(H), Y)
    return W, H, Z

#### Função para encontrar o Y em dado um modelo ELM 
def ELM_y(X_data, W, Z):
    X = X_data
    Xaug_t = np.append(X, np.ones((X.shape[0], 1)), 1)
    H_t = np.tanh(np.matmul(Xaug_t, Z))
    Y_hat = np.matmul(H_t, W)
    return np.where(Y_hat < 0, 1, -1)