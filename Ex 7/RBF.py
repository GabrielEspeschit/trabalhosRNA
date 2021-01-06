import numpy as np
from numpy import matmul
from numpy.linalg import inv, det
from sklearn.cluster import KMeans as kmeans

def treinaRBF(xin, yin, p):
    
    ######### Função Radial Gaussiana #########
    #Definindo função para calcular a PDF
    def pdf_mv(x, m, K, n):
        if n == 1:
            r = np.sqrt(K)
            px = 1/(np.sqrt(2*np.pi*r**2))*np.exp(-0.5 * ((x-m)/(r**2)))
        else:
            px = (1/(np.sqrt(2*np.pi**(n*det(K)))))
            px = px * np.exp(-0.5 * (matmul(matmul((x-m).T, inv(K)), (x-m))))
        return(px)

    ##################################################

    N = xin.shape[0] # Número de amostras
    n = xin.shape[1] # Dimensão de entrada

    xclust = kmeans(n_clusters=p).fit(xin) # Fazendo o Clustering com a função kmeans do sklearn

    # Armazena o centro dasd funções
    m = xclust.cluster_centers_
    covlist = []

    for i in range(p):
        xci = xin[xclust.labels_ == i, :]
        
        if n == 1:
            covi = np.var(xci)
        else:
            covi = np.cov(xci)
        covlist.append(covi)

    H = np.zeros((N, p))

    for j in range(N):
        for i in range(p):
            mi = m[i, :]
            cov = covlist[i]
            H[j, i] = pdf_mv(xin[j, :], mi, cov, n)

    # print(H)
    Haug = np.append(np.ones((N,1)), H, axis = 1)
    W = matmul(inv(matmul(matmul(Haug.T, Haug), Haug.T)), yin)

    return(m, covlist, W, H)

def YRBF(xin, modRBF):

    ######### Função Radial Gaussiana #########
    #Definindo função para calcular a PDF
    def pdf_mv(x, m, K, n):
        if n == 1:
            r = np.sqrt(K)
            px = 1/(np.sqrt(2*np.pi*r**2))*np.exp(-0.5 * ((x-m)/(r**2)))
        else:
            px = (1/(np.sqrt(2*np.pi**(n*det(K)))))
            px = px * np.exp(-0.5 * (matmul(matmul((x-m).T, inv(K)), (x-m))))
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
            H[j, i] = pdf_mv(xin[j, :], mi, cov, n)

    Haug = np.append(np.ones((N,1)), H, axis = 1)
    Yhat = matmul(Haug, W)

    return Yhat