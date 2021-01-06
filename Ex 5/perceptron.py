import numpy as np

def cria_dados (num_amostras, centros, des_padroes, num_dist, dim =2):
    '''
    Função que gera conjuntos de dados utilizando a distribuição normal de num_dists 'classes' diferentes
    Entradas:
        - num_amostras: numero de amostras por conjunto de dados
        - centros: listas contendo os centros de cada distribuição normal
        - des_padroes: lista contendo os desvios padrões de cada distribuição normal
        - num_dist: numero de classes a ser criado
        - dim: numero de dimensões de cada conjunto de dados
    Retorna:
        data_set: data-set criado
    '''
    
    data_set = np.empty((num_amostras*num_dist, dim+1))
    
    for i in range(num_dist):
        x = np.random.normal(loc=centros[i], scale=des_padroes[i], size=(num_amostras,dim))
        dist_col = np.full((num_amostras, 1), i)
        data_set[num_amostras*i:(i+1)*num_amostras, :] = np.append(x, dist_col, 1)
    
    return data_set

def trainperceptron (xin, yd, eta, tol, maxepocas):
    
    '''
    Função que aplica o metodo para treinamento de perceptron
    yd: tem que ser garado para as xin (concatenado xall), metade 0 e metade 1
    xin: Entrada Nxn de dados de matriz
    eta: Peso de atualizacao do passo
    tol: tolerancia do erro
    maxepocas: numero maximo de epocas permitido
    retorna:
        - wt: parametros da função avaliada
        - evec: erro médio por época
    '''

    N = xin.shape[0]  #recebe as linhas
    n = xin.shape[1] # recebe as colunas
    xin = np.append(np.ones((N,1)), xin,axis = 1)

    wt = np.random.randn(n+1, 1)*0.01

    nepocas = 0
    eepoca = tol+1
    # inicializa vetor erro evec 
    evec = np.empty([maxepocas+1, 1])
    while ((nepocas < maxepocas) and (eepoca>tol)): #eepocas erro da epoca e tol tolerancia
        ei2 = 0
        if (nepocas+1)%10 == 0:
            print(f'Epoca: {nepocas+1}') 
        #sequencia aleatoria para treinamento
        xseq = (np.arange(N))
        np.random.shuffle(xseq)
        for i in range(N):
            #padrao para sequencia aleatoria
            irand = xseq[i]
            yhati = (np.matmul(xin[None, irand, :], wt)) >= 0
            ei = yd[irand]-yhati
            dw = eta * ei * xin[None, irand, :]
            #atualizacao do peso w
            wt = wt + dw.T
            #erro acumulado
            ei2 = ei2+ei*ei
        #numero de epocas
        nepocas = nepocas+1
        evec[nepocas] = ei2/N
        #erro por epoca
        eepoca = evec[nepocas]
    return wt, evec[1:nepocas]

def yperceptron(xin, w):

    '''
    Função que retorna a saída de um sistema cujo parametros foram obtidos usando a função trainadaline
    xin: vetor x de entrada
    w: parametros a serem considerados
    retorna: vetor y correspondente ao modelo com parametros w
    '''

    return ((np.matmul(np.append(np.ones((xin.shape[0],1)), xin,axis = 1 ), w))>=0).astype(int)

def train_test_balanceados(data, train_v = 0.9, dist = 2):
    '''
    Fazer divisão em dados de teste e treino
    Argumentos:
    dados = dados que se deseja dividir
    train_v = Porcentagem de dados de treino (deve ser menor que 1)
    dist = número de distribuições no meu dataframe
    Retorna: Vetores X, Y de treino e teste adequadamente distribuidos 
    '''
    assert(train_v<=1), 'train_v deve ser menor que 1'
    
    np.random.shuffle(data)
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(dist):
        dados = np.copy(data[data[:, -1] == i])
        train, test = dados[:int(0.7*dados.shape[0])], dados[int(0.7*dados.shape[0]):]
        X_train.append(train[:, :train.shape[1]-1])
        y_train.append(train[:,train.shape[1]-1:])
        X_test.append(test[:, :train.shape[1]-1])
        y_test.append(test[:,train.shape[1]-1:])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    train = np.append(X_train, y_train, 1)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    test = np.append(X_test, y_test, 1)

    np.random.shuffle(train)
    np.random.shuffle(test)

    return (train, test)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Label real')
    plt.xlabel('Label previsto\naccuracia={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()