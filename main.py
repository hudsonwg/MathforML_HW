import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def get_dense(path, path2):
    with open(path, 'r') as file:
        lines = file.readlines()
    rows = []
    cols = []
    values = []
    for line in lines:
        components = line.strip().split()
        row_index = int(components[0])
        col_index = int(components[1])
        value = int(components[2])
        rows.append(row_index)
        cols.append(col_index)
        values.append(value)
    shape = (max(rows) + 1, max(cols) + 1)
    sparse_matrix = csr_matrix((values, (rows, cols)), shape=shape)
    #NOW GET DENSE

    column_sums = np.array(sparse_matrix.sum(axis=0)).flatten()
    columns_to_remove = np.where(column_sums < 1000)[0]
    filtered_columns = np.setdiff1d(np.arange(sparse_matrix.shape[1]), columns_to_remove)
    filtered_matrix = sparse_matrix[:, filtered_columns]

###
###
###

    with open(path2, 'r') as file:
        lines = file.readlines()
    rows = []
    cols = []
    values = []
    for line in lines:
        components = line.strip().split()
        row_index = int(components[0])
        col_index = int(components[1])
        value = int(components[2])
        rows.append(row_index)
        cols.append(col_index)
        values.append(value)
    shape = (max(rows) + 1, max(cols) + 1)
    sparse_matrix = csr_matrix((values, (rows, cols)), shape=shape)
    #NOW GET DENSE

    column_sums = np.array(sparse_matrix.sum(axis=0)).flatten()
    filtered_columns = np.setdiff1d(np.arange(sparse_matrix.shape[1]), columns_to_remove)
    filtered_matrix2 = sparse_matrix[:, filtered_columns]
    return filtered_matrix, filtered_matrix2
def sparse_to_dense(sm):
    return sm.toarray()

if __name__ == '__main__':
    train = sparse_to_dense(get_dense('data/20news-bydate/matlab/train.data', 'data/20news-bydate/matlab/test.data')[0])
    test = sparse_to_dense(get_dense('data/20news-bydate/matlab/train.data', 'data/20news-bydate/matlab/test.data')[1])

    print(train.shape)
    print(test.shape)

    with open('data/20news-bydate/matlab/train.label', 'r') as file:
        lines = file.readlines()
        currentLine = ""
        finals = []
        pc = [np.zeros(len(train[0]),)]
        count = 0
        labels = []
        for line in lines:
            labels.append(line)
            if(line != currentLine):
                currentLine = line
                #normalize, push, and reset pc holder
                finals.append(pc)
                pc = [np.zeros(len(train[0]),)]
            else:
                pc = pc + train[count]
            count += 1

        norms = []
        for i in finals:
            norms.append(np.array(i) / np.array(i).sum(axis=1, keepdims=True))
        #NORMS IS 20 LENGTH ARRAY OF PCS


        #BUILDING THE CLASSIFIER

        pc = np.array(norms).reshape(20, 292)[1:]
        wc = np.log(pc)

        counts = np.bincount(labels)
        counts = counts[1:-1]
        cp = counts / len(labels)
        #print(cp)

        bc = np.log(cp)
        #NOW WE HAVE PC, WC, AND BC, so we can form our classifier

        scores = train.dot(wc.T) + bc
        scores = np.nan_to_num(scores, nan=0)
        pred = np.argmax((-1)*scores, axis=1)
        #print(scores)
        #print(pred)

        labels.append(0)
        #accuracy = (pred == np.array(labels))
        #print(accuracy)

        ## TFIDF STUFF- should return 292xdocument number length vector

        '''
        loop thru all columns
        check column count how many times value is nonzero (y), then take idf = log(total docs/y)
        then check document specific stats and crunch those numbers and append the value
        '''
        TFIDF = np.zeros(train.shape)
        for i in range(0, len(train[0])):
            y = float(np.count_nonzero(train[:][i]))
            if y==0:

                y=1
            idf = np.log(float(len(train[:][i])/y))
            for j in range(0, len(train)):
                a = train[j][i]
                b = np.sum(train[j])
                if(a == 0):
                    TFIDF[j][i] = 0
                else:
                    TFIDF[j][i] = (a/b)*idf
        #print(TFIDF)
        U, S, Vt = np.linalg.svd(TFIDF)
        #print(U)
        #print(S)
        #print(Vt)

        k = 2
        U_pca = U[:, :k]
        S_pca = np.diag(S[:k])
        Vt_pca = Vt[:k, :]

        # Compute PCA by multiplying matrices
        Y = np.dot(U_pca, S_pca)
        print(Y)
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]

        # Plot the vectors using plt.scatter()
        #plt.figure(figsize=(8, 6))
        #plt.scatter(x_coords, y_coords)
        #plt.xlabel('X-axis')
        #plt.ylabel('Y-axis')
        #plt.title('Plot of Y after PCA on TFIDF')
        #plt.grid(True)
        #plt.show()


        #6

        kGMM = 20  # Number of clusters
        initial_means = ...
        initial_covariances = ...
        initial_mixing_coeffs = ...

        # Step 3: Expectation-Maximization (EM) Algorithm
        gmm = GaussianMixture(n_components=kGMM, init_params='random', random_state=42)
        gmm.means_init = initial_means
        gmm.covariances_init = initial_covariances
        gmm.weights_init = initial_mixing_coeffs
        gmm.fit(TFIDF)  # data matrix obtained from LSI

        # Compute PCA projection Θ
        pca = PCA(n_components=kGMM)
        Theta = pca.fit_transform(TFIDF)

        # Compute Θμc for each cluster
        Theta_means = np.dot(gmm.means_, pca.components_)

        ####