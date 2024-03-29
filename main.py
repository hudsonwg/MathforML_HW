import numpy as np
from scipy.sparse import csr_matrix


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
        for line in lines:

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
        pc = norms
        wc = np.log(pc)