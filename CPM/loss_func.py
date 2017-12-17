import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  

def ideal_addGaussian(x, y, height, width):
    sigma = 21.0

    xx = np.linspace(1.0, float(height), height)
    yy = np.linspace(1.0, float(width), width)

    X, Y = np.meshgrid(xx, yy)
    X = X - x
    Y = Y - y

    D2 = np.power(X, 2) + np.power(Y, 2)

    Exponent = D2 * (1 / sigma) * (1 / sigma) * 0.5 * (-1)
    label_matrix = np.exp(Exponent)
    return label_matrix
'''
Args:
    stageOut: list of stage output matrix 
    index: which picture you used for this train iteration in 2000 dataset pictures
    height: trainpic height pixels 
    width: trainpic width pixels

Returns:
    loss value
'''
def loss_func(stageOut, index, height, width):
    # Path for dataset
    matfn = '/Users/yuzumon/Documents/Grad/Term 3/545/Project/lsp_dataset/joints.mat'
    data = sio.loadmat(matfn)  
 
    data = data['joints']  
    coordinate_list = data[:, :, index]
    matrix_list = []
    for i in range(14):
        matrix_list.append(ideal_addGaussian(coordinate_list[0][i], coordinate_list[1][i], height, width))

    ideal_matrix = matrix_list[0]
    for i in range(13):
        ideal_matrix = np.dstack(ideal_matrix, matrix_list[i + 1])

    ideal_matrix = np.dstack(ideal_matrix, np.zeros((height, width)))

    sum2 = 0.0

    for i in range(6):
        stageOutMatrix = stageOut[i]
        tmpMatrix = stageOutMatrix - ideal_matrix
        afterNormMatrix = np.linalg.norm(tmpMatrix, axis = 2)
        afterNormMatrix = np.power(afterNormMatrix, 2)
        sum1 = np.sum(afterNormMatrix, axis = 1)
        sum2 += np.sum(sum1, axis = 0)

    return sum2

