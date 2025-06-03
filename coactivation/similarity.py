import numpy as np

def gradient(arr):
    grad = np.empty_like(arr)
    grad[:-1] = np.diff(arr)
    grad[-1] = arr[0]-arr[-1]
    return grad

def mi(x, y):
    def entropy(arr):
        _, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p))
    def joint_entropy(a, b):
        pairs = np.column_stack((a, b))
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p))
    return entropy(x) + entropy(y) - joint_entropy(x, y)

def g(x, y):
    gradx = gradient(x)
    grady = gradient(y)
    G = 0
    for i in range(len(x)):
        alpha = np.arccos(np.dot(gradx[i], grady[i]) / (abs(gradx[i]) * abs(grady[i])))
        if alpha!=alpha:
            alpha=0
        w = (np.cos(2*alpha)+1)/2
        G += w * min(abs(gradx[i]), abs(grady[i]))
    return G

def gmi(cm1, cm2):
    values=[]
    for i in range(3):
        G = g(cm1.iloc[i,:].values, cm2.iloc[i,:].values)
        I = mi(cm1.iloc[i,:].values, cm2.iloc[i,:].values)
        values.append(G*I)
    return values, np.mean(values)

def ruzicka(x, y):
    def norm(arr):
        return arr / np.linalg.norm(arr)
    x_norm = norm(x)
    y_norm = norm(y)
    return np.sum(np.minimum(x_norm, y_norm))


def mi_cm(cm1, cm2):
    values=[]
    for i in range(3):
        I = mi(cm1.iloc[i,:].values, cm2.iloc[i,:].values,)
        values.append(I)
    return values, np.mean(values)

def g_cm(cm1, cm2):
    values=[]
    for i in range(3):
        G = g(cm1.iloc[i,:].values, cm2.iloc[i,:].values)
        values.append(G)
    return values, np.mean(values)

def gmi_cm(cm1, cm2):
    values=[]
    for i in range(3):
        G = g(cm1.iloc[i,:].values, cm2.iloc[i,:].values)
        I = mi(cm1.iloc[i,:].values, cm2.iloc[i,:].values)
        values.append(G*I)
    return values, np.mean(values)

def ruzicka_cm(cm1, cm2):
    values=[]
    for i in range(3):
        R = ruzicka(cm1.iloc[i,:].values, cm2.iloc[i,:].values)
        values.append(R)
    return values, np.mean(values)
