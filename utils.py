import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
import numpy as np

def generate_data(csv_path, factor=0.2):
    """Load the MDP data and split for train/test set
    Parameters:
        csv_path:   data path
        factor:     the split factor for test set
    Returns:
        train_data, train_label:  pd.Dataframe, list
        test_data, test_label: pd.Dataframe, list
    """
    csv_data = pd.read_csv(csv_path)
    total_data, total_label = csv_data.iloc[:, :-1], csv_data.iloc[:, -1]
    total_label = [{'N':0, 'Y':1}[i] for i in total_label]

    train_data, test_data, train_label, test_label = \
        train_test_split(total_data, total_label, test_size=factor)
    
    return train_data, train_label, test_data, test_label

def update_eta(train_data, train_label, test_data, test_label, tabu):
    """更新启发式信息, eta = TPR / d
    Returns:
        eta: 从tabu最后一个节点出发到其余节点的启发式信息
    """
    n_dims = train_data.shape[1]
    eta = np.zeros(n_dims)
    flist = list(set(range(n_dims)) - set(tabu)) 

    for i in flist:
        clf = svm.SVC(C=2, kernel='rbf', gamma=5, decision_function_shape='ovr')
        clf.fit(train_data.iloc[:, tabu+[i]], train_label)
        pred = clf.predict(test_data.iloc[:, tabu+[i]])
        cf_matrix = confusion_matrix(test_label, pred)
        FN, TP = cf_matrix[1][0], cf_matrix[1][1]
        eta[i] = TP / (TP + FN) / (len(tabu)+1)

    return eta

def select_route(prob):
    """按路径转移概率选择下一个特征
    """
    cs = np.cumsum(prob)
    p = np.random.rand()
    for i in range(len(cs)):
        if cs[i] > p:
            break
    
    return i

def fitness_func(train_data, train_label, test_data, test_label, selected, omega=0.7):
    """适应度函数，评估特征子集好坏
    Returns:
        result: 适应度
    """
    clf = svm.SVC(C=2, kernel='rbf', gamma=5, decision_function_shape='ovr')
    clf.fit(train_data.iloc[:, selected], train_label)
    pred = clf.predict(test_data.iloc[:, selected])

    cf_matrix = confusion_matrix(test_label, pred)
    TN, FP = cf_matrix[0][0], cf_matrix[0][1]
    FPR = FP / (TN + FP)
    f_result = omega*FPR + (1-omega)*(len(selected)/train_data.shape[1])
    acc = accuracy_score(test_label, pred)
    return f_result, acc

if __name__ == '__main__':
    csv_path = 'E:/dataset/mdp_classify-master/MDP/JM1.csv'
    train_data, train_label, test_data, test_label = generate_data(csv_path)

    print(train_data.shape, test_data.shape)

