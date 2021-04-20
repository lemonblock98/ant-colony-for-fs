from utils import *
from tqdm import tqdm

csv_path = 'E:/dataset/mdp_classify-master/MDP/JM1.csv'
train_data, train_label, test_data, test_label = generate_data(csv_path)
n_epochs = 100
n_ants = 10
n_dims = train_data.shape[1]
print("the Num of the total features:", n_dims)

alpha = 1
beta = 0.2
omega = 0.7
rho = 0.3
mu = 0.7
gamma = 0.7

tau = np.ones([n_dims, n_dims]) # tau 即信息素矩阵
# eta = np.ones([n_dims, n_dims]) # eta 即启发式信息
feat_list = list(range(n_dims)) # feature 总list
best_score = 0.0
score_list = []
acc_list = []

for epoch in range(n_epochs):

    # ============ apply once fs ============
    ant_path = np.zeros([n_ants, n_dims])           # 初始化每只蚂蚁在当前迭代中的路径, 88 means NULL
    ant_acc = np.zeros(n_ants)
    ant_score = np.zeros(n_ants)
    n_feats = np.random.randint(1, n_dims, size=n_ants) # 初始化每只蚂蚁需要选择的特征数

    for i in tqdm(range(n_ants), ncols=100, desc="Epoch %d" % (epoch + 1), total=n_ants):

        ant_path[i, 0] = np.random.randint(n_dims)  # 为每只蚂蚁选择起始节点（特征）
        visited = []                                # 已选择的 feature list
        
        for d in range(n_feats[i]-1):               # 共选择 n_feats-1 次特征
            visited.append(ant_path[i, d])          # 更新 selected 表
            eta = update_eta(train_data, train_label, test_data, test_label, visited)
                                                    # 更新启发式信息, eta = TPR / d, array(n_dims,)
            p = (tau[int(visited[-1])] ** alpha) * (eta ** beta)
            prob = p / sum(p)                       # 计算路径转移矩阵
            route = select_route(prob)              # 寻找下一个特征
            ant_path[i, d+1] = route

    # ==== evaluate each selected subset ====
    for j in range(n_ants):
        selected = list(ant_path[j, :n_feats[j]])
        f, acc = fitness_func(train_data, train_label, test_data, test_label, selected, omega)
                                                    # 计算适应度函数
        ant_score[j] = f
        ant_acc[j] = acc
        if f > best_score:                          # 保存为全局的最优解
            best_path = ant_path[j]
            best_score = f
            best_path_acc = acc
    
    best_ant = np.argmax(ant_score)                 # 最优蚂蚁
    near_ant = np.argmax(np.concatenate([ant_score[:best_ant], [0], ant_score[best_ant+1:]]))
                                                    # 第二优蚂蚁
    print("Best Score: {}, the Accuracy: {}, Num of Features: {}".format(\
        ant_score[best_ant], ant_acc[best_ant], n_feats[best_ant]))
    
    score_list.append(ant_score[best_ant])
    acc_list.append(ant_acc[best_ant])

    # ======== update the eta matrix ========
    
    # stage 1 updating
    deta_tau_k = np.zeros([n_ants, n_dims, n_dims])
    for k in range(n_ants):
        value = mu * ant_acc[k] + (1-mu) / n_feats[k] # 更新值
        for m in range(n_feats[k]-1):
            a, b = int(ant_path[k, m]), int(ant_path[k, m+1])
            deta_tau_k[int(k), a, b] = value

    deta_tau_1 = np.sum(deta_tau_k, 0)

    # stage 2 updating
    deta_tau_2 = np.zeros([n_dims, n_dims])
    for p in range(n_feats[best_ant]-1):
        a, b = int(ant_path[best_ant, p]), int(ant_path[best_ant, k+1])
        deta_tau_2[a, b] = gamma * deta_tau_1[a, b]
        
    for p in range(n_feats[near_ant]-1):
        a, b = int(ant_path[near_ant, p]), int(ant_path[near_ant, k+1])
        deta_tau_2[a, b] += (1-gamma) * deta_tau_1[a, b]
    
    # update
    tau = (1-rho) * tau + rho * deta_tau_1 + deta_tau_2

print("The Best Ant Path: ", best_path)
print("The Best Score: ", best_score)
print("The Accuracy use Best Path: ", best_path_acc)
