# 融合signet_f的数据
import numpy as np
import os
import scipy.io as sio


def stretch_features(data, L):
    '''
    将 N*2048维的features拉伸到 N*L维
    Params:
        data: 数据集合
        L: 目标维度
    Return:
        data: 拉伸后的数据
    '''

    features = np.array(data)
    X = data.shape[1]
    features = np.mat(features)  # 将数组转换为矩阵

    # create stretching matrix
    M = np.random.standard_normal(size=(X, L))
    M = np.mat(M)
    # 进行矩阵拉伸
    features_cst = features * M

    return features_cst
def create_dataset(signetclass,datasetname,multiclass,signetmaxcols,signetmincols,signet_fmaxcols,signet_fmincols):
    if signetclass == "signet":
        maxcols = signetmaxcols
        mincols = signetmincols
    elif signetclass == "signet_f":
        maxcols = signet_fmaxcols
        mincols = signet_fmincols
    if(signetclass != "signet_f" and multiclass != "noise"):
        return

    if datasetname == "gpds":
        partlist = ("/development","/validation","/exploitation")
    else:
        partlist = ("/")
    for part in partlist:

        # 预先为np分配空间
        size = 0
        for parent, dirnames, filenames in os.walk('data/'+datasetname+'_' + signetclass+part):
            for filename in filenames:
                file_path = os.path.join(parent, filename)
                if (os.path.splitext(file_path)[1] == ".mat"):
                    size = size + sio.loadmat(file_path)['features'].shape[0]
        X = np.zeros((size, 2048))
        X2 = np.zeros((size, 2048))
        y = np.zeros(size)
        label = np.zeros(size)
        # 读取集合数据
        index = 0
        for parent, dirnames, filenames in os.walk('data/'+datasetname+'_' + signetclass+part):
            for filename in filenames:
                file_path = os.path.join(parent, filename)
                if (os.path.splitext(file_path)[1] == ".mat"):
                    #第一个视图
                    data = sio.loadmat(file_path)['features']
                    data_shape = data.shape
                    data_rows = data_shape[0]
                    data_cols = data_shape[1]
                    t = np.empty((data_rows, data_cols))
                    for i in range(data_cols):
                        t[:, i] = (data[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
                    #第二个视图
                    if  multiclass == "noise":
                        # p = t*0.3 + np.random.rand(t.shape[0], t.shape[1])*0.7
                        p = stretch_features(t,2048)
                        np.random.shuffle(p)
                    else:
                        dataview2 = sio.loadmat(file_path.replace("signet_f","signet"))['features']
                        p = np.empty((data_rows, data_cols))
                        for i in range(data_cols):
                            p[:, i] = (dataview2[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
                    numlen = t.shape[0]
                    X[index:index + numlen] = t
                    X2[index:index + numlen] = p
                    label[index:index + numlen] = int("real" in filename)
                    y[index:index + numlen] = int(filename.split('_')[1].split('.')[0])
                    index = index + numlen
        if part == "/development":
            tX1 = np.array(X)
            tX2 = np.array(X2)
            tXy = np.array(y)
            tXlabel = np.array(label)
        elif part == "/validation":
            tXV1 = np.array(X)
            tXV2 = np.array(X2)
            tXVy = np.array(y)
            tXVlabel = np.array(label)
        elif part == "/exploitation":
            tXTe1 = np.array(X)
            tXTe2 = np.array(X2)
            tXTey = np.array(y)
            tXTelabel = np.array(label)
        else:
            tX1 = np.array(X)
            tX2 = np.array(X2)
            tXy = np.array(y)
            tXlabel = np.array(label)
    if datasetname == "gpds":
        sio.savemat("noisedata/gpds_" + signetclass + "multiclass=" + multiclass + ".mat",
                    {'X1': tX1, 'X2': tX2, 'Xy': tXy, 'Xlabel': tXlabel, 'XV1': tXV1, 'XV2': tXV2, 'XVy': tXVy,
                     'XVlabel': tXVlabel, 'XTe1': tXTe1, 'XTe2': tXTe2, 'XTey': tXTey, 'XTelabel': tXTelabel})
        print("noisedata/gpds_" + signetclass + "multiclass=" + multiclass + ".mat saved")
    else:
        sio.savemat("noisedata/" + datasetname + "_" + signetclass + "multiclass=" + multiclass + ".mat",
                    {'X1': tX1, 'X2': tX2, 'Xy': tXy, 'Xlabel': tXlabel})
        print("noisedata/" + datasetname + "_" + signetclass + "multiclass=" + multiclass + ".mat saved")
    return
X = np.zeros((28653, 2048))
#计算gpds训练集每个维度的最大最小值
index = 0
ospath = "data/gpds_signet/development"
for parent, dirnames, filenames in os.walk(ospath):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        if (os.path.splitext(file_path)[1] == ".mat"):
            data = sio.loadmat(file_path)['features']
            numlen = data.shape[0]
            X[index:index+numlen] = data
            index = index + numlen
signetmaxcols = X.max(axis=0)
signetmincols = X.min(axis=0)
X = np.zeros((28653, 2048))
# 计算gpds训练集每个维度的最大最小值
index = 0
ospath = "data/gpds_signet_f/development"
for parent, dirnames, filenames in os.walk(ospath):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        if (os.path.splitext(file_path)[1] == ".mat"):
            data = sio.loadmat(file_path)['features']
            numlen = data.shape[0]
            X[index:index+numlen] = data
            index = index + numlen
signet_fmaxcols = X.max(axis=0)
signet_fmincols = X.min(axis=0)
for signetclass in ["signet_f","signet"]:
    for datasetname in ["gpds","cedar","brazilian","mcyt"]:
        for multiclass in ["noise","2feature"]:
            create_dataset(signetclass,datasetname,multiclass,signetmaxcols,signetmincols,signet_fmaxcols,signet_fmincols)
