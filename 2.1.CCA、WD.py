# CCA训练
import numpy as np
import os
import scipy.io as sio
import mysvm
import argparse
from CCA import linCCA
from myreadinput import read_gpds,read_traingpds
from myreadinput import read_dataset
# from drawgraph import savephoto
def traindcca(arguments):
    # Some other configurations parameters for mnist.
    rcov1 = 0.0001
    rcov2 = 0.0001
    classfile="2_viewdata/" + arguments.signetclass+"multiclass=" + arguments.multiclass + ".mat" + '-cca.mat'
    if os.path.isfile(classfile):
        print("Job is already finished!")
        return classfile

    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpuid

    # Set random seeds.
    np.random.seed(0)

    print(classfile)

    if (arguments.signetclass == "signet"):
        trainData, tuneData, testData = read_gpds('noisedata/gpds_signetmulticlass=noise.mat')
    else:
        trainData, tuneData, testData = read_gpds('noisedata/gpds_signet_fmulticlass='+arguments.multiclass+'.mat')


    # Satisfy constraint.
    A,B,m1,m2,_=linCCA(trainData.images1, trainData.images2, 2048, rcov1, rcov2)

    # TSNE visualization and clustering.
    print("Visualizing shared variables!")
    trainData, tuneData, testData = read_gpds(
        "noisedata/gpds" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(trainData.images1 - m1, A)
    z_tune = np.matmul(tuneData.images1 - m1, A)
    z_test = np.matmul(testData.images1 - m1, A)
    gpds = {'development': z_train, 'devy': trainData.y, 'devlabel': trainData.labels, 'validation': z_tune,
            'valy': tuneData.y, 'vallabel': tuneData.labels, 'exploitation': z_test, 'expy': testData.y,
            'explabel': testData.labels}

    dataset = read_dataset(
        "noisedata/cedar" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(dataset.images1 - m1, A)
    cedar = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset(
        "noisedata/mcyt" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(dataset.images1 - m1, A)
    mcyt = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset(
        "noisedata/brazilian" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(dataset.images1 - m1, A)
    brazilian = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    sio.savemat(classfile,
                {'gpds': gpds, 'cedar': cedar, 'mcyt': mcyt, 'brazilian': brazilian})

    return classfile

parser = argparse.ArgumentParser()

parser.add_argument('--aname',default="CCA")
parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
parser.add_argument('--svm-c', type=float, default=1)
parser.add_argument('--svm-gamma', type=float, default=2 ** -11)
parser.add_argument('--datasetname', choices=['gpds','brazilian','cedar','mcyt' ], default="cedar")
parser.add_argument('--gpdssize', type=int, default=50)

parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--gpuid',default="0")
parser.add_argument('--signetclass', default="signet_f")

parser.add_argument('--trainsvmuser', type=int,default=-1)
parser.add_argument('--multiclass', choices=['noise', '2feature'], default="noise")
arguments = parser.parse_args()
print(arguments)

filename = traindcca(arguments)
print(filename)
mysvm.main(arguments,filename)
