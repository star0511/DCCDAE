# CCA训练
import numpy as np
import os
import scipy.io as sio
import mysvm
import argparse
from CCA import linCCA
from myreadinput import read_gpds,read_traingpds
from myreadinput import read_dataset
from drawgraph import savephoto
def traindcca(arguments):
    # Some other configurations parameters for mnist.
    rcov1 = 0.0001
    rcov2 = 0.0001
    classfile="2_viewdata/" + arguments.signetclass+"multiclass=" + arguments.multiclass + ".mat" + '-base.mat'
    if os.path.isfile(classfile):
        print("Job is already finished!")
        return classfile

    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpuid

    # Set random seeds.
    np.random.seed(0)

    print(classfile)

    if (arguments.signetclass == "signet_f"):
        trainData, tuneData, testData = read_traingpds('noisedata/gpds_signet_fmulticlass=noise.mat')
    else:
        if (arguments.multiclass == "noise"):
            trainData, tuneData, testData = read_traingpds('noisedata/gpds_signetmulticlass=noise.mat')
        else:
            trainData, tuneData, testData = read_traingpds('noisedata/gpds_signetmulticlass=signet-signet_f.mat')


    # Satisfy constraint.
    A,B,m1,m2,_=linCCA(trainData.images1, trainData.images2, 2048, rcov1, rcov2)

    # TSNE visualization and clustering.
    print("Visualizing shared variables!")
    trainData, tuneData, testData = read_gpds(
        "noisedata/gpds" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    gpds = {'development': trainData.images1, 'devy': trainData.y, 'devlabel': trainData.labels, 'validation': tuneData.images1,
            'valy': tuneData.y, 'vallabel': tuneData.labels, 'exploitation': testData.images1, 'expy': testData.y,
            'explabel': testData.labels}

    dataset = read_dataset(
        "noisedata/cedar" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    cedar = {'development': dataset.images1, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset(
        "noisedata/mcyt" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    mcyt = {'development': dataset.images1, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset(
        "noisedata/brazilian" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    brazilian = {'development': dataset.images1, 'devy': dataset.y, 'devlabel': dataset.labels}
    sio.savemat(classfile,
                {'gpds': gpds, 'cedar': cedar, 'mcyt': mcyt, 'brazilian': brazilian})

    return classfile

parser = argparse.ArgumentParser()

parser.add_argument('--modelclass', default="baseline")
parser.add_argument('--signetclass', default="signet_f")
parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
parser.add_argument('--datasetname', choices=['gpds','brazilian','cedar','mcyt' ], default="cedar")
parser.add_argument('--gen-for-train', type=int, default=12)
parser.add_argument('--gen-for-test', type=int, default=10)
parser.add_argument('--forg-from_exp', type=int, default=0)
parser.add_argument('--forg-from_dev', type=int, default=10)#训练集中反例个数
parser.add_argument('--svm-c', type=float, default=1)
parser.add_argument('--svm-gamma', type=float, default=2 ** -11)

parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--gpuid',default="1")
parser.add_argument('--gpdssize', type=int, default=50)
parser.add_argument('--multiclass', default="noise")
arguments = parser.parse_args()
print(arguments)
filename = traindcca(arguments)
mysvm.main(arguments,filename)
