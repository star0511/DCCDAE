# DCCA训练
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow.compat.v1 as tf

import requests
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
import scipy.io as sio
import mysvm
import argparse
import DDCCAE as ddccae
from CCA import linCCA
from myreadinput import read_gpds
from myreadinput import read_dataset
def trainddccae(arguments):
    # Some other configurations parameters for mnist.
    learning_rate = arguments.learning_rate
    l2_penalty = 0.0001
    rcov1 = 0.0001
    rcov2 = 0.0001
    # filename = "noisedata/" + arguments.datasetname + "_" + arguments.signetclass+"multiclass=" + arguments.multiclass + ".mat"
    classfile="2_viewdata/" + arguments.signetclass + "_batchsize"+str(arguments.batch_size)+"epoch"+str(arguments.epoch)+"lr"+str(learning_rate) + "-"+str(arguments.net_hidden_layer[0])+"-"+str(arguments.net_hidden_layer[1])+"-"+str(arguments.net_hidden_layer[2])+"-"+str(arguments.net_hidden_layer[3])+"lamda"+str(arguments.ddccaelamda)+"droprate"+str(arguments.droprate)+'-ddccae.mat'
    if os.path.isfile(classfile):
        print("Job is already finished!")
        return classfile
    # Handle multiple gpu issues.
    # tf.reset_default_graph()
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpuid
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=0.5 # （尽量用这种方式）设置最大占有GPU不超过显存的25%
    sess = tf.Session(config=config)
    
    # Set random seeds.
    np.random.seed(0)
    tf.set_random_seed(0)

    # Define network architectures.
    network_architecture=dict(
        n_input1=2048, # feature1 data input (shape: 2048)
        n_input2=2048, # feature2 data input (shape: 2048)
        n_z=arguments.net_hidden_layer[-1],  # Dimensionality of shared latent space
        F_hidden_widths=arguments.net_hidden_layer,
        F_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None],
        G_hidden_widths=arguments.net_hidden_layer,
        G_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
        )

    # First, build the model.
    model=ddccae.DDCCAE(classfile,network_architecture, rcov1, rcov2, learning_rate, l2_penalty,arguments.ddccaelamda)
    print(classfile)

    if (arguments.signetclass == "signet"):
        trainData, tuneData, testData = read_gpds('noisedata/gpds_signetmulticlass=noise.mat')
    else:
        if (arguments.multiclass == "noise"):
            trainData, tuneData, testData = read_gpds('noisedata/gpds_signet_fmulticlass=noise.mat')
        else:
            trainData, tuneData, testData = read_gpds('noisedata/gpds_signet_fmulticlass=2feature.mat')

    # Traning.
    model=ddccae.train(model, trainData, tuneData, batch_size=arguments.batch_size, max_epochs=arguments.epoch,droprate=arguments.droprate)

    # Satisfy constraint.
    FX1,_=model.compute_projection(1, trainData.images1)
    FX2,_=model.compute_projection(2, trainData.images2)
    A,B,m1,m2,_=linCCA(FX1, FX2, model.n_z, rcov1, rcov2)

    trainData, tuneData, testData = read_gpds(
        "noisedata/gpds" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(model.compute_projection(1, trainData.images1)[0] - m1, A)
    z_tune = np.matmul(model.compute_projection(1, tuneData.images1)[0] - m1, A)
    z_test = np.matmul(model.compute_projection(1, testData.images1)[0] - m1, A)
    gpds = {'development': z_train, 'devy': trainData.y, 'devlabel': trainData.labels, 'validation': z_tune,
            'valy': tuneData.y, 'vallabel': tuneData.labels, 'exploitation': z_test, 'expy': testData.y,
            'explabel': testData.labels}

    dataset = read_dataset(
        "noisedata/cedar" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    cedar = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset(
        "noisedata/mcyt" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    mcyt = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    dataset = read_dataset(
        "noisedata/brazilian" + "_" + arguments.signetclass + "multiclass=" + arguments.multiclass + ".mat")
    z_train = np.matmul(model.compute_projection(1, dataset.images1)[0] - m1, A)
    brazilian = {'development': z_train, 'devy': dataset.y, 'devlabel': dataset.labels}
    sio.savemat(classfile,
                {'gpds': gpds, 'cedar': cedar, 'mcyt': mcyt, 'brazilian': brazilian})
    return classfile

parser = argparse.ArgumentParser()


parser.add_argument('--aname',default="DCCDAE")
parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
parser.add_argument('--svm-c', type=float, default=1)
parser.add_argument('--svm-gamma', type=float, default=2 ** -11)

parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--gpuid',default="0")
parser.add_argument('--datasetname', choices=['gpds','brazilian','cedar','mcyt' ], default="cedar")
parser.add_argument('--gpdssize', type=int, default=160)
parser.add_argument('--signetclass', choices=['signet', 'signet_f'], default="signet_f")
parser.add_argument('--multiclass', choices=['noise', '2feature'], default="noise")
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=3000)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--ddccaelamda', type=float, default=2000)
parser.add_argument('--droprate', type=float, default=0.5)
parser.add_argument('--trainsvmuser', type=int,default=-1)
parser.add_argument('--net-hidden-layer', nargs='+', type=int ,default=[2048, 2048, 4096, 2048])
arguments = parser.parse_args()
print(arguments)

filename = trainddccae(arguments)
print(filename)
mysvm.main(arguments,filename)