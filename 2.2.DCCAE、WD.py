#!/usr/bin/env python
# coding=utf-8
# DCCA训练
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
import mysvm
import requests
import argparse
import DCCAE as dccae
from CCA import linCCA
from myreadinput import read_gpds
from myreadinput import read_dataset
def traindccae(arguments):
    # Some other configurations parameters for mnist.
    learning_rate = arguments.learning_rate
    l2_penalty = 0.0001
    rcov1 = 0.0001
    rcov2 = 0.0001
    args = {"Z": arguments.net_hidden_layer[-1], "dropprob": 0.0,"checkpoint":"./dccae_mnist","batchsize":arguments.batch_size,"epoch":arguments.epoch,"gpuid":"0"}
    # filename = "noisedata/" + "_" + arguments.signetclass+"multiclass=" + arguments.multiclass + ".mat"
    classfile="2_viewdata/" + arguments.signetclass + "_batchsize"+str(args['batchsize'])+"_epoch"+str(args['epoch'])+"_lr"+str(learning_rate) + "-"+str(arguments.net_hidden_layer[0])+"-"+str(arguments.net_hidden_layer[1])+"-"+str(arguments.net_hidden_layer[2])+"-"+str(arguments.net_hidden_layer[3])+"lamda"+str(arguments.dccaelamda)+'-dccae.mat'
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

    # Obtain parsed arguments.
    Z=args['Z']
    print("Dimensionality of shared variables: %d" % Z)
    dropprob=args['dropprob']
    print("Dropout rate: %f" % dropprob)
    checkpoint=args['checkpoint']
    print("Trained model will be saved at %s" % checkpoint)


    net_hidden_widths = arguments.net_hidden_layer
    # Define network architectures.
    network_architecture=dict(
        n_input1=2048, # feature1 data input (shape: 2048)
        n_input2=2048, # feature2 data input (shape: 2048)
        n_z=Z,  # Dimensionality of shared latent space
        F_hidden_widths=net_hidden_widths,
        F_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None],
        G_hidden_widths=net_hidden_widths,
        G_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
        )

    # First, build the model.
    model=dccae.DCCAE(classfile,network_architecture, rcov1, rcov2, learning_rate, l2_penalty,arguments.dccaelamda)
    saver=tf.train.Saver()

#     # Second, load the saved moded, if provided.
#     if checkpoint and os.path.isfile(checkpoint + ".meta"):
#         print("loading model from %s " % checkpoint)
#         saver.restore(model.sess, checkpoint)
#         epoch=model.sess.run(model.epoch)
#         print("picking up from epoch %d " % epoch)
#         tunecost=model.sess.run(model.tunecost)
#         print("tuning cost so far:")
#         print(tunecost[0:epoch])
#     else:
#         print("checkpoint file not given or not existent!")

    # File for saving classification results.
    # classfile=filename + 'dcca.mat'
    # if os.path.isfile(classfile):
    #     print("Job is already finished!")
    #     return classfile
    print(classfile)

    if (arguments.signetclass == "signet"):
        trainData, tuneData, testData = read_gpds('noisedata/gpds_signetmulticlass=noise.mat')
    else:
        trainData, tuneData, testData = read_gpds('noisedata/gpds_signet_fmulticlass='+arguments.multiclass+'.mat')


    # Traning.
    model=dccae.train(model, trainData, tuneData, saver, checkpoint, batch_size=args['batchsize'], max_epochs=args['epoch'], save_interval=1, keepprob=(1.0-dropprob))

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

parser.add_argument('--aname',default="DCCAE")
parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
parser.add_argument('--svm-c', type=float, default=1)
parser.add_argument('--svm-gamma', type=float, default=2 ** -11)

parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--gpuid',default="0")
parser.add_argument('--datasetname', choices=['gpds','brazilian','cedar','mcyt' ], default="cedar")
parser.add_argument('--gpdssize', type=int, default=160)
parser.add_argument('--signetclass', choices=['signet', 'signet_f'], default="signet_f")
parser.add_argument('--multiclass', choices=['noise', '2feature'], default="noise")
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=3000)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--dccaelamda', type=float, default=2000)
parser.add_argument('--trainsvmuser', type=int,default=-1)
parser.add_argument('--net-hidden-layer', nargs='+', type=int ,default=[2048, 2048, 4096, 2048])
arguments = parser.parse_args()
print(arguments)

filename = traindccae(arguments)
print(filename)
mysvm.main(arguments,filename)