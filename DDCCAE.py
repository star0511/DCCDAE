import numpy as np
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from CCA import npdccaecoss, ccaloss,aeloss
import visdom

########################## CONSTRUCT AN DENSE DNN ##############################
def autoencoder(inputs, D_in, layer_widths, layer_activations, keeprate, name, variable_reuse, initializer):
    """ Expects flattened inputs.
    """

    width=D_in
    with tf.variable_scope(name, reuse=variable_reuse, initializer=initializer):
        activation = tf.nn.dropout(inputs, rate=1 - keeprate)
        # activation = tf.zeros(shape=tf.shape(inputs), dtype=np.float32)
        for i in range(len(layer_widths)):
            weights=tf.get_variable("weights_layer_" + str(i+1), [width, layer_widths[i]])
            biases=tf.get_variable("biases_layer_" + str(i+1), [layer_widths[i]])
            activation=tf.add(tf.matmul(activation, weights), biases)
            if layer_activations[i] is not None:
                activation=layer_activations[i](activation)
            width=layer_widths[i]
        layer_widths = layer_widths[::-1]
        layer_widths[-1] = D_in
        r = activation
        layer_widths = layer_widths[::-1]
        del (layer_widths[0])
        layer_widths.append(2048)
        for i in range(len(layer_widths)):
            weights = tf.get_variable("weights_layer_" + str(len(layer_widths) + i + 1), [width, layer_widths[i]])
            biases = tf.get_variable("biases_layer_" + str(len(layer_widths) + i + 1), [layer_widths[i]])
            r = tf.add(tf.matmul(r, weights), biases)
            if layer_activations[i] is not None:
                r = layer_activations[i](r)
            width = layer_widths[i]
    return activation,r

class DDCCAE(object):
    
    def __init__(self, filename, architecture,rcov1=0, rcov2=0, learning_rate=0.0001, l2_penalty=0.0, ddccaelamda=100):
        # Save the architecture and parameters.
        self.network_architecture=architecture
        self.l2_penalty=l2_penalty
        self.ddccaelamda = ddccaelamda
        self.learning_rate=tf.Variable(learning_rate,trainable=False)
        # self.learning_rate=learning_rate
        self.rcov1=rcov1
        self.rcov2=rcov2
        self.n_input1=n_input1=architecture["n_input1"]
        self.n_input2=n_input2=architecture["n_input2"]
        self.n_z=n_z=architecture["n_z"]
        # Tensorflow graph inputs.
        self.batchsize=tf.placeholder(tf.float32)
        self.x1=tf.placeholder(tf.float32, [None, n_input1])
        self.x2=tf.placeholder(tf.float32, [None, n_input2])
        self.keepprob=tf.placeholder(tf.float32)
        self.filename = filename

        # Variables to record training progress.
        self.epoch=tf.Variable(0, trainable=False)
        self.tunecost=tf.Variable(tf.zeros([1000]), trainable=False)
        self.wind = visdom.Visdom(env=filename.replace("2_viewdata/", ""), port=2333)
        # Initialize network weights and biases.
        # initializer = tf.zeros_initializer()
        # initializer = tf.glorot_uniform_initializer()
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        
        # Use the recognition network to obtain the Gaussian distribution (mean and log-variance) of latent codes.
        self.FX1 = autoencoder(self.x1, self.n_input1, architecture["F_hidden_widths"],
                               architecture["F_hidden_activations"], self.keepprob, "F", None, initializer)
        self.FX2 = autoencoder(self.x2, self.n_input2, architecture["G_hidden_widths"],
                               architecture["G_hidden_activations"], self.keepprob, "G", None, initializer)
        self.ccaloss = ccaloss(self.FX1[0], self.FX2[0], self.batchsize, n_z, n_z, n_z, self.rcov1, self.rcov2)
        self.aeloss = self.ddccaelamda * aeloss(self.x1, self.x2, self.FX1[1], self.FX2[1])
        # Weight decay.
        self.weightdecay=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # Define cost and use the ADAM optimizer.
        self.cost = self.ccaloss + self.aeloss + l2_penalty * self.weightdecay
        # LEARNING_RATE_DECAY = 0.9  # 学习率衰减率
        # LEARNING_RATE_STEP = 10  # 喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE
        # # 运行了几轮BATCH_SIZE的计数器，初值给0, 设为不被训练
        # self.global_step = tf.Variable(0, trainable=False)
        # # 定义指数下降学习率
        # self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, LEARNING_RATE_STEP,
        #                                      LEARNING_RATE_DECAY, staircase=True)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost, global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # self.optimizer=tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.cost)
        # self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate,).minimize(self.cost)

        self.saver = tf.train.Saver()
        # Initializing the tensor flow variables and launch the session.
        init=tf.global_variables_initializer()
        # init = tf.ones_initializer()
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        # self.loadcheckpoint()
        self.sess.run(init)

    def assign_lr(self, lr):
        self.sess.run(tf.assign(self.learning_rate, lr))

    def assign_epoch(self, EPOCH_VALUE):
        self.sess.run(tf.assign(self.epoch, EPOCH_VALUE))
    
    def assign_tunecost(self, TUNECOST_VALUE):
        self.sess.run(tf.assign(self.tunecost, TUNECOST_VALUE))

    
    def partial_fit(self, X1, X2, droprate):
        opt, FX1, FX2,aeloss, ccaloss, cost = self.sess.run([self.optimizer,self.FX1,self.FX2,self.aeloss, self.ccaloss, self.cost], feed_dict={self.x1:X1, self.x2:X2, self.batchsize: X1.shape[0], self.keepprob: 1 - droprate})
        return aeloss, ccaloss, cost

    def computeloss(self, X1, X2):
        aeloss, ccaloss, cost = self.sess.run([self.aeloss, self.ccaloss, self.cost],
                                              feed_dict={self.x1: X1, self.x2: X2, self.batchsize: X1.shape[0],
                                                         self.keepprob: 1.0})
        return aeloss, ccaloss, cost

    def computeinputloss(self, X1, X2):
        ccaloss = self.sess.run(self.ccaloss,
                                feed_dict={self.FX1[0]: X1, self.FX2[0]: X2, self.batchsize: X1.shape[0],
                                           self.keepprob: 1.0})
        return ccaloss
    def compute_projection(self, view, X):
        
        N=X.shape[0]
        Dout=self.n_z
        
        FX=np.zeros([N, Dout], dtype=np.float32)
        if view==1:
            Din = self.n_input1
        else:
            Din = self.n_input2
        R = np.zeros([N, Din], dtype=np.float32)
        batchsize=5000
        for batchidx in range(np.ceil(N / batchsize).astype(int)):
            idx=range( batchidx*batchsize, min(N, (batchidx+1)*batchsize) )
            if view==1:
                tmp=self.sess.run(self.FX1, feed_dict={self.x1: X[idx,:], self.keepprob: 1.0})
            else:
                tmp=self.sess.run(self.FX2, feed_dict={self.x2: X[idx,:], self.keepprob: 1.0})
            FX[idx,:] = tmp[0]
            R[idx, :] = tmp[1]
        return FX,R

    def savecheckpoint(self):
        graph = tf.get_default_graph()
        print(self.sess.run(graph.get_tensor_by_name("F/weights_layer_1:0")))
        print(self.sess.run(self.sess.graph.get_tensor_by_name("F/weights_layer_1:0")))

        self.saver.save(self.sess, "checkpoint/" + self.filename + "/model")
    def loadcheckpoint(self):
        try:
            savagraph = tf.Graph()
            with savagraph.as_default():
                with tf.Session() as sess:
                # sess = tf.Session(graph=savagraph)
                    saver = tf.train.import_meta_graph("checkpoint/2_viewdata/brazilian_signetmulticlass=noise.matbatchsize5000lr0.001-4096-4096-4096-2048lamda2000droprate0.2-ae.mat/model.meta")
                    saver.restore(sess, tf.train.latest_checkpoint("checkpoint/2_viewdata/brazilian_signetmulticlass=noise.matbatchsize5000lr0.001-4096-4096-4096-2048lamda2000droprate0.2-ae.mat/"))

                # saver = tf.train.import_meta_graph("checkpoint/" + self.filename + "/model.meta")
                # saver.restore(sess, tf.train.latest_checkpoint("checkpoint/" + self.filename))
                # print(sess.run(sess.graph.get_tensor_by_name("F/weights_layer_1:0")))
                # print(self.sess.run(self.sess.graph.get_tensor_by_name("F/weights_layer_1:0")))
                    layer_widths = self.network_architecture["F_hidden_widths"]
                    for scope in ("F","G"):
                        for i in range(len(layer_widths)):
                            self.sess.run(
                                tf.assign(self.sess.graph.get_tensor_by_name(scope + "/weights_layer_" + str(i + 1) + ":0"),
                                          sess.run(
                                              sess.graph.get_tensor_by_name(scope + "/weights_layer_" + str(i + 1) + ":0"))))

                            self.sess.run(
                                tf.assign(self.sess.graph.get_tensor_by_name(scope+"/biases_layer_" + str(i + 1)+":0"),
                                          sess.run(
                                              sess.graph.get_tensor_by_name(scope+"/biases_layer_" + str(i + 1)+":0"))))

                        # scope+"/weights_layer_" + str(i + 1)+":0"
                        # scope+"/biases_layer_" + str(i + 1)+":0"
                        for i in range(len(layer_widths)):
                            self.sess.run(
                                tf.assign(self.sess.graph.get_tensor_by_name(scope + "/weights_layer_" + str(len(layer_widths) + i + 1) + ":0"),
                                          sess.run(
                                              sess.graph.get_tensor_by_name(scope + "/weights_layer_" + str(len(layer_widths) + i + 1) + ":0"))))

                            self.sess.run(
                                tf.assign(self.sess.graph.get_tensor_by_name(scope + "/biases_layer_" + str(len(layer_widths) + i + 1) + ":0"),
                                          sess.run(
                                              sess.graph.get_tensor_by_name(scope + "/biases_layer_" + str(len(layer_widths) + i + 1) + ":0"))))
                        # scope+"/weights_layer_" + str(len(layer_widths) + i + 1)+":0"
                        # scope+"/biases_layer_" + str(len(layer_widths) + i + 1)+":0"
            # sess.close()
            print("已加载预训练模型")
        except:
            raise
            print("未加载保存的模型")

    
def train(model, trainData, tuneData, batch_size=100, max_epochs=10,droprate=0):
    epoch=model.sess.run(model.epoch)
    TUNECOST=model.sess.run(model.tunecost)
    lr=model.sess.run(model.learning_rate)

    n_samples=trainData.num_examples
    if batch_size != 0:
        total_batch=int(math.ceil(1.0 * n_samples / batch_size))
    inputtuneccacost = model.computeinputloss(tuneData.images1, tuneData.images2)
    inputtrainccacost = model.computeinputloss(trainData.images1, trainData.images2)

    _, outputinittrainccacost, _ = model.computeloss(trainData.images1, trainData.images2)
    _, outputinittuneccacost, _ = model.computeloss(tuneData.images1, tuneData.images2)
    print("input train cca = %12.8f" % (inputtrainccacost))
    print("input val cca = %12.8f" % (inputtuneccacost))
    print("output init train cca =  %12.8f" % (outputinittrainccacost))
    print("output init val cca =  %12.8f" % (outputinittuneccacost))
    model.wind.text(
        "input cca cost = %12.8f<br>input val cca = %12.8f<br>output init train cca =  %12.8f<br>output init val cca = %12.8f" % (
            inputtrainccacost, inputtuneccacost, outputinittrainccacost, outputinittuneccacost), win='init')
    model.wind.line([outputinittrainccacost],  # Y的第一个点的坐标
                    [0],  # X的第一个点的坐标
                    win='cca_loss',  # 窗口的名称
                    name='train',
                    opts=dict(title='cca_loss', showlegend=True, xlabel='epoch', ylabel='cca_loss')  # 图像的标例
                    )
    model.wind.line([outputinittuneccacost],  # Y的第一个点的坐标
                    [0],  # X的第一个点的坐标
                    win='cca_loss',  # 窗口的名称
                    name='val',
                    update='append'
                    )
    model.wind.line([inputtrainccacost],  # Y的第一个点的坐标
                    [0],  # X的第一个点的坐标
                    win='cca_loss',  # 窗口的名称
                    name='train_init',
                    opts=dict(dash=np.array(['dash'])),
                    update='append'
                    )
    model.wind.line([inputtuneccacost],  # Y的第一个点的坐标
                    [0],  # X的第一个点的坐标
                    win='cca_loss',  # 窗口的名称
                    name='val_init',
                    opts=dict(dash=np.array(['dash'])),
                    update='append'
                    )
    # model.loadcheckpoint()
    # Training cycle.
    while epoch < max_epochs:
        print("Current learning rate %f" % lr)
        avg_cost=0.0
        ae_avg_cost = 0.0
        cca_avg_cost = 0.0
        # Loop over all batches.
        NANERROR=False
        if batch_size == 0:
            batch_x1, batch_x2 = trainData.images1, trainData.images2

            # Fit training using batch data.
            ae_avg_cost,cca_avg_cost,avg_cost = model.partial_fit(batch_x1, batch_x2,droprate)
            # Compute average loss.
            if np.isnan(avg_cost):
                NANERROR = True
        else:
            for i in range(total_batch):
                batch_x1, batch_x2, _=trainData.next_batch(batch_size)

                # Fit training using batch data.
                aecost,ccacost,cost=model.partial_fit(batch_x1, batch_x2,droprate)
                print("minibatch %d/%d: cost=%f aecost=%f ccacost=%f" % (i+1, total_batch, cost,aecost,ccacost))

                # Compute average loss.
                if not np.isnan(cost):
                    avg_cost +=cost / n_samples * batch_size
                    ae_avg_cost += aecost / n_samples * batch_size
                    cca_avg_cost += ccacost / n_samples * batch_size
                else:
                    NANERROR=True
                    break
        if NANERROR:
            print("Loss is nan. Reverting to previously saved model ...")
            # saver.restore(model.sess, checkpoint)
            # epoch = model.sess.run(model.epoch)
            # TUNECOST = model.sess.run(model.tunecost)
            return
        # Compute validation error, turn off dropout.
        # FXV1,r1=model.compute_projection(1, tuneData.images1)
        # FXV2,r2=model.compute_projection(2, tuneData.images2)
        # ae_tune_canoncorr,cca_tune_canoncorr=npdccaecoss(FXV1, FXV2, tuneData.images1, tuneData.images2, r1, r2,model.n_z, model.rcov1, model.rcov2)
        # ae_tune_canoncorr = model.ddccaelamda * ae_tune_canoncorr
        # tune_canoncorr = ae_tune_canoncorr + cca_tune_canoncorr
        # TUNECOST[epoch]=tune_canoncorr
        # Display logs per epoch step.
        trainaecost, trainccacost, traincost = model.computeloss(trainData.images1, trainData.images2)
        valaecost, tuneccacost, valcost = model.computeloss(tuneData.images1, tuneData.images2)
        epoch=epoch+1
        print("Epoch: %04d, train regret=%12.8f, tune cost=%12.8f" % (epoch, trainccacost, tuneccacost))
        if epoch == 1:
            model.wind.line([traincost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='loss',  # 窗口的名称
                            name='train',
                            opts=dict(title='loss', showlegend=True, xlabel='epoch', ylabel='loss')  # 图像的标例
                            )
            model.wind.line([trainaecost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='ae_loss',  # 窗口的名称
                            name='train',
                            opts=dict(title='ae_loss', showlegend=True, xlabel='epoch', ylabel='ae_loss')  # 图像的标例
                            )
            model.wind.line([trainccacost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='cca_loss',  # 窗口的名称
                            name='train',
                            update='append'
                            )

            model.wind.line([valcost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='loss',  # 窗口的名称
                            name='val',
                            update='append'
                            )
            model.wind.line([valaecost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='ae_loss',  # 窗口的名称
                            name='val',
                            update='append'
                            )
            model.wind.line([tuneccacost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='cca_loss',  # 窗口的名称
                            name='val',
                            update='append'
                            )
        else:
            model.wind.line([avg_cost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='loss',  # 窗口的名称
                            name='train',
                            update='append'
                            )
            model.wind.line([ae_avg_cost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='ae_loss',  # 窗口的名称
                            name='train',
                            update='append'
                            )
            model.wind.line([trainccacost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='cca_loss',  # 窗口的名称
                            name='train',
                            update='append'
                            )

            model.wind.line([valcost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='loss',  # 窗口的名称
                            name='val',
                            update='append'
                            )
            model.wind.line([valaecost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='ae_loss',  # 窗口的名称
                            name='val',
                            update='append'
                            )
            model.wind.line([tuneccacost],  # Y的第一个点的坐标
                            [epoch],  # X的第一个点的坐标
                            win='cca_loss',  # 窗口的名称
                            name='val',
                            update='append'
                            )
        model.wind.line([inputtrainccacost],  # Y的第一个点的坐标
                        [epoch],  # X的第一个点的坐标
                        win='cca_loss',  # 窗口的名称
                        name='train_init',
                        opts=dict(dash=np.array(['dash'])),
                        update='append'
                        )
        model.wind.line([inputtuneccacost],  # Y的第一个点的坐标
                        [epoch],  # X的第一个点的坐标
                        win='cca_loss',  # 窗口的名称
                        name='val_init',
                        opts=dict(dash=np.array(['dash'])),
                        update='append'
                        )
    # model.savecheckpoint()
    # raise
    return model
