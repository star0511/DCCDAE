import scipy.io as sio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
import gzip
def load_pickle(f):
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret
class DataSet(object):
    
    def __init__(self, images1, images2,y, labels, fake_data=False, one_hot=False,
                 dtype=tf.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                        labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                        labels.shape))
            self._num_examples = images1.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 1
            #images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            if dtype == tf.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images1 = images1.astype(np.float32)

            if dtype == tf.float32 and images2.dtype != np.float32:
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def images1(self):
        return self._images1
    
    @property
    def images2(self):
        return self._images2
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 2048
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        
        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]


class MnistDataSet(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=tf.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.

        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # assert images.shape[3] == 1
            # images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            if dtype == tf.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images1 = images1.astype(np.float32)

            if dtype == tf.float32 and images2.dtype != np.float32:
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 2048
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_image for _ in xrange(batch_size)], [fake_label for _
                                                                                                        in xrange(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]

def readmnist():
    view1 = gzip.open("mnist/noisymnist_view1.gz", 'rb')
    view2 = gzip.open("mnist/noisymnist_view1.gz", 'rb')
    train_set, valid_set, test_set = load_pickle(view1)
    train_set2, valid_set2, test_set2 = load_pickle(view2)
    view1.close()
    view2.close()
    train = MnistDataSet(train_set[0],train_set2[0],train_set[1])
    tune = MnistDataSet(valid_set[0], valid_set[0], valid_set[1])
    test = MnistDataSet(test_set[0], test_set[0], test_set[1])

    return train, tune, test


# def read_traingpds(filename):
#     data = sio.loadmat(filename)
#     partnp = 0.9
#     X1 = np.concatenate((data['X1'],data['XV1'],data['XTe1']))
#     X2 = np.concatenate((data['X2'], data['XV2'],data['XTe2']))
#     y = np.append(np.append(data['Xy'], data['XVy']),data['XTey'])
#     label = np.append(np.append(data['Xlabel'], data['XVlabel']),data['XTelabel'])
#     usernumlist = np.unique(y)
#     sizetrain = 0
#     sizeval = 0
#     for user in usernumlist:
#         usersiglist = np.argwhere(y == user)
#         part = int(usersiglist.shape[0] * partnp)
#         sizetrain = sizetrain + part
#         sizeval = sizeval + usersiglist.shape[0] - part
#     t1 = np.zeros((sizetrain, 2048))
#     t2 = np.zeros((sizetrain, 2048))
#     v1 = np.zeros((sizeval, 2048))
#     v2 = np.zeros((sizeval, 2048))
#     tindex = 0
#     vindex = 0
#     for user in usernumlist:
#         usersiglist = np.argwhere(y == user)
#         np.random.shuffle(usersiglist)
#         part = int(usersiglist.shape[0] * partnp)
#         t1[tindex:tindex + part] = np.squeeze(X1[usersiglist][:part])
#         t2[tindex:tindex + part] = np.squeeze(X2[usersiglist][:part])
#         v1[vindex:vindex + usersiglist.shape[0] - part] = np.squeeze(X1[usersiglist][part:])
#         v2[vindex:vindex + usersiglist.shape[0] - part] = np.squeeze(X2[usersiglist][part:])
#         tindex = tindex + part
#         vindex = vindex + usersiglist.shape[0] - part
#
#     # indexlist =  np.random.permutation(np.arange(X1.shape[0]))
#     # X1 = X1[indexlist]
#     # X2 = X2[indexlist]
#     # part = int(X1.shape[0]*0.5)
#     train = DataSet(t1, t2, y[:sizetrain].T,label[:sizetrain].T)
#
#     tune = DataSet(v1, v2, y[sizetrain:].T,label[sizetrain:].T)
#
#     test = None
#
#     return train, tune, test

def read_traingpds(filename):
    data = sio.loadmat(filename)

    train = DataSet(data['X1'], data['X2'], data['Xy'].T, data['Xlabel'].T)

    tune = DataSet(data['XV1'], data['XV2'], data['XVy'].T, data['XVlabel'].T)

    test = DataSet(data['XTe1'], data['XTe2'], data['XTey'].T, data['XTelabel'].T)

    return train, tune, test
def read_gpds(filename):

    data=sio.loadmat(filename)

    train=DataSet(data['X1'],data['X2'],data['Xy'].T,data['Xlabel'].T)
    
    tune=DataSet(data['XV1'],data['XV2'],data['XVy'].T,data['XVlabel'].T)
    
    test=DataSet(data['XTe1'],data['XTe2'],data['XTey'].T,data['XTelabel'].T)
    
    return train, tune, test


def read_dataset(filename):

    data = sio.loadmat(filename)

    alldata = DataSet(data['X1'], data['X2'], data['Xy'].T, data['Xlabel'].T)

    return alldata

