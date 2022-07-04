#!/usr/bin/env python
# coding=utf-8


from sklearn.svm import SVC
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os
from typing import Optional, Union
from multiprocessing import Pool, Manager
rng = np.random.RandomState(1234)
from visdom import Visdom
import numpy as np
import sklearn.metrics as sk_metrics
from typing import List, Tuple, Dict


def compute_metrics(genuine_preds: List[np.ndarray],
                    random_preds: List[np.ndarray],
                    skilled_preds: List[np.ndarray],
                    global_threshold: float) -> Dict:
    """ Compute metrics given the predictions (scores) of genuine signatures,
    random forgeries and skilled forgeries.

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    random_preds: list of np.ndarray
        A list of predictions of random forgeries (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)
    global_threshold: float
        The global threshold used to compute false acceptance and false rejection rates

    Returns
    -------
    dict
        A dictionary containing:
        'FRR': false rejection rate
        'FAR_random': false acceptance rate for random forgeries
        'FAR_skilled': false acceptance rate for skilled forgeries
        'mean_AUC': mean Area Under the Curve (average of AUC for each user)
        'EER': Equal Error Rate using a global threshold
        'EER_userthresholds': Equal Error Rate using user-specific thresholds
        'auc_list': the list of AUCs (one per user)
        'global_threshold': the optimum global threshold (used in EER)
    """
    all_genuine_preds = np.concatenate(genuine_preds)
    all_random_preds = np.concatenate(random_preds)
    all_skilled_preds = np.concatenate(skilled_preds)

    FRR = 1 - np.mean(all_genuine_preds >= global_threshold)
    FAR_random = 1 - np.mean(all_random_preds < global_threshold)
    FAR_skilled = 1 - np.mean(all_skilled_preds < global_threshold)

    aucs, meanAUC = compute_AUCs(genuine_preds, skilled_preds)

    EER, global_threshold = compute_EER(all_genuine_preds, all_skilled_preds)
    EER_userthresholds = calculate_EER_user_thresholds(genuine_preds, skilled_preds)
    ROC = compute_ROC(genuine_preds, skilled_preds)

    all_metrics = {'FRR': FRR,
                   'FAR_random': FAR_random,
                   'FAR_skilled': FAR_skilled,
                   'mean_AUC': meanAUC,
                   'EER': EER,
                   'EER_userthresholds': EER_userthresholds,
                   'auc_list': aucs,
                   'global_threshold': global_threshold,
                   'ROC':ROC}

    return all_metrics


def compute_AUCs(genuine_preds: List[np.ndarray],
                 skilled_preds: List[np.ndarray]) -> Tuple[List[float], float]:
    """ Compute the area under the curve for the classifiers

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)

    Returns
    -------
    list
        The list of AUCs (one per user)
    float
        The mean AUC

    """
    aucs = []
    for thisRealPreds, thisSkilledPreds in zip(genuine_preds, skilled_preds):
        y_true = np.ones(len(thisRealPreds) + len(thisSkilledPreds))
        y_true[len(thisRealPreds):] = -1
        y_scores = np.concatenate([thisRealPreds, thisSkilledPreds])
        aucs.append(sk_metrics.roc_auc_score(y_true, y_scores))
    meanAUC = np.mean(aucs)
    return aucs, meanAUC.item()

def compute_ROC(genuine_preds: List[np.ndarray],
                                  skilled_preds: List[np.ndarray]) -> float:
    """ 计算FAR和FRR在不同user阈值下的分布

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)

    Returns
    -------
    list of np.ndarray
        不同阈值下far和frr的值

    """
    all_genuine_errors = []
    all_skilled_errors = []

    nRealPreds = 0
    nSkilledPreds = 0

    for this_real_preds, this_skilled_preds in zip(genuine_preds, skilled_preds):
        user_all_genuine_errors=[]
        user_all_skilled_errors=[]
        nRealPreds = len(this_real_preds)
        nSkilledPreds = len(this_skilled_preds)
        for t in np.arange(-1,1,0.01):
            genuineErrors = np.sum(this_real_preds < t)
            skilledErrors = np.sum(this_skilled_preds >= t)
            user_all_genuine_errors.append(genuineErrors)
            user_all_skilled_errors.append(skilledErrors)
        all_genuine_errors.append(user_all_genuine_errors)
        all_skilled_errors.append(user_all_skilled_errors)
        nRealPreds += len(this_real_preds)
        nSkilledPreds += len(this_skilled_preds)

    genuineErrors = np.mean(all_genuine_errors,axis=0) / nRealPreds
    skilledErrors = np.mean(all_skilled_errors,axis=0) / nSkilledPreds

    return [genuineErrors,skilledErrors]
def compute_EER(all_genuine_preds: np.ndarray,
                all_skilled_preds: np.ndarray) -> Tuple[float, float]:
    """ Calculate Equal Error Rate with a global decision threshold.

    Parameters
    ----------
    all_genuine_preds: np.ndarray
        Scores for genuine predictions of all users
    all_skilled_preds: np.ndarray
    Scores for skilled forgery predictions of all users

    Returns
    -------
    float:
        The Equal Error Rate
    float:
        The optimum global threshold (a posteriori)

    """

    all_preds = np.concatenate([all_genuine_preds, all_skilled_preds])
    all_ys = np.concatenate([np.ones_like(all_genuine_preds), np.ones_like(all_skilled_preds) * -1])
    fpr, tpr, thresholds = sk_metrics.roc_curve(all_ys, all_preds)

    # Select the threshold closest to (FPR = 1 - TPR)
    t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]
    genuineErrors = 1 - np.mean(all_genuine_preds >= t).item()
    skilledErrors = 1 - np.mean(all_skilled_preds < t).item()
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER, t


def calculate_EER_user_thresholds(genuine_preds: List[np.ndarray],
                                  skilled_preds: List[np.ndarray]) -> float:
    """ Calculate Equal Error Rate with a decision threshold specific for each user

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)

    Returns
    -------
    float
        The Equal Error Rate when using user-specific thresholds

    """
    all_genuine_errors = []
    all_skilled_errors = []

    nRealPreds = 0
    nSkilledPreds = 0

    for this_real_preds, this_skilled_preds in zip(genuine_preds, skilled_preds):
        # Calculate user AUC
        y_true = np.ones(len(this_real_preds) + len(this_skilled_preds))
        y_true[len(this_real_preds):] = -1
        y_scores = np.concatenate([this_real_preds, this_skilled_preds])

        # Calculate user threshold
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_scores)
        # Select the threshold closest to (FPR = 1 - TPR).
        index = sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]
        t = thresholds[index]

        genuineErrors = np.sum(this_real_preds < t)
        skilledErrors = np.sum(this_skilled_preds >= t)

        all_genuine_errors.append(genuineErrors)
        all_skilled_errors.append(skilledErrors)

        nRealPreds += len(this_real_preds)
        nSkilledPreds += len(this_skilled_preds)

    genuineErrors = float(np.sum(all_genuine_errors)) / nRealPreds
    skilledErrors = float(np.sum(all_skilled_errors)) / nSkilledPreds

    # Errors should be nearly equal, up to a small rounding error since we have few examples per user.
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER

def main(args,filename):
    data = sio.loadmat(os.path.dirname(os.path.abspath(__file__))+"/"+filename)
    data = data[args.datasetname]
    if(args.datasetname == "gpds"):
        if(args.gpdssize==160):
            exp_users = range(0, 205)
            dev_users = range(205, 961)
        elif(args.gpdssize==300):
            exp_users = range(0, 352)
            dev_users = range(352,961)
        else:
            exp_users = range(352, 403)
            dev_users = range(352, 403)
        features = np.concatenate((data['development'][0][0].real, data['validation'][0][0].real, data['exploitation'][0][0].real))
        y = np.concatenate((data['devy'][0][0].ravel(), data['valy'][0][0].ravel(), data['expy'][0][0].ravel()))
        yfrog = np.concatenate((data['devlabel'][0][0].ravel(), data['vallabel'][0][0].ravel(), data['explabel'][0][0].ravel()))
        for i in range(len(yfrog)):
            if(yfrog[i] == 0):
                yfrog[i] = 1
            else:
                yfrog[i] = 0
    else:
        if(args.datasetname == "mcyt"):
            exp_users = range(0, 120)
            dev_users = range(0, 120)
        elif(args.datasetname == "cedar"):
            exp_users = range(0, 56)
            dev_users = range(0, 56)
        else:
            exp_users = range(0, 61)
            dev_users = range(61, 169)
        features = data['development'][0][0].real
        y = data['devy'][0][0].ravel()
        yfrog = data['devlabel'][0][0].ravel()
        for i in range(len(yfrog)):
            if (yfrog[i] == 0):
                yfrog[i] = 1
            else:
                yfrog[i] = 0
    data = (features,y,yfrog)
    exp_set = get_subset(data, exp_users)
    dev_set = get_subset(data, dev_users)

    result = []
    eer_u_list = []
    eer_list = []
    all_results = []
    frr_list = []
    far_random_list = []
    far_skilled_list = []
    mean_auc_list = []
    ROCFAR = []
    ROCFRR = []

    if args.datasetname=="gpds":
        forg_from_dev = 14
        gen_for_test = 10
        if args.trainsvmuser == -1:
            gen_for_train = 12
        else:
            gen_for_train = args.trainsvmuser
    elif args.datasetname=="brazilian":
        forg_from_dev = 30
        gen_for_test = 10
        if args.trainsvmuser == -1:
            gen_for_train = 30
        else:
            gen_for_train = args.trainsvmuser
    elif args.datasetname=="cedar":
        forg_from_dev = 12
        gen_for_test = 10
        if args.trainsvmuser == -1:
            gen_for_train = 12
        else:
            gen_for_train = args.trainsvmuser
    elif args.datasetname=="mcyt":
        forg_from_dev = 10
        gen_for_test = 5
        if args.trainsvmuser == -1:
            gen_for_train = 10
        else:
            gen_for_train = args.trainsvmuser
    # rng = np.random.RandomState(1234)

    p = Pool()
    for i in range(args.folds):
        result.append([p.apply_async(nultitrain, args=(args.datasetname,gen_for_train,gen_for_test,forg_from_dev,exp_set,dev_set,args,43*i))])#43

    p.close()
    p.join()
    for results in result:
        results = results[0]._value
        this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)
        frr_list.append(results['all_metrics']['FRR'])
        far_random_list.append(results['all_metrics']['FAR_random'])
        far_skilled_list.append(results['all_metrics']['FAR_skilled'])
        mean_auc_list.append(results['all_metrics']['mean_AUC'])
        ROCFAR.append(results['all_metrics']['ROC'][0])
        ROCFRR.append(results['all_metrics']['ROC'][1])
    ROCFAR = np.mean(ROCFAR,axis=0)
    ROCFRR = np.mean(ROCFRR,axis=0)
    # result.append([nultitrain(
    # args.datasetname, gen_for_train, gen_for_test, forg_from_dev, exp_set, dev_set, args, 43 * i)])  # 43
    #
    # for results in result:
    #     results = results[0]
    #     this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
    #     all_results.append(results)
    #     eer_u_list.append(this_eer_u)
    #     eer_list.append(this_eer)
    #     frr_list.append(results['all_metrics']['FRR'])
    #     far_random_list.append(results['all_metrics']['FAR_random'])
    #     far_skilled_list.append(results['all_metrics']['FAR_skilled'])
    #     mean_auc_list.append(results['all_metrics']['mean_AUC'])
    #     ROC.append(results['all_metrics']['ROC'])
    print('FRR: {:.2f} (+- {:.2f})'.format(np.mean(frr_list) * 100, np.std(frr_list) * 100))
    print('FAR_random: {:.2f} (+- {:.2f})'.format(np.mean(far_random_list) * 100, np.std(far_random_list) * 100))
    print('FAR_skilled: {:.2f} (+- {:.2f})'.format(np.mean(far_skilled_list) * 100, np.std(far_skilled_list) * 100))
    print('mean_AUC: {:.2f} (+- {:.2f})'.format(np.mean(mean_auc_list), np.std(mean_auc_list)))
    print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))
    viz = Visdom(env=filename.replace("2_viewdata/", ""), port=2333)
    viz.text(str(args)+'<br>FRR: {:.2f} (+- {:.2f})<br>'.format(np.mean(frr_list) * 100, np.std(
        frr_list) * 100) + 'FAR_random: {:.2f} (+- {:.2f})<br>'.format(np.mean(far_random_list) * 100, np.std(
        far_random_list) * 100) + 'FAR_skilled: {:.2f} (+- {:.2f})<br>'.format(np.mean(far_skilled_list) * 100, np.std(
        far_skilled_list) * 100) + 'mean_AUC: {:.2f} (+- {:.2f})<br>'.format(np.mean(mean_auc_list), np.std(
        mean_auc_list)) + 'EER (global threshold): {:.2f} (+- {:.2f})<br>'.format(np.mean(eer_list) * 100, np.std(
        eer_list) * 100) + 'EER (user thresholds): {:.2f} (+- {:.2f})<br>'.format(np.mean(eer_u_list) * 100,
                                                                                  np.std(eer_u_list) * 100),
             win=str(args))
    fnew = open("eer"+args.svm_type+str(args.gpuid)+".txt", "a", encoding="utf-8")
    fnew.write('\n\n\n' + str(args) + '\n')
    fnew.write('FRR: {:.2f} (+- {:.2f})'.format(np.mean(frr_list) * 100, np.std(frr_list) * 100) + '\n')
    fnew.write('FAR_random: {:.2f} (+- {:.2f})'.format(np.mean(far_random_list) * 100, np.std(far_random_list) * 100) + '\n')
    fnew.write('FAR_skilled: {:.2f} (+- {:.2f})'.format(np.mean(far_skilled_list) * 100, np.std(far_skilled_list) * 100) + '\n')
    fnew.write('mean_AUC: {:.2f} (+- {:.2f})'.format(np.mean(mean_auc_list), np.std(mean_auc_list)) + '\n')
    fnew.write(
        'EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100) + '\n')
    fnew.write(
        'EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100) + '\n')
    fnew.write(str(['FAR{:.4f}'.format(i) for i in ROCFAR]))
    fnew.write(str(['FRR{:.4f}'.format(i) for i in ROCFRR]))
    fnew.close()
    # if args.save_path is not None:
    #     print('Saving results to {}'.format(args.save_path))
    #     with open(args.save_path, 'wb') as f:
    #         pickle.dump(all_results, f)
    return all_results
def nultitrain(datasetname,gen_for_train,gen_for_test,forg_from_dev,exp_set,dev_set,args,rngunum):
    rng = np.random.RandomState(rngunum)
    classifiers, results = train_test_all_users(datasetname,exp_set,
                                                dev_set,
                                                svm_type=args.svm_type,
                                                C=args.svm_c,
                                                gamma=args.svm_gamma,
                                                num_gen_train=gen_for_train,
                                                num_forg_from_exp=0,
                                                num_forg_from_dev=forg_from_dev,
                                                num_gen_test=gen_for_test,
                                                rng=rng)
    return results
def train_wdclassifier_user(training_set: Tuple[np.ndarray, np.ndarray],
                            svmType: str,
                            C: float,
                            gamma: Optional[float]) -> sklearn.svm.SVC:
    """ Trains an SVM classifier for a user

    Parameters
    ----------
    training_set: Tuple (x, y)
        The training set (features and labels). y should have labels -1 and 1
    svmType: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel

    Returns
    -------
    sklearn.svm.SVC:
        The learned classifier

    """

    assert svmType in ['linear', 'rbf']

    train_x = training_set[0]
    train_y = training_set[1]

    # Adjust for the skew between positive and negative classes
    n_genuine = len([x for x in train_y if x == 1])
    n_forg = len([x for x in train_y if x == -1])
    skew = n_forg / float(n_genuine)

    # Train the model
    if svmType == 'rbf':
        model = SVC(C=C, gamma=gamma, class_weight={1: skew})
    else:
        model = SVC(kernel='linear', C=C, class_weight={1: skew})

    model_with_scaler = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                           ('classifier', model)])

    model_with_scaler.fit(train_x, train_y)

    return model_with_scaler


def test_user(model: sklearn.svm.SVC,
              genuine_signatures: np.ndarray,
              random_forgeries: np.ndarray,
              skilled_forgeries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Test the WD classifier of an user

    Parameters
    ----------
    model: sklearn.svm.SVC
        The learned classifier
    genuine_signatures: np.ndarray
        Genuine signatures for test
    random_forgeries: np.ndarray
        Random forgeries for test (signatures from other users)
    skilled_forgeries: np.ndarray
        Skilled forgeries for test

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        The predictions(scores) for genuine signatures,
        random forgeries and skilled forgeries

    """
    # Get predictions
    genuinePred = model.decision_function(genuine_signatures)
    randomPred = model.decision_function(random_forgeries)
    skilledPred = model.decision_function(skilled_forgeries)

    return genuinePred, randomPred, skilledPred


def train_all_users(exp_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    svm_type: str,
                    C: float,
                    gamma: float,
                    num_forg_from_dev: int,
                    num_forg_from_exp: int,
                    rng: np.random.RandomState) -> Dict[int, sklearn.svm.SVC]:
    """ Train classifiers for all users in the exploitation set

    Parameters
    ----------
    exp_train: tuple of np.ndarray (x, y, yforg)
        The training set split of the exploitation set (system users)
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development set
    svm_type: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel
    num_forg_from_dev: int 0
        Number of forgeries from each user in the development set to
        consider as negative samples
    num_forg_from_exp: int 14
        Number of forgeries from each user in the exploitation set (other
        than the current user) to consider as negative sample.
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    Dict int -> sklearn.svm.SVC
        A dictionary of trained classifiers, where the keys are the users.

    """
    classifiers = {}
    # num_forg_from_dev：14
    # num_forg_from_exp：0
    exp_y = exp_train[1]
    users = np.unique(exp_y)
    # num_forg_from_dev:14
    if num_forg_from_dev > 0:
        # n*user总数幅度原始图像矩阵w*h
        other_negatives = get_random_forgeries_from_dev(dev_set, num_forg_from_dev, rng)
    else:
        other_negatives = []
    # 进度条
    for user in tqdm(users):
        # 创建训练的集合
        training_set = create_training_set_for_user(user, exp_train, num_forg_from_exp, other_negatives, rng)
        classifiers[user] = train_wdclassifier_user(training_set, svm_type, C, gamma)

    return classifiers


def test_all_users(classifier_all_user: Dict[int, sklearn.svm.SVC],
                   exp_test: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   global_threshold: float) -> Dict:
    """ Test classifiers for all users and return the metrics

    Parameters
    ----------
    classifier_all_user: dict (int -> sklearn.svm.SVC)
        The trained classifiers for all users
    exp_test: tuple of np.ndarray (x, y, yforg)
        The testing set split from the exploitation set
    global_threshold: float
        The threshold used to compute false acceptance and
        false rejection rates

    Returns
    -------
    dict
        A dictionary containing a variety of metrics, including
        false acceptance and rejection rates, equal error rates

    """
    xfeatures_test, y_test, yforg_test = exp_test

    genuinePreds = []
    randomPreds = []
    skilledPreds = []

    users = np.unique(y_test)
    for user in users:
        model = classifier_all_user[user]

        # Test the performance for the user without replicates
        skilled_forgeries = xfeatures_test[(y_test == user) & (yforg_test == 1)]
        rng.shuffle(skilled_forgeries)
        test_genuine = xfeatures_test[(y_test == user) & (yforg_test == 0)]
        rng.shuffle(test_genuine)
        random_forgeries = xfeatures_test[(y_test != user) & (yforg_test == 0)]
        rng.shuffle(random_forgeries)
        genuinePredUser = model.decision_function(test_genuine)
        skilledPredUser = model.decision_function(skilled_forgeries)
        randomPredUser = model.decision_function(random_forgeries)

        genuinePreds.append(genuinePredUser)
        skilledPreds.append(skilledPredUser)
        randomPreds.append(randomPredUser)

    # Calculate al metrics (EER, FAR, FRR and AUC)
    all_metrics = compute_metrics(genuinePreds, randomPreds, skilledPreds, global_threshold)

    results = {'all_metrics': all_metrics,
               'predictions': {'genuinePreds': genuinePreds,
                               'randomPreds': randomPreds,
                               'skilledPreds': skilledPreds}}

    print(all_metrics['EER'], all_metrics['EER_userthresholds'])
    return results


def train_test_all_users(datasetname:str,
                         exp_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         svm_type: str,
                         C: float,
                         gamma: float,
                         num_gen_train: int,
                         num_forg_from_exp: int,
                         num_forg_from_dev: int,
                         num_gen_test: int,
                         global_threshold: float = 0,
                         rng: np.random.RandomState = np.random.RandomState()) \
        -> Tuple[Dict[int, sklearn.svm.SVC], Dict]:
    """ Train and test classifiers for every user in the exploitation set,
        and returns the metrics.

    Parameters
    ----------
    exp_set: tuple of np.ndarray (x, y, yforg)
        The exploitation set
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development set
    svm_type: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel


    num_gen_train: int   gen_for_train
        Number of genuine signatures available for training
    num_forg_from_dev: int  forg_from_dev
        Number of forgeries from each user in the development set to
        consider as negative samples
    num_forg_from_exp: int  forg_from_exp
        Number of forgeries from each user in the exploitation set (other
        than the current user) to consider as negative sample.
    num_gen_test: int  gen_for_test
        Number of genuine signatures for testing
    global_threshold: float
        The threshold used to compute false acceptance and
        false rejection rates
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    dict (int -> sklearn.svm.SVC)
        The classifiers for all users

    dict
        A dictionary containing a variety of metrics, including
        false acceptance and rejection rates, equal error rates

    """
    # exp_train 12*真实签名
    # exp_test 10*真实签名+假签名
    # x,y,frog
    exp_train, exp_test = split_train_test(datasetname,exp_set, num_gen_train, num_gen_test, rng)

    classifiers = train_all_users(exp_train, dev_set, svm_type, C, gamma,
                                  num_forg_from_dev, num_forg_from_exp, rng)

    results = test_all_users(classifiers, exp_test, global_threshold)

    return classifiers, results

# ---------------------------------------data
def split_train_test(datasetname:str,
                     exp_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     num_gen_train: int,
                     num_gen_test: int,
                     rng: np.random.RandomState) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                          Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ Splits a set into training and testing. Both sets contains the same users. The
        training set contains only genuine signatures, while the testing set contains
        genuine signatures and forgeries. Note that the number of genuine signatures for
        training plus the number of genuine signatures for test must be smaller or equal to
        the total number of genuine signatures (to ensure no overlap)

    Parameters
    ----------
    datasetname:String
        数据集名称
    exp_set: tuple of np.ndarray (x, y, yforg)
        The dataset
    num_gen_train: int 12
        The number of genuine signatures to be used for training
    num_gen_test: int 10
        The number of genuine signatures to be used for testing
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    tuple of np.ndarray (x, y, yforg)c
        The training set

    tuple of np.ndarray (x, y, yforg)
        The testing set
    """
    x, y, yforg = exp_set
    users = np.unique(y)

    train_idx = []
    test_idx = []
    #####test集合需要重组
    for user in users:
        user_genuines = np.flatnonzero((y == user) & (yforg == False))
        rng.shuffle(user_genuines)
        user_train_idx = user_genuines[0:num_gen_train]
        user_test_idx = user_genuines[-num_gen_test:]

        # Sanity check to ensure training samples are not used in test:
        assert len(set(user_train_idx).intersection(user_test_idx)) == 0

        train_idx += user_train_idx.tolist()
        test_idx += user_test_idx.tolist()
        if datasetname == "mcyt":
            user_forgeries = np.flatnonzero((y == user) & (yforg == True))
            test_idx += user_forgeries.tolist()
        elif datasetname == "brazilian":
            user_forgeries = np.flatnonzero((y == user) & (yforg == True))
            test_idx += user_forgeries.tolist()[-10:]
        else:
            user_forgeries = np.flatnonzero((y == user) & (yforg == True))
            rng.shuffle(user_forgeries)
            test_idx += user_forgeries.tolist()[-10:]


    exp_train = x[train_idx], y[train_idx], yforg[train_idx]
    exp_test = x[test_idx], y[test_idx], yforg[test_idx]

    return exp_train, exp_test
def get_random_forgeries_from_dev(dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  num_forg_from_dev: int,
                                  rng: np.random.RandomState):
    """ Obtain a set of random forgeries form a development set (to be used
        as negative samples)

    Parameters
    ----------
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development dataset
    num_forg_from_dev: int
        The number of random forgeries (signatures) from each user in the development
        set to be considered
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    np.ndarray (N x M)
        The N negative samples (M is the dimensionality of the feature set)

    """

    x, y, yforg = dev_set
    users = np.unique(y)

    random_forgeries = []
    ylist = []
    for user in users:
        idx = np.flatnonzero((y == user) & (yforg == False))
        chosen_idx = rng.choice(idx, num_forg_from_dev, replace=False)
        random_forgeries.append(x[chosen_idx])
        ylist.append(user)

    return random_forgeries,ylist
def create_training_set_for_user(user: int,
                                 exp_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 num_forg_from_exp: int,
                                 other_negatives: np.ndarray,
                                 rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """ Creates a training set for training a WD classifier for a user

    Parameters
    ----------
    user: int
        The user for which a dataset will be created
    exp_train: tuple of np.ndarray (x, y, yforg)
        The training set split of the exploitation dataset
    num_forg_from_exp: int
        The number of random forgeries from each user in the exploitation set
        (other than "user") that will be used as negatives
    other_negatives: np.ndarray
        A collection of other negative samples (e.g. from a development set)
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    np.ndarray (N), np.ndarray (N)
        The dataset for the user (x, y), where N is the number of signatures
        (genuine + random forgeries)
    """
    exp_x, exp_y, exp_yforg = exp_train

    positive_samples = exp_x[(exp_y == user) & (exp_yforg == 0)]
    if num_forg_from_exp > 0:
        users = np.unique(exp_y)
        other_users = list(set(users).difference({user}))
        negative_samples_from_exp = []
        for other_user in other_users:
            idx = np.flatnonzero((exp_y == other_user) & (exp_yforg == False))
            chosen_idx = rng.choice(idx, num_forg_from_exp, replace=False)
            negative_samples_from_exp.append(exp_x[chosen_idx])
        negative_samples_from_exp = np.concatenate(negative_samples_from_exp)
    else:
        negative_samples_from_exp = []

    if len(other_negatives[0]) > 0 and len(negative_samples_from_exp) > 0:
        negative_samples = np.concatenate((negative_samples_from_exp, other_negatives[0]))
    elif len(other_negatives[0]) > 0:
        # 这部分代码是为了删除自己的特征
        other_features,y = other_negatives
        idx = np.flatnonzero((y != user)).tolist()
        other_features_noself = []
        for i in range(len(other_features)):
            if(i in idx):
                other_features_noself.append(other_features[i])
        negative_samples = np.concatenate(other_features_noself)#[idx]

    elif len(negative_samples_from_exp) > 0:
        negative_samples = negative_samples_from_exp
    else:
        raise ValueError('Either random forgeries from exploitation or from development sets must be used')

    train_x = np.concatenate((positive_samples, negative_samples))
    train_y = np.concatenate((np.full(len(positive_samples), 1),
                              np.full(len(negative_samples), -1)))

    return train_x, train_y
def get_subset(data: Tuple[np.ndarray, ...],
               subset: Union[list, range],
               y_idx: int = 1) -> Tuple[np.ndarray, ...]:
    """ Gets a data for a subset of users (the second array in data)

    Parameters
    ----------
    data: Tuple (x, y, ...)
        The dataset
    subset: list
        The list of users to include
    y_idx: int
        The index in data that refers to the users (usually index=1)

    Returns
    -------
    Tuple (x, y , ...)
        The dataset containing only data from users in the subset

    """
    to_include = np.isin(data[y_idx], subset)

    return tuple(d[to_include] for d in data)