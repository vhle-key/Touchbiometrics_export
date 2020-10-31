"""The classifier to be used for all touch biometrics dataset.
Author: VH Le
To do:
1) Bundling strokes: Done
2) Combine the CV-hyperparameter-tuning module. with the classifier module.
2) Splits so that y_fake is equally sampled from all users.
3) New train-test split to obtain intersession, intrasession errors: Done

4) Overfit with train set: Change estimator's criteria from EER to ROC AUC? Done. Gives most consistently low EER of all versions.
4) Customise the code (rely less on libraries) to optimize the code.
"""
import warnings
from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC

warnings.filterwarnings(action='ignore')

"""# Define EER scoring functions and helper functions"""


def eer_loss_function(y_true, y_pred_proba):
    """Method 4: Using ROC curve to make an EER function
    EER = max(FRR, FAR), to be applicable no matter if there are thresholds with FAR=FRR or not"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    eer_index = np.argmin(abs(fpr + tpr - 1))  # the point at which FAR is closest to FRR
    # print("EER threshold", thresholds[eer_index], "FAR", fpr[eer_index], "FRR", 1-tpr[eer_index])
    return (fpr[eer_index] + 1 - tpr[eer_index]) / 2


eer_loss = make_scorer(eer_loss_function, greater_is_better=False, needs_threshold=True)


def eer_loss_bundled_function(y_true, y_pred_proba):
    """# Method 4.1: Make EER function with bundled strokes (N consecutive strokes tested together for 1 decision)"""
    y_true_pos, y_pred_proba_pos = y_true[y_true == 1], y_pred_proba[y_true == 1]
    y_true_neg, y_pred_proba_neg = y_true[y_true == 0], y_pred_proba[y_true == 0]
    n_bundle = min(n0_bundle, y_true_pos.shape[0], y_true_neg.shape[0])
    y_true_pos_bundled = np.mean([y_true_pos[i: i + n_bundle] for i in range(y_true_pos.shape[0] - n_bundle + 1)],
                                 axis=1)
    y_true_neg_bundled = np.mean([y_true_neg[i: i + n_bundle] for i in range(y_true_neg.shape[0] - n_bundle + 1)],
                                 axis=1)
    y_pred_proba_pos_bundled = np.mean(
        [y_pred_proba_pos[i: i + n_bundle] for i in range(y_pred_proba_pos.shape[0] - n_bundle + 1)], axis=1)
    y_pred_proba_neg_bundled = np.mean(
        [y_pred_proba_neg[i: i + n_bundle] for i in range(y_pred_proba_neg.shape[0] - n_bundle + 1)], axis=1)

    y_true_bundled = np.concatenate((y_true_pos_bundled, y_true_neg_bundled), axis=0)
    y_pred_proba_bundled = np.concatenate((y_pred_proba_pos_bundled, y_pred_proba_neg_bundled), axis=0)

    # Debugs: See if the new y label array is in correct shape
    # print(y_true.shape, y_true_bundled.shape, y_pred_proba.shape, y_pred_proba_bundled.shape)
    # print(np.unique(y_true, return_counts=True), np.unique(y_true_bundled, return_counts=True))
    return eer_loss_function(y_true_bundled, y_pred_proba_bundled)


eer_loss_bundled = make_scorer(eer_loss_bundled_function, greater_is_better=False, needs_threshold=True)


def choice_or_all(np_array, size):
    """Numpy random choice, but choose whole array if array is less than choose size"""
    if len(np_array) > size:
        return np.random.choice(np_array, size=size, replace=False)
    else:
        return np_array


def get_int(prompt):
    """Helper function to get integer or raise error for invalid input"""
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Needs to be integer")
            continue


def helper():
    """Asks for input parameters of experiments"""
    switcher = {
        1: 'antal_dataset1.csv',
        2: 'antal_dataset2.csv'
    }

    print("----------EXPERIMENT: ANTAL DATASETS------------")
    runs = get_int("Number of runs:")
    while True:
        case = get_int("1 for dataset1, 2 for dataset2:")
        if case not in [1, 2]:
            print("Invalid input, try again")
            continue
        else:
            break
    train_size = get_int("Train size:")
    n0_bundle = get_int("n_bundle:")

    file_name = switcher.get(case, print("Invalid input, try again"))
    feature_names = ['userID', 'strokeDuration', 'xStart', 'yStart', 'xStop', 'yStop', 'end2endDist', 'avgLength',
                     'UDLRflag', 'direction', 'maxDevFromLine', 'avgDirection', 'trajectoryLength', 'avgVelo',
                     'midPressure', 'midArea']
    dataset_train = pd.read_csv(file_name, names=feature_names)
    dataset_train['UDLRflag'] = np.round(dataset_train['UDLRflag'] * 3)  # 0=up, 1=D(if y is reversed?); 2=left, 3=right
    dataset_train = dataset_train[np.isin(dataset_train['UDLRflag'], [2, 3])]

    return train_size, n0_bundle, dataset_train, file_name, runs


if __name__ == "__main__":
    """Constants and initial feature vectors of all strokes"""
    t_start = time()
    n_fake_user = 10
    train_size, n0_bundle, dataset_train, file_name, runs = helper()
    col_feats = ['midArea', 'midPressure', 'xStart', 'yStart', 'xStop', 'yStop', 'direction', 'avgDirection', 'avgVelo',
                 'trajectoryLength', 'strokeDuration', 'end2endDist', 'avgLength', 'maxDevFromLine']

    X_train_all = scale(dataset_train[col_feats])
    y_train_all = dataset_train['userID'].to_numpy()
    X_test_all, y_test_all = X_train_all.copy(), y_train_all.copy()
    user_id_range = np.unique(y_train_all)
    print("-----------BENCHMARKING FOR DATASET %r------------" % file_name)
    print("-----------TRAIN SIZE %i, n_BUNDLE %i-----------" % (train_size, n0_bundle))
    print("Counts of userID",
          "Train", np.unique(y_train_all, return_counts=True),
          "Test", np.unique(y_test_all, return_counts=True))

    users_eerb_all_seed = []
    for seed in range(runs):
        np.random.seed();
        print("-----------RUN NUMBER: %i --------------" % seed)
        users_eerb = []

        # Turns data into binary classifier
        for (count, real_user_id) in enumerate(user_id_range):
            other_user_id = np.array([id for id in user_id_range if id != real_user_id])

            # Samples train data from one user (Genuine) and equally from n_fake_user of other users (Imposter)
            if np.count_nonzero(y_train_all == real_user_id) <= train_size:
                continue
            train_real_index = choice_or_all(np.where(y_train_all == real_user_id)[0], size=train_size)

            t0 = time()
            other_user_id_select = np.random.choice(other_user_id, size=n_fake_user, replace=False)
            train_size_small = max(1, train_size // n_fake_user)
            train_fake_index = np.hstack([choice_or_all(np.where(y_train_all == id)[0], size=train_size_small)
                                          for id in other_user_id_select])
            train_index = np.concatenate((train_real_index, train_fake_index), axis=0)
            X_train, y_train = X_train_all[train_index], (y_train_all[train_index] == real_user_id)

            # Test data is the rest of dataset
            test_index = np.delete(np.arange(dataset_train.shape[0]), train_index)
            X_test, y_test = X_test_all[test_index], (y_test_all[test_index] == real_user_id)
            print("User id", real_user_id, "Other", other_user_id_select,
                  "Train labels:", np.unique(y_train, return_counts=True), "Test labels:",
                  np.unique(y_test, return_counts=True))

            """Cross validation to find best hyperparamters"""
            clf_test_scores = []
            t0 = time()
            param_grid = [{'C': np.logspace(0, 3, num=5), 'gamma': np.logspace(-3, 0, num=5)}]
            scoring = {'EER': eer_loss}

            clf = GridSearchCV(SVC(class_weight='balanced'), param_grid,
                               cv=StratifiedKFold(3), scoring=scoring, refit='EER')
            clf = clf.fit(X_train, y_train)
            eerb = -100 * eer_loss_bundled(clf, X_test, y_test)
            users_eerb.append(eerb)
            print("EER bundled for user", real_user_id, "done in %0.3fs" % (time() - t0),
                  "Best parameters set found on development set, test score:", clf.best_params_, clf.best_score_,
                  eer_loss_bundled(clf, X_test, y_test))
            print()

        print("All users EERb", users_eerb)
        print("Average %.5f" % (np.mean(users_eerb)),
              "Median %.5f" % (np.median(users_eerb)))
        print("All done in %.3f s" % (time() - t_start))

        print("Average EER run", seed, np.mean(users_eerb))
        print("Median EER run", seed, np.median(users_eerb))
        users_eerb_all_seed.append(users_eerb)

    print("Users average and median EER", np.mean(users_eerb_all_seed, axis=0).tolist())
    print("SD EER of users", np.std(users_eerb_all_seed, axis=0).tolist())

    print("Across all users average and median EER", np.mean(users_eerb_all_seed), np.median(users_eerb_all_seed))
    print("Average SD EER across all users", np.mean(np.std(users_eerb_all_seed, axis=0)))
