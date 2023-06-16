import numpy as np
from spider import SPIDER
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)

# datasets = ['datasets/glass1', 'datasets/yeast1', 'datasets/ecoli1, 'datasets/ecoli2', 'datasets/glass6',
#             'datasets/yeast3', 'datasets/ecoli3', 'datasets/yeast-2_vs_4', 'datasets/glass2', 'datasets/ecoli4',
#             'datasets/glass-0-1-6_vs_5', 'datasets/glass5', 'datasets/yeast4', 'datasets/yeast6']

# datasets = ['datasets/glass1', 'datasets/glass6']
datasets = ['datasets/glass1']

classifiers = {
    'DT': DecisionTreeClassifier(),
    'MNB': MultinomialNB(),
    'KNN': KNeighborsClassifier(),
}

preprocs = {
    'none': None,
    'smote': SMOTE(),
    'spider': SPIDER(amplification_type='strong_amplification'),
    #    'spiderw': SPIDER(amplification_type='weak_amplification'),
    #    'adasyn': ADASYN(),
    #    'bsmote': BorderlineSMOTE(),
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

scores = np.zeros(shape=(len(classifiers), len(datasets), len(preprocs), rskf.get_n_splits(), len(metrics)))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("%s.csv" % dataset, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for preproc_id, preproc in enumerate(preprocs):
            if preprocs[preproc] is None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = preprocs[preproc].fit_resample(
                    X[train], y[train])

            for clf_id, clf_prot in enumerate(classifiers):
                clf = clone(classifiers[clf_prot])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X[test])

                for metric_id, metric in enumerate(metrics):
                    scores[clf_id, data_id, preproc_id, fold_id, metric_id] = metrics[metric](
                        y[test], y_pred)

np.save("scores", scores)

print("Scores:\n", scores)
print("\nScores shape:\n", scores.shape)
print("Classifiers, Datasets, Preprocessing methods, Folds, Metrics")

mean_scores = np.mean(scores, axis=3)
np.save("mean_scores", mean_scores)

print("\nMean scores:\n", mean_scores)
print("\nMean Scores shape:\n", mean_scores.shape)
print("Classifiers, Datasets, Preprocessing methods, Metrics")

