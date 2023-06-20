import numpy as np
import plotly.graph_objects as go
from scipy.stats import ttest_ind, rankdata, ranksums

from spider import SPIDER
from imblearn.over_sampling import SMOTE
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)

datasets = ['datasets/glass1', 'datasets/yeast1', 'datasets/ecoli1', 'datasets/ecoli2', 'datasets/glass6',
            'datasets/yeast3', 'datasets/ecoli3', 'datasets/yeast-2_vs_4', 'datasets/glass2', 'datasets/ecoli4',
            'datasets/glass-0-1-6_vs_5', 'datasets/glass5', 'datasets/yeast4', 'datasets/yeast6']

# szybkie do testowania:
# datasets = ['datasets/glass1', 'datasets/glass6', 'datasets/ecoli3', 'datasets/yeast4', 'datasets/ecoli2']

classifiers = {
    'DT': DecisionTreeClassifier(),
    'MNB': MultinomialNB(),
    'KNN': KNeighborsClassifier(),
}

preprocs = {
    'none': None,
    'smote': SMOTE(k_neighbors=3),
    'spiderW': SPIDER(amplification_type='weak_amplification'),
    'spiderWR': SPIDER(amplification_type='weak_amplification_and_relabeling'),
    'spiderS': SPIDER(amplification_type='strong_amplification'),
}

metrics = {
    "Recall": recall,
    'Precision': precision,
    'Specificity': specificity,
    'F1 score': f1_score,
    'G-mean': geometric_mean_score_1,
    'BAC': balanced_accuracy_score,
}

scores = np.zeros(shape=(len(classifiers), len(datasets), len(preprocs), rskf.get_n_splits(), len(metrics)))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("%s.csv" % dataset, delimiter=",")
    X = dataset[:, :-1]
    y = np.array(dataset[:, -1].astype(int))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for preproc_id, preproc in enumerate(preprocs):
            if preprocs[preproc] is None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = preprocs[preproc].fit_resample(X[train], y[train])

            for clf_id, clf_prot in enumerate(classifiers):
                clf = clone(classifiers[clf_prot])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X[test])

                for metric_id, metric in enumerate(metrics):
                    scores[clf_id, data_id, preproc_id, fold_id, metric_id] = metrics[metric](y[test], y_pred)

# PRINTOWANIE MACIERZY WYNIKOWYCH:

# print("Scores:\n", scores)
# print("\nScores shape:\n", scores.shape)
# print("Classifiers, Datasets, Preprocessing methods, Folds, Metrics")

mean_scores = np.mean(scores, axis=3)
# print("\nMean scores:\n", mean_scores)
# print("\nMean Scores shape:\n", mean_scores.shape)
# print("Classifiers, Datasets, Preprocessing methods, Metrics")

mean_mean_scores = np.mean(mean_scores, axis=1)
# print("\nMean mean scores:\n", mean_mean_scores)
# print("\nMean mean Scores shape:\n", mean_mean_scores.shape)
# print("Classifiers, Preprocessing methods, Metrics")

metrics_list = list(metrics.keys())
clf_list = list(classifiers.keys())
preprocs_list = list(preprocs.keys())

# ŁADNE PRINTOWANIE METRYK:

# for i in range(len(clf_list)):
#     print("\nClassifier:", clf_list[i])
#     for j in range(len(datasets)):
#         print("Dataset:", datasets[j].split("/", 1)[1])
#         for k in range(len(preprocs_list)):
#             print("Preprocessing method:", preprocs_list[k])
#             for l in range(len(metrics_list)):
#                 print(metrics_list[l], ":", mean_scores[i][j][k][l])

# TABELE I STATYSTYKI:

alpha = 0.05

for clf_id, clf_name in enumerate(classifiers):
    print("\n--------------------------------------------------------------------------------------------------------")
    print("Classifier:", clf_name)
    print(preprocs_list)
    for data_id, dataset_name in enumerate(datasets):
        print("\n\hline")
        print(dataset_name.split("datasets/")[1], end=" & ")

        stats_list = ["---", "---", "---", "---", "---"]
        for preproc_id, preproc_name in enumerate(preprocs_list):
            print(f"{mean_scores[clf_id, data_id, preproc_id, 5]:.4f}", end=" & ")

            for preproc_id2 in range(preproc_id + 1, len(preprocs_list)):
                bac_list1 = scores[clf_id, data_id, preproc_id, :, 5]
                bac_list2 = scores[clf_id, data_id, preproc_id2, :, 5]
                t_stat, p_value = ttest_ind(bac_list1, bac_list2)  # test t-Studenta

                # czy ważne statystycznie:
                if p_value < alpha:
                    if t_stat > 0:  # jeżeli t_stat dodatnie to pierwsza metoda lepsza od drugiej, ujemne odwrotnie
                        if stats_list[preproc_id] != "---":
                            stats_list[preproc_id] = stats_list[preproc_id] + "," + str(preproc_id2 + 1)
                        else:
                            stats_list[preproc_id] = str(preproc_id2 + 1)
                    else:
                        if stats_list[preproc_id2] != "---":
                            stats_list[preproc_id2] = stats_list[preproc_id2] + "," + str(preproc_id + 1)
                        else:
                            stats_list[preproc_id2] = str(preproc_id + 1)
        print("")
        for stat in stats_list:
            print(stat, end=" & ")

    bac_list = mean_scores[clf_id, :, :, 5]
    mean_bac_list = np.mean(bac_list, axis=0)

    ranks = []
    for bac in bac_list:
        ranks.append(rankdata(bac).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)

    print("\n\hline")
    print("Średnie BAC", end=" & ")
    for preproc_id, preproc_name in enumerate(preprocs_list):
        print(f"{mean_bac_list[preproc_id]:.4f}", end=" & ")

    print("\nŚrednia ranga", end=" & ")
    for rank in mean_ranks:
        print(f"{rank:.4f}", end=" & ")

    w_stats_list = ["---", "---", "---", "---", "---"]
    for preproc_id, preproc_name in enumerate(preprocs_list):
        for preproc_id2 in range(preproc_id + 1, len(preprocs_list)):
            rank_list1 = ranks[:, preproc_id]
            rank_list2 = ranks[:, preproc_id2]
            w_stat, p_value = ranksums(rank_list1, rank_list2)  # test Wilcoxona

            # czy ważne statystycznie:
            if p_value < alpha:
                if w_stat > 0:  # jeżeli w_stat dodatnie to pierwsza metoda lepsza od drugiej, ujemne odwrotnie
                    if w_stats_list[preproc_id] != "---":
                        w_stats_list[preproc_id] = w_stats_list[preproc_id] + "," + str(preproc_id2 + 1)
                    else:
                        w_stats_list[preproc_id] = str(preproc_id2 + 1)
                else:
                    if w_stats_list[preproc_id2] != "---":
                        w_stats_list[preproc_id2] = w_stats_list[preproc_id2] + "," + str(preproc_id + 1)
                    else:
                        w_stats_list[preproc_id2] = str(preproc_id + 1)
    print("")
    for w_stat in w_stats_list:
        print(w_stat, end=" & ")
    print("\n\hline")

# WYKRESY:

for clf_id, clf_name in enumerate(classifiers):
    fig = go.Figure()
    theta = metrics_list
    theta += theta[:1]
    for preproc_id, preproc in enumerate(preprocs):
        values = mean_mean_scores[clf_id, preproc_id, :].tolist()
        values += values[:1]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            name=preproc
        ))

    fig.update_layout(
        title="Radar plot for " + clf_name,
        polar=dict(
            radialaxis_angle=30,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )),
        showlegend=True
    )
    fig.show()
