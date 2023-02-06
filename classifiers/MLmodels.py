import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


def my_rf(x_train, x_test, y_train, y_test, criterion='gini', metric='accuracy'):
    # print('\trf')
    # parameters = {'n_estimators': [50, 100, 200, 300, 500], 'max_depth': [3, 5, 7, 9]}
    # model_gs = RandomForestClassifier(random_state=42)
    # clf = GridSearchCV(model_gs, parameters)
    # clf.fit(x_train, y_train)
    # print(f'\t\t{clf.best_params_}')
    # whole trajectory arima {'max_depth': 7, 'n_estimators': 100} {'max_depth': 9, 'n_estimators': 50}
    # whole trajectory ou {'max_depth': 9, 'n_estimators': 200} {'max_depth': 9, 'n_estimators': 300}
    # whole trajectory varma {'max_depth': 9, 'n_estimators': 500}
    # navigation arima {'max_depth': 9, 'n_estimators': 500}
    # navigation ou {'max_depth': 7, 'n_estimators': 50}
    # navigation varma {'max_depth': 7, 'n_estimators': 50}
    # port arima {'max_depth': 7, 'n_estimators': 300}
    # port ou {'max_depth': 5, 'n_estimators': 50}
    # port varma {'max_depth': 5, 'n_estimators': 200}
    # port vs navigation {'max_depth': 9, 'n_estimators': 50}

    # model = RandomForestClassifier(n_estimators=clf.best_params_['n_estimators'],
    #                                criterion=criterion, max_depth=clf.best_params_['max_depth'], random_state = 42)
    model = RandomForestClassifier(n_estimators=300,
                                   criterion=criterion, max_depth=9, random_state=42)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')

    matrix = confusion_matrix(y_test, y_pred) #, labels=[30, 37, 60, 80])
    acc_per_class = matrix.diagonal() / matrix.sum(axis=1)

    return scores.mean(), scores.std(), test_score, acc_per_class, matrix


def my_xgb(x_train, x_test, y_train, y_test, metric='accuracy'):
    # print('\txgb')
    # parameters = {'n_estimators': [50, 100, 200, 300, 500], 'max_depth': [3, 5, 7, 9],
    #               'learning_rate': [0.1, 0.01, 0.001]}
    # model_gs = GradientBoostingClassifier(random_state = 42)
    # clf = GridSearchCV(model_gs, parameters)
    # clf.fit(x_train, y_train)
    # print(f'\t\t{clf.best_params_}')
    # whole trajectory arima {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
    # whole trajectory ou {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 500}
    # whole trajectory varma
    # navigation arima
    # navigation ou
    # navigation varma
    # port arima
    # port ou
    # port varma

    # model = GradientBoostingClassifier(n_estimators=clf.best_params_['n_estimators'],
    #                                    learning_rate=clf.best_params_['learning_rate'],
    #                                    max_depth=clf.best_params_['max_depth'], random_state = 42)
    model = GradientBoostingClassifier(n_estimators=500,
                                       learning_rate=0.05,
                                       max_depth=7, random_state = 42)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')

    matrix = confusion_matrix(y_test, y_pred)#, labels=[30, 37, 60, 80])
    acc_per_class = matrix.diagonal() / matrix.sum(axis=1)

    return scores.mean(), scores.std(), test_score, acc_per_class, matrix


def my_dt(x_train, x_test, y_train, y_test, criterion='gini', metric='accuracy'):
    # print('\tdt')
    # parameters = {'max_depth': [3, 5, 7, 9]}
    # model_gs = DecisionTreeClassifier(random_state = 42)
    # clf = GridSearchCV(model_gs, parameters)
    # clf.fit(x_train, y_train)
    # print(f'\t\t{clf.best_params_}')
    # whole trajectory arima {'max_depth': 5}
    # whole trajectory ou {'max_depth': 7}
    # whole trajectory varma {'max_depth': 7}
    # navigation arima {'max_depth': 5}
    # navigation ou {'max_depth': 3}
    # navigation varma {'max_depth': 5}
    # port arima {'max_depth': 7}
    # port ou {'max_depth': 9}
    # port varma {'max_depth': 5}

    # model = DecisionTreeClassifier(criterion=criterion, max_depth=clf.best_params_['max_depth'], random_state = 42)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state = 42)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')

    matrix = confusion_matrix(y_test, y_pred)#, labels=[30, 37, 60, 80])
    acc_per_class = matrix.diagonal() / matrix.sum(axis=1)

    return scores.mean(), scores.std(), test_score, acc_per_class


def my_svm(x_train, x_test, y_train, y_test, metric='accuracy'):
    # print('\tsvm')
    # parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10]}
    # model_gs = SVC(gamma='auto', random_state = 42)
    # clf = GridSearchCV(model_gs, parameters)
    # clf.fit(x_train, y_train)
    # print(f'\t\t{clf.best_params_}')
    # whole trajectory arima {'C': 10, 'kernel': 'rbf'}
    # navigation
    # port
    # navigation arima
    # navigation ou
    # navigation varma
    # port arima
    # port ou
    # port varma

    # model = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], random_state = 42)
    model = SVC(kernel='rbf', C=10, gamma='auto', random_state = 42)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')

    matrix = confusion_matrix(y_test, y_pred)#, labels=[30, 37, 60, 80])
    acc_per_class = matrix.diagonal() / matrix.sum(axis=1)

    return scores.mean(), scores.std(), test_score, acc_per_class, matrix


def my_lg(x_train, x_test, y_train, y_test, metric='accuracy'):
    # print('\tlg')
    # parameters = {'max_iter': [100, 300, 500, 1000]}
    # model_gs = LogisticRegression(random_state = 42)
    # clf = GridSearchCV(model_gs, parameters)
    # clf.fit(x_train, y_train)
    # print(f'\t\t{clf.best_params_}')
    # whole trajectory {'max_iter': 100}
    # whole trajectory ou {'max_iter': 1000}
    # whole trajectory varma {'max_iter': 100}
    # navigation arima  {'max_iter': 100}
    # navigation ou {'max_iter': 100}
    # navigation varma {'max_iter': 100}
    # port arima {'max_iter': 100}
    # port ou {'max_iter': 100}
    # port varma {'max_iter': 100}

    # model = LogisticRegression(max_iter=clf.best_params_['max_iter'], random_state = 42)
    model = LogisticRegression(max_iter=100, random_state = 42)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')

    matrix = confusion_matrix(y_test, y_pred)#, labels=[30, 37, 60, 80])
    acc_per_class = matrix.diagonal() / matrix.sum(axis=1)

    return scores.mean(), scores.std(), test_score, acc_per_class, matrix


def my_knn(x_train, x_test, y_train, y_test, metric='accuracy'):
    # print('\tknn')
    # parameters = {'n_neighbors': [3, 5, 7, 9]}
    # model_gs = KNeighborsClassifier()
    # clf = GridSearchCV(model_gs, parameters)
    # clf.fit(x_train, y_train)
    # print(f'\t\t{clf.best_params_}')
    # whole trajectory {'n_neighbors': 7}
    # whole trajectory ou {'n_neighbors': 3}
    # whole trajectory varma {'n_neighbors': 7}
    # navigation arima  {'n_neighbors': 3}
    # navigation ou {'n_neighbors': 5}
    # navigation varma {'n_neighbors': 3}
    # port arima {'n_neighbors': 7}
    # port ou {'n_neighbors': 9}
    # port varma {'n_neighbors': 9}

    # model = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
    model = KNeighborsClassifier(n_neighbors=7)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')

    matrix = confusion_matrix(y_test, y_pred)#, labels=[30, 37, 60, 80])
    acc_per_class = matrix.diagonal() / matrix.sum(axis=1)

    return scores.mean(), scores.std(), test_score, acc_per_class


def my_mlp(x_train, x_test, y_train, y_test, metric='accuracy'):
    # print('\tmlp')
    # parameters = {'hidden_layer_sizes': [(256,512), (100,100), (128,256), (256,512,1024), (512, 1024), (100,200,100)], 'max_iter': [100, 300, 500, 1000]}
    # model_gs = MLPClassifier(random_state = 42)
    # clf = GridSearchCV(model_gs, parameters)
    # clf.fit(x_train, y_train)
    # print(f'\t\t{clf.best_params_}')
    # whole trajectory arima{'hidden_layer_sizes': 400, 'max_iter': 300} {'hidden_layer_sizes': 100, 'max_iter': 1000}
    # whole trajectory ou {'hidden_layer_sizes': 200, 'max_iter': 500}
    # whole trajectory varma {'hidden_layer_sizes': 400, 'max_iter': 1000}
    # navigation arima  {'hidden_layer_sizes': 300, 'max_iter': 500}
    # navigation ou {'hidden_layer_sizes': 300, 'max_iter': 100}
    # navigation varma {'hidden_layer_sizes': 300, 'max_iter': 300}
    # port arima {'hidden_layer_sizes': 50, 'max_iter': 1000}
    # port ou {'hidden_layer_sizes': 200, 'max_iter': 100}
    # port varma {'hidden_layer_sizes': 300, 'max_iter': 500}

    # model = MLPClassifier(hidden_layer_sizes=clf.best_params_['hidden_layer_sizes'],
    #                       max_iter=clf.best_params_['max_iter'], random_state = 42)
    model = MLPClassifier(hidden_layer_sizes=(256, 512, ),
                          max_iter=500, random_state = 42)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')

    matrix = confusion_matrix(y_test, y_pred)#, labels=[30, 37, 60, 80])
    acc_per_class = matrix.diagonal() / matrix.sum(axis=1)

    return scores.mean(), scores.std(), test_score, acc_per_class


def stratify_dataset(x, lbl='vessel_type'):
    new_data = pd.DataFrame()
    labels = x[lbl].unique()
    n_min = min(x.groupby(lbl).count().mmsi)
    for l in labels:
        px = x[x[lbl] == l]
        px = px.sample(n=n_min)
        new_data = pd.concat([new_data, px], axis=0)
    return new_data


def apply_ML_vt(data_path, label=False, folder='results', name='whole'):
    # Set seed for libraries to ensure reproducability
    np.random.seed(42)
    sklearn.random.seed(42)
    random.seed(42)

    dataset = pd.read_csv(f'{data_path}/dataset.csv', index_col=0)
    # balacing dataset
    dataset = stratify_dataset(dataset)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    dataset.drop(['mmsi'], axis=1, inplace=True)
    features =dataset.drop(['vessel_type', 'label'], axis=1)
    labels = dataset['vessel_type']
    if label:
        labels = dataset['label']
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

    # normalization
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    # models
    rf_mean, rf_std, rf_test, rf_acc_pc, rf_conf_matrix = my_rf(x_train, x_test, y_train, y_test)
    xgb_mean, xgb_std, xgb_test, xgb_acc_pc, xgb_conf_matrix = my_xgb(x_train, x_test, y_train, y_test)
    svm_mean, svm_std, svm_test, svm_acc_pc, svm_conf_matrix = my_svm(x_train, x_test, y_train, y_test)
    dt_mean, dt_std, dt_test, dt_acc_pc = my_dt(x_train, x_test, y_train, y_test)
    lg_mean, lg_std, lg_test, lg_acc_pc, lg_conf_matrix = my_lg(x_train, x_test, y_train, y_test)
    knn_mean, knn_std, knn_test, knn_acc_pc = my_knn(x_train, x_test, y_train, y_test)
    mlp_mean, mlp_std, mlp_test, mlp_acc_pc = my_mlp(x_train, x_test, y_train, y_test)

    # print
    print(data_path)
    # print(f'& RF & ${round(rf_mean, 2)} \pm {round(rf_std, 2)}$ & ${round(rf_test, 2)}$')
    # print(f'& XGB & ${round(xgb_mean, 2)} \pm {round(xgb_std, 2)}$ & ${round(xgb_test, 2)}$')
    # print(f'& DT & ${round(dt_mean, 2)} \pm {round(dt_std, 2)}$ & ${round(dt_test, 2)}$')
    # print(f'& LR & ${round(lg_mean, 2)} \pm {round(lg_std, 2)}$ & ${round(lg_test, 2)}$')
    # print(f'& SVM & ${round(svm_mean, 2)} \pm {round(svm_std, 2)}$ & ${round(svm_test, 2)}$')
    # print(f'& KNN & ${round(knn_mean, 2)} \pm {round(knn_std, 2)}$ & ${round(knn_test, 2)}$')
    # print(f'& MLP & ${round(mlp_mean, 2)} \pm {round(mlp_std, 2)}$ & ${round(mlp_test, 2)}$')
    #
    # # print confucion matrix
    # print('')
    # print(f'& Random Forest & {rf_acc_pc.round(2)[0]} & {rf_acc_pc.round(2)[3]} & {rf_acc_pc.round(2)[2]} & {rf_acc_pc.round(2)[1]}')
    # print(f'& XGBoost & {xgb_acc_pc.round(2)[0]} & {xgb_acc_pc.round(2)[3]} & {xgb_acc_pc.round(2)[2]} & {xgb_acc_pc.round(2)[1]}')
    # print(f'& Decision Tree & {dt_acc_pc.round(2)[0]} & {dt_acc_pc.round(2)[3]} & {dt_acc_pc.round(2)[2]} & {dt_acc_pc.round(2)[1]}')
    # print(f'& Logistic Regression & {lg_acc_pc.round(2)[0]} & {lg_acc_pc.round(2)[3]} & {lg_acc_pc.round(2)[2]} & {lg_acc_pc.round(2)[1]}')
    # print(f'& SVM & {svm_acc_pc.round(2)[0]} & {svm_acc_pc.round(2)[3]} & {svm_acc_pc.round(2)[2]} & {svm_acc_pc.round(2)[1]}')
    # print(f'& KNN & {knn_acc_pc.round(2)[0]} & {knn_acc_pc.round(2)[3]} & {knn_acc_pc.round(2)[2]} & {knn_acc_pc.round(2)[1]}')
    # print(f'& MLP & {mlp_acc_pc.round(2)[0]} & {mlp_acc_pc.round(2)[3]} & {mlp_acc_pc.round(2)[2]} & {mlp_acc_pc.round(2)[1]}')

    # all results
    print('')
    print(f'& RF & ${round(rf_mean, 2)} \pm {round(rf_std, 2)}$ & ${round(rf_test, 2)}$ & {rf_acc_pc.round(2)[0]} & {rf_acc_pc.round(2)[3]} & {rf_acc_pc.round(2)[2]} & {rf_acc_pc.round(2)[1]} \\\\')
    print(f'& XGB & ${round(xgb_mean, 2)} \pm {round(xgb_std, 2)}$ & ${round(xgb_test, 2)}$ & {xgb_acc_pc.round(2)[0]} & {xgb_acc_pc.round(2)[3]} & {xgb_acc_pc.round(2)[2]} & {xgb_acc_pc.round(2)[1]} \\\\')
    print(f'& DT & ${round(dt_mean, 2)} \pm {round(dt_std, 2)}$ & ${round(dt_test, 2)}$ & {dt_acc_pc.round(2)[0]} & {dt_acc_pc.round(2)[3]} & {dt_acc_pc.round(2)[2]} & {dt_acc_pc.round(2)[1]} \\\\')
    print(f'& LR & ${round(lg_mean, 2)} \pm {round(lg_std, 2)}$ & ${round(lg_test, 2)}$ & {lg_acc_pc.round(2)[0]} & {lg_acc_pc.round(2)[3]} & {lg_acc_pc.round(2)[2]} & {lg_acc_pc.round(2)[1]} \\\\')
    print(f'& SVM & ${round(svm_mean, 2)} \pm {round(svm_std, 2)}$ & ${round(svm_test, 2)}$ & {svm_acc_pc.round(2)[0]} & {svm_acc_pc.round(2)[3]} & {svm_acc_pc.round(2)[2]} & {svm_acc_pc.round(2)[1]} \\\\')
    print(f'& KNN & ${round(knn_mean, 2)} \pm {round(knn_std, 2)}$ & ${round(knn_test, 2)}$ & {knn_acc_pc.round(2)[0]} & {knn_acc_pc.round(2)[3]} & {knn_acc_pc.round(2)[2]} & {knn_acc_pc.round(2)[1]} \\\\')
    print(f'& MLP & ${round(mlp_mean, 2)} \pm {round(mlp_std, 2)}$ & ${round(mlp_test, 2)}$ & {mlp_acc_pc.round(2)[0]} & {mlp_acc_pc.round(2)[3]} & {mlp_acc_pc.round(2)[2]} & {mlp_acc_pc.round(2)[1]} \\\\')

    plt.style.use("seaborn")

    # acc_matrix = np.array([])
    # acc_matrix = np.concatenate((acc_matrix, rf_acc_pc), axis=0)
    # acc_matrix = np.vstack((acc_matrix, xgb_acc_pc))
    # acc_matrix = np.vstack((acc_matrix, dt_acc_pc))
    # acc_matrix = np.vstack((acc_matrix, lg_acc_pc))
    # acc_matrix = np.vstack((acc_matrix, svm_acc_pc))
    # acc_matrix = np.vstack((acc_matrix, knn_acc_pc))
    # acc_matrix = np.vstack((acc_matrix, mlp_acc_pc))
    #
    # acc_matrix_df = pd.DataFrame(acc_matrix)
    # acc_matrix_df.index = ['Random Forest', 'XGBoost', 'DT', 'LG', 'SVM', 'KNN', 'MLP']
    # acc_matrix_df.columns = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    # acc_matrix_df.to_csv(f'{folder}/acc_matrix_{name}.csv')

    rf_conf_matrix = pd.DataFrame(rf_conf_matrix)
    rf_conf_matrix.index = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    rf_conf_matrix.columns = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    rf_conf_matrix = rf_conf_matrix/rf_conf_matrix.sum(axis=1)
    rf_conf_matrix.to_csv(f'{folder}/rf_conf_matrix_{name}.csv')

    xgb_conf_matrix = pd.DataFrame(xgb_conf_matrix)
    xgb_conf_matrix.index = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    xgb_conf_matrix.columns = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    xgb_conf_matrix = xgb_conf_matrix / xgb_conf_matrix.sum(axis=1)
    xgb_conf_matrix.to_csv(f'{folder}/rf_conf_matrix_{name}.csv')

    svm_conf_matrix = pd.DataFrame(svm_conf_matrix)
    svm_conf_matrix.index = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    svm_conf_matrix.columns = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    svm_conf_matrix = svm_conf_matrix / svm_conf_matrix.sum(axis=1)
    svm_conf_matrix.to_csv(f'{folder}/rf_conf_matrix_{name}.csv')

    lg_conf_matrix = pd.DataFrame(lg_conf_matrix)
    lg_conf_matrix.index = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    lg_conf_matrix.columns = ['Fishing', 'Pleasure Craft', 'Passengers', 'Tanker']
    lg_conf_matrix = lg_conf_matrix / lg_conf_matrix.sum(axis=1)
    lg_conf_matrix.to_csv(f'{folder}/rf_conf_matrix_{name}.csv')

    cmap = sns.light_palette("seagreen")
    plt.figure(figsize=(17, 10))
    heat_map = sns.heatmap(rf_conf_matrix, cmap=cmap, linewidth=1, annot=True,
                           annot_kws={
                               'fontsize': 25,
                               'fontweight': 'bold',
                               # 'fontfamily': 'serif'
                           }
                           )
    heat_map.tick_params(labelbottom=False, bottom=False, right=False, top=True, labelright=False, labeltop=True, labelrotation=0)
    plt.yticks(rotation=0, fontsize=25)
    plt.xticks(rotation=0, fontsize=25)
    plt.savefig(f'{folder}/correlation_rf_{name}.png')

    cmap = sns.light_palette("seagreen")
    plt.figure(figsize=(17, 10))
    heat_map = sns.heatmap(xgb_conf_matrix, cmap=cmap, linewidth=1, annot=True,
                           annot_kws={
                               'fontsize': 25,
                               'fontweight': 'bold',
                               # 'fontfamily': 'serif'
                           }
                           )
    heat_map.tick_params(labelbottom=False, bottom=False, right=False, top=True, labelright=False, labeltop=True,
                         labelrotation=0)
    plt.yticks(rotation=0, fontsize=25)
    plt.xticks(rotation=0, fontsize=25)
    plt.savefig(f'{folder}/correlation_xgb_{name}.png')

    cmap = sns.light_palette("seagreen")
    plt.figure(figsize=(17, 10))
    heat_map = sns.heatmap(svm_conf_matrix, cmap=cmap, linewidth=1, annot=True,
                           annot_kws={
                               'fontsize': 25,
                               'fontweight': 'bold',
                               # 'fontfamily': 'serif'
                           }
                           )
    heat_map.tick_params(labelbottom=False, bottom=False, right=False, top=True, labelright=False, labeltop=True,
                         labelrotation=0)
    plt.yticks(rotation=0, fontsize=25)
    plt.xticks(rotation=0, fontsize=25)
    plt.savefig(f'{folder}/correlation_svm_{name}.png')

    cmap = sns.light_palette("seagreen")
    plt.figure(figsize=(17, 10))
    heat_map = sns.heatmap(lg_conf_matrix, cmap=cmap, linewidth=1, annot=True,
                           annot_kws={
                               'fontsize': 25,
                               'fontweight': 'bold',
                               # 'fontfamily': 'serif'
                           }
                           )
    heat_map.tick_params(labelbottom=False, bottom=False, right=False, top=True, labelright=False, labeltop=True,
                         labelrotation=0)
    plt.yticks(rotation=0, fontsize=25)
    plt.xticks(rotation=0, fontsize=25)
    plt.savefig(f'{folder}/correlation_lg_{name}.png')


def apply_ML_np(data_path, port=False):
    # Set seed for libraries to ensure reproducability
    np.random.seed(42)
    sklearn.random.seed(42)
    random.seed(42)

    dataset = pd.read_csv(f'{data_path}/dataset.csv', index_col=0)
    # balacing dataset
    dataset = stratify_dataset(dataset, lbl='labels2')

    dataset.drop(['mmsi', 'vessel_type'], axis=1, inplace=True)
    features = dataset.iloc[:, 0:-2]
    labels = dataset['labels2']
    if port:
        labels = dataset['label']
    labels[labels == 'navigating'] = 0
    labels[labels == 'port'] = 1
    labels = labels.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

    # normalization
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    # models
    rf_mean, rf_std, rf_test, rf_acc_pc, rf_conf_matrix = my_rf(x_train, x_test, y_train, y_test)
    xgb_mean, xgb_std, xgb_test, xgb_acc_pc, xgb_conf_matrix = my_xgb(x_train, x_test, y_train, y_test)
    svm_mean, svm_std, svm_test, svm_acc_pc, svm_conf_matrix = my_svm(x_train, x_test, y_train, y_test)
    dt_mean, dt_std, dt_test, dt_acc_pc = my_dt(x_train, x_test, y_train, y_test)
    lg_mean, lg_std, lg_test, lg_acc_pc, lg_conf_matrix = my_lg(x_train, x_test, y_train, y_test)
    knn_mean, knn_std, knn_test, knn_acc_pc = my_knn(x_train, x_test, y_train, y_test)
    mlp_mean, mlp_std, mlp_test, mlp_acc_pc = my_mlp(x_train, x_test, y_train, y_test)

    # print
    print(data_path)
    print(f'& Random Forest & {round(rf_mean, 2)} & {round(rf_std, 2)} & {round(rf_test, 2)}')
    print(f'& XGBoost & {round(xgb_mean, 2)} & {round(xgb_std, 2)} & {round(xgb_test, 2)}')
    print(f'& Decision Tree & {round(dt_mean, 2)} & {round(dt_std, 2)} & {round(dt_test, 2)}')
    print(f'& Logistic Regression & {round(lg_mean, 2)} & {round(lg_std, 2)} & {round(lg_test, 2)}')
    print(f'& SVM & {round(svm_mean, 2)} & {round(svm_std, 2)} & {round(svm_test, 2)}')
    print(f'& KNN & {round(knn_mean, 2)} & {round(knn_std, 2)} & {round(knn_test, 2)}')
    print(f'& MLP & {round(mlp_mean, 2)} & {round(mlp_std, 2)} & {round(mlp_test, 2)}')

    # print confucion matrix
    # print(f'& Random Forest & {rf_acc_pc.round(2)}')
    # print(f'& XGBoost & {xgb_acc_pc.round(2)}')
    # print(f'& Decision Tree & {dt_acc_pc.round(2)}')
    # print(f'& Logistic Regression & {lg_acc_pc.round(2)}')
    # print(f'& SVM & {svm_acc_pc.round(2)}')
    # print(f'& KNN & {knn_acc_pc.round(2)}')
    # print(f'& MLP & {mlp_acc_pc.round(2)}')

    # for each vessel type
    print('\nFOR EACH VESSEL TYPE\n')
    # Set seed for libraries to ensure reproducability
    np.random.seed(42)
    sklearn.random.seed(42)
    random.seed(42)

    dataset = pd.read_csv(f'{data_path}/dataset.csv', index_col=0)
    # balacing dataset
    dataset = stratify_dataset(dataset, lbl='labels2')
    vt_list = np.unique(dataset['vessel_type'])

    for vt in vt_list:
        dataset_cur = dataset[dataset['vessel_type'] == vt]
        dataset_cur.drop(['mmsi', 'vessel_type'], axis=1, inplace=True)
        features = dataset_cur.iloc[:, 0:-2]
        labels = dataset_cur['labels2']
        if port:
            labels = dataset_cur['label']
        labels[labels == 'navigating'] = 0
        labels[labels == 'port'] = 1
        labels = labels.astype(int)

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

        # normalization
        # scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)

        # models
        rf_mean, rf_std, rf_test, rf_acc_pc, rf_conf_matrix = my_rf(x_train, x_test, y_train, y_test)
        xgb_mean, xgb_std, xgb_test, xgb_acc_pc, xgb_conf_matrix = my_xgb(x_train, x_test, y_train, y_test)
        svm_mean, svm_std, svm_test, svm_acc_pc, svm_conf_matrix = my_svm(x_train, x_test, y_train, y_test)
        dt_mean, dt_std, dt_test, dt_acc_pc = my_dt(x_train, x_test, y_train, y_test)
        lg_mean, lg_std, lg_test, lg_acc_pc, lg_conf_matrix = my_lg(x_train, x_test, y_train, y_test)
        knn_mean, knn_std, knn_test, knn_acc_pc = my_knn(x_train, x_test, y_train, y_test)
        mlp_mean, mlp_std, mlp_test, mlp_acc_pc = my_mlp(x_train, x_test, y_train, y_test)

        # print
        print(f'{data_path} -- {vt}')
        print(f'& Random Forest & {round(rf_mean, 2)} & {round(rf_std, 2)} & {round(rf_test, 2)}')
        print(f'& XGBoost & {round(xgb_mean, 2)} & {round(xgb_std, 2)} & {round(xgb_test, 2)}')
        print(f'& Decision Tree & {round(dt_mean, 2)} & {round(dt_std, 2)} & {round(dt_test, 2)}')
        print(f'& Logistic Regression & {round(lg_mean, 2)} & {round(lg_std, 2)} & {round(lg_test, 2)}')
        print(f'& SVM & {round(svm_mean, 2)} & {round(svm_std, 2)} & {round(svm_test, 2)}')
        print(f'& KNN & {round(knn_mean, 2)} & {round(knn_std, 2)} & {round(knn_test, 2)}')
        print(f'& MLP & {round(mlp_mean, 2)} & {round(mlp_std, 2)} & {round(mlp_test, 2)}')

    # print confucion matrix
    # print(f'& Random Forest & {rf_acc_pc.round(2)}')
    # print(f'& XGBoost & {xgb_acc_pc.round(2)}')
    # print(f'& Decision Tree & {dt_acc_pc.round(2)}')
    # print(f'& Logistic Regression & {lg_acc_pc.round(2)}')
    # print(f'& SVM & {svm_acc_pc.round(2)}')
    # print(f'& KNN & {knn_acc_pc.round(2)}')
    # print(f'& MLP & {mlp_acc_pc.round(2)}')

