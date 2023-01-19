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


def my_rf(x_train, x_test, y_train, y_test, n_trees=100, criterion='gini', max_depth=7, metric='accuracy'):
    model = RandomForestClassifier(n_estimators=n_trees, criterion=criterion, max_depth=max_depth)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric=='accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')
    return scores.mean(), scores.std(), test_score


def my_xgb(x_train, x_test, y_train, y_test, n_trees=100, lr=0.1, max_depth=7, metric='accuracy'):
    model = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=lr, max_depth=max_depth)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')
    return scores.mean(), scores.std(), test_score


def my_dt(x_train, x_test, y_train, y_test, criterion='gini', max_depth=10, metric='accuracy'):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')
    return scores.mean(), scores.std(), test_score


def my_svm(x_train, x_test, y_train, y_test, kernel='rbf', gamma='auto', metric='accuracy'):
    model = SVC(kernel=kernel, gamma=gamma)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')
    return scores.mean(), scores.std(), test_score


def my_lg(x_train, x_test, y_train, y_test, max_iter=500, metric='accuracy'):
    model = LogisticRegression(max_iter=max_iter)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')
    return scores.mean(), scores.std(), test_score


def my_knn(x_train, x_test, y_train, y_test, n_neighbors=5, metric='accuracy'):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')
    return scores.mean(), scores.std(), test_score


def my_mlp(x_train, x_test, y_train, y_test, hidden_layer_sizes=(50,100,200,50,), max_iter=500, metric='accuracy'):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == 'accuracy':
        test_score = accuracy_score(y_pred, y_test)
    else:
        test_score = f1_score(y_pred, y_test, average='macro')
    return scores.mean(), scores.std(), test_score


def stratify_dataset(x):
    new_data = pd.DataFrame()
    labels = x['vessel_type'].unique()
    n_min = min(x.groupby('vessel_type').count().mmsi)
    for l in labels:
        px = x[x['vessel_type']==l]
        px = px.sample(n=n_min)
        new_data = pd.concat([new_data, px], axis=0)
    return new_data


def apply_ML(data_path):
    dataset = pd.read_csv(f'{data_path}/dataset.csv', index_col=0)
    # balacing dataset
    dataset = stratify_dataset(dataset)

    features = dataset.iloc[:, 1:-1]
    labels = dataset['vessel_type']
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

    #normalization
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    #models
    rf_mean, rf_std, rf_test = my_rf(x_train, x_test, y_train, y_test)
    xgb_mean, xgb_std, xgb_test = my_xgb(x_train, x_test, y_train, y_test)
    svm_mean, svm_std, svm_test = my_svm(x_train, x_test, y_train, y_test)
    dt_mean, dt_std, dt_test = my_dt(x_train, x_test, y_train, y_test)
    lg_mean, lg_std, lg_test = my_lg(x_train, x_test, y_train, y_test)
    knn_mean, knn_std, knn_test = my_knn(x_train, x_test, y_train, y_test)
    mlp_mean, mlp_std, mlp_test = my_mlp(x_train, x_test, y_train, y_test)

    #print
    print(data_path)
    print(f'Random Forest & {rf_mean} & {rf_std} & {rf_test}')
    print(f'XGBoost & {xgb_mean} & {xgb_std} & {xgb_test}')
    print(f'Decision Tree & {dt_mean} & {dt_std} & {dt_test}')
    print(f'Logistic Regression & {lg_mean} & {lg_std} & {lg_test}')
    print(f'SVM & {svm_mean} & {svm_std} & {svm_test}')
    print(f'KNN & {knn_mean} & {knn_std} & {knn_test}')
    print(f'MLP & {mlp_mean} & {mlp_std} & {mlp_test}')

