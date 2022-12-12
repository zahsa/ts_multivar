import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def my_rf(x_train, x_test, y_train, y_test, n_trees=100, criterion='gini', max_depth=7):
    model = RandomForestClassifier(n_estimators=n_trees, criterion=criterion, max_depth=max_depth)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1_test = f1_score(y_pred, y_test, average='macro')
    summary = classification_report(y_test, y_pred)
    acc_test = accuracy_score(y_pred, y_test)
    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)
    f1_train = f1_score(y_pred, y_train, average='macro')
    print(f'RF: {acc_train} - {acc_test}')
    print(f'RF: {f1_train} - {f1_test}')
    return summary


def my_xgb(x_train, x_test, y_train, y_test, n_trees=100, lr=0.1, max_depth=7):
    model = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=lr, max_depth=max_depth)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1_test = f1_score(y_pred, y_test, average='macro')
    summary = classification_report(y_test, y_pred)
    acc_test = accuracy_score(y_pred, y_test)
    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)
    f1_train = f1_score(y_pred, y_train, average='macro')
    print(f'XGB: {acc_train} - {acc_test}')
    print(f'XGB: {f1_train} - {f1_test}')
    return summary


def my_dt(x_train, x_test, y_train, y_test, criterion='gini', max_depth=10):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1_test = f1_score(y_pred, y_test, average='macro')
    summary = classification_report(y_test, y_pred)
    acc_test = accuracy_score(y_pred, y_test)
    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)
    f1_train = f1_score(y_pred, y_train, average='macro')
    print(f'DT: {acc_train} - {acc_test}')
    print(f'DT: {f1_train} - {f1_test}')
    return summary


def my_svm(x_train, x_test, y_train, y_test, kernel='rbf', gamma='auto'):
    model = SVC(kernel=kernel, gamma=gamma)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1_test = f1_score(y_pred, y_test, average='macro')
    summary = classification_report(y_test, y_pred)
    acc_test = accuracy_score(y_pred, y_test)
    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)
    f1_train = f1_score(y_pred, y_train, average='macro')
    print(f'SVM: {acc_train} - {acc_test}')
    print(f'SVM: {f1_train} - {f1_test}')
    return summary


def my_lg(x_train, x_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1_test = f1_score(y_pred, y_test, average='macro')
    summary = classification_report(y_test, y_pred)
    acc_test = accuracy_score(y_pred, y_test)
    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)
    f1_train = f1_score(y_pred, y_train, average='macro')
    print(f'LG: {acc_train} - {acc_test}')
    print(f'LG: {f1_train} - {f1_test}')
    return summary


def my_knn(x_train, x_test, y_train, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1_test = f1_score(y_pred, y_test, average='macro')
    summary = classification_report(y_test, y_pred)
    acc_test = accuracy_score(y_pred, y_test)
    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)
    f1_train = f1_score(y_pred, y_train, average='macro')
    print(f'KNN: {acc_train} - {acc_test}')
    print(f'KNN: {f1_train} - {f1_test}')
    return summary


def my_mlp(x_train, x_test, y_train, y_test, hidden_layer_sizes=(50,100,200,50,), max_iter=300):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1_test = f1_score(y_pred, y_test, average='macro')
    summary = classification_report(y_test, y_pred)
    acc_test = accuracy_score(y_pred, y_test)
    y_pred = model.predict(x_train)
    acc_train = accuracy_score(y_pred, y_train)
    f1_train = f1_score(y_pred, y_train, average='macro')
    print(f'MLP: {acc_train} - {acc_test}')
    print(f'MLP: {f1_train} - {f1_test}')
    return summary


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
    summary_rf = my_rf(x_train, x_test, y_train, y_test)
    # print(summary_rf)
    summary_xgb = my_xgb(x_train, x_test, y_train, y_test)
    # print(summary_xgb)
    summary_svm = my_svm(x_train, x_test, y_train, y_test)
    # print(summary_svm)
    summary_dt = my_dt(x_train, x_test, y_train, y_test)
    # print(summary_dt)
    summary_lg = my_lg(x_train, x_test, y_train, y_test)
    # print(summary_lg)
    summary_knn = my_knn(x_train, x_test, y_train, y_test)
    # print(summary_knn)
    summary_mlp = my_mlp(x_train, x_test, y_train, y_test)
    # print(summary_mlp)

