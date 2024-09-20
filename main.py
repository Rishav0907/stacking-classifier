import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from Logistic_regression_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from meta_classifier import MetaClassifier
from svm import SVM
from config import LOGISTIC_REGRESSION_CONFIG
from torch.optim import Adam
import torch.nn as nn
import torch


def meta_trainer(X_train, y_train, model):
    optimizer = Adam(lr=0.001, params=model.parameters())
    loss = nn.BCELoss()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    for epoch in range(100):
        optimizer.zero_grad()
        y_predicted = model.forward(X_train)
        loss_value = loss(y_predicted, y_train)
        loss_value.backward()
        optimizer.step()
    return model


def meta_predict(X_test, y_test, model):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)
    y_predicted = model.forward(X_test)
    return y_predicted


if __name__ == '__main__':
    heart_disease = fetch_ucirepo(id=45)
    data = heart_disease['data']
    # print(data.keys())
    heart_disease_data = data['features']
    heart_disease_target = data['targets']
    heart_disease_data_df = pd.DataFrame(heart_disease_data)
    heart_disease_target_df = pd.DataFrame(heart_disease_target)
    feature_columns = list(heart_disease_data_df.columns)
    target_column = list(heart_disease_target_df.columns)

    concatenated_data = np.concatenate(
        (heart_disease_data_df, heart_disease_target_df), axis=-1)
    concatenated_dataframe = pd.DataFrame(
        concatenated_data, columns=feature_columns+target_column)
    concatenated_dataframe = concatenated_dataframe.dropna()
    # print(concatenated_dataframe['num'])
    heart_disease_data = concatenated_dataframe.drop(columns=target_column)
    heart_disease_target = concatenated_dataframe[target_column]
    heart_disease_target = heart_disease_target.replace(
        to_replace=[2, 3, 4], value=1)
    X_train, X_test, y_train, y_test = train_test_split(
        heart_disease_data, heart_disease_target, test_size=0.2)
    # print(heart_disease_target)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # dt = DecisionTree()
    # dt.fit(X_train, y_train)
    # dt.build_decision_tree(
    # X_train, y_train, current_tree_depth=0, feature_columns=feature_columns)
    # print(LOGISTIC_REGRESSION_CONFIG['learning_rate'])

    lr = LogisticRegression(
        input_dim=LOGISTIC_REGRESSION_CONFIG['num_features'], out_dim=LOGISTIC_REGRESSION_CONFIG['num_classes'])
    lr.train(X_train, y_train)
    logistic_regression_resuslt = lr.test(X_test)
    logic_regression_train_result = lr.test(X_train)
    print(
        f"Logistic Regression Accuracy :{accuracy_score(y_test, logistic_regression_resuslt)*100:.2f}%")
    # print(logistic_regression_resuslt)
    # print(y_train.replace(0, -1))
    svm = SVM()
    svm_y_train = y_train.replace({0: -1, 1: 1})
    svm_y_test = y_test.replace({0: -1, 1: 1})
    svm.train(X_train, svm_y_train)
    svm_result = svm.test(X_test).reshape(-1, 1)
    svm_result_train = svm.test(X_train).reshape(-1, 1)
    svm_result = np.where(svm_result == -1, 0, 1)
    svm_result_train = np.where(svm_result_train == -1, 0, 1)
    print(
        f"SVM Classifier Accuracy :{accuracy_score(y_test, svm_result)*100}%")
    meta_train_dataset = np.hstack(
        (logic_regression_train_result, svm_result_train))
    meta_test_dataset = np.hstack(
        (logistic_regression_resuslt, svm_result))
    # print(len(meta__traindataset))
    meta_classifier = MetaClassifier()
    meta_model = meta_trainer(
        meta_train_dataset, y_train, model=meta_classifier)
    meta_classifier_result = meta_predict(
        meta_test_dataset, y_test, model=meta_classifier)
    meta_classifier_result = np.where(meta_classifier_result > 0.5, 1, 0)
    print(
        f"Meta Classifier Accuracy: {accuracy_score(y_test, meta_classifier_result)*100:.2f}%")
    # meta_classifier = MetaClassifier()
    # meta_classifier.fit(meta_train_dataset, y_train)
    # meta_classifier_result = meta_classifier.predict(meta_test_dataset)
    # print(accuracy_score(y_test, meta_classifier_result))
