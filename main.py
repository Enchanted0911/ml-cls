import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

global x_train, x_test, y_train, y_test, x, y, xx_train, xx_test, yy_train, yy_test


def drop_low_samples(data, label_name, drop_below):
    df_count = pd.DataFrame(data[label_name].value_counts())
    df_count.reset_index(inplace=True)
    df_count.columns = [label_name, 'counts']
    df_with_count = data.merge(df_count, on=label_name)
    df_with_count = df_with_count[df_with_count.counts >= drop_below]
    return df_with_count.drop(['counts'], axis=1)


def generate_train_and_test(raw_data):
    global x_train, x_test, y_train, y_test, x, y, xx_train, xx_test, yy_train, yy_test
    df = pd.read_csv(raw_data)
    df.rename(columns={'Unnamed: 0': 'vin'}, inplace=True)
    year_df = df
    year_df = drop_low_samples(data=df, label_name='productDate', drop_below=10)
    x = year_df.drop(['productDate', 'dateIndex', 'vin'], axis=1)
    y = year_df['productDate']
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    le = LabelEncoder().fit(y)
    year_labels = le.transform(y)
    year_classes = list(le.classes_)
    # print(year_classes)
    print(len(year_classes))
    # draw_pic(year_labels)
    seed = 3
    test_size = 0.1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    xx_train, xx_test, yy_train, yy_test = train_test_split(x, year_labels, test_size=test_size, random_state=seed)


def train_decision_tree():
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf = tree_clf.fit(x_train, y_train)
    return tree_clf


def train_random_forest():
    rfc = RandomForestClassifier()
    rfc = rfc.fit(x, y)
    return rfc


def train_extra_forest():
    etc = ExtraTreesClassifier()
    etc = etc.fit(x, y)
    return etc


def train_xgboost():
    xg_train = xgb.DMatrix(xx_train, label=yy_train)
    xg_eval = xgb.DMatrix(xx_test, label=yy_test)

    # 设置模型参数

    params = {
        'objective': 'multi:softmax',
        'eta': 0.08,
        'max_depth': 12,
        'num_class': 240,
        'lambda': 1.5,
    }

    watchlist = [(xg_train, 'train'), (xg_eval, 'test')]
    num_round = 180
    bst = xgb.train(params, xg_train, num_round, watchlist)

    # 模型预测

    pred = bst.predict(xg_eval)
    print(pred)

    # 模型评估

    # error_rate=np.sum(pred!=test.lable)/test.lable.shape[0]
    error_rate = np.sum(pred != yy_test) / yy_test.shape[0]

    print('测试集错误率(softmax):{}'.format(error_rate))

    accuray = 1 - error_rate
    print('测试集准确率：%.4f' % accuray)

    # 模型保存
    bst.save_model("./model_set/xgb02.model")


def evaluation_random_forest(rfc, x_data):
    rfc_predictions = rfc.predict(x_data)
    acc = accuracy_score(y_test, rfc_predictions)
    print("RANDOM FOREST => VEHICLE PRODUCT DATE ACC: {:.4%}".format(acc))
    return rfc_predictions


def evaluation_decision_tree(dt, x_data):
    dt_predictions = dt.predict(x_data)
    acc = accuracy_score(y_test, dt_predictions)
    print("DECISION TREE => VEHICLE PRODUCT DATE ACC: {:.4%}".format(acc))
    return dt_predictions


def evaluation_extra_forest(etc, x_data):
    etc_predictions = etc.predict(x_data)
    acc = accuracy_score(y_test, etc_predictions)
    print("EXTRA FOREST => VEHICLE PRODUCT DATE ACC: {:.4%}".format(acc))
    return etc_predictions


def evaluation_xgboost(xgboost, x_data):
    xgboost_predictions = xgboost.predict(x_data)
    acc = accuracy_score(y_test, xgboost_predictions)
    print("XGBOOST => VEHICLE PRODUCT DATE ACC: {:.4%}".format(acc))
    return xgboost_predictions


def single_decision_tree():
    res = train_decision_tree()
    score = cross_val_score(res, x, y)
    print("DECISION TREE => CROSS VALIDATION MEAN: {:.4%}".format(score.mean()))


def random_forest_classifier():
    res = train_random_forest()
    score = cross_val_score(res, x, y)
    print("RANDOM FOREST => CROSS VALIDATION MEAN: {:.4%}".format(score.mean()))


def extra_forest_classifier():
    res = train_extra_forest()
    score = cross_val_score(res, x, y)
    print("EXTRA FOREST => CROSS VALIDATION MEAN: {:.4%}".format(score.mean()))


def xgboost_():
    train_xgboost()
    # evaluation_xgboost(res, x_test)
    # score = res.cv(res, x, y)
    # print("XGBOOST => CROSS VALIDATION MEAN: {:.4%}".format(score.mean()))


def draw_pic(data):
    plt.hist(data, 536, rwidth=None, range=(0, 536), density=None)
    plt.savefig('C:\\Users\\wujs\\Desktop\\all.jpg')
    plt.show()
    plt.hist(data, 536, rwidth=None, range=(0, 50), density=None)
    plt.savefig('C:\\Users\\wujs\\Desktop\\0-50.jpg')
    plt.show()
    plt.hist(data, 536, rwidth=None, range=(200, 300), density=None)
    plt.savefig('C:\\Users\\wujs\\Desktop\\0-50.jpg')
    plt.show()
    plt.hist(data, 536, rwidth=None, range=(300, 536), density=None)
    plt.savefig('C:\\Users\\wujs\\Desktop\\300-all.jpg')
    plt.show()
    plt.hist(data, 536, rwidth=None, range=(500, 536), density=None)
    plt.savefig('C:\\Users\\wujs\\Desktop\\500-all.jpg')
    plt.show()


if __name__ == '__main__':
    data_path = "./processed_data/processed_data.csv"
    generate_train_and_test(data_path)
    # single_decision_tree()
    # random_forest_classifier()
    # extra_forest_classifier()
    xgboost_()
    # 模型加载
    # bst = xgb.Booster()
    # bst.load_model("./model_set/xgb01.model")
    # xg_eval = xgb.DMatrix(xx_test, label=yy_test)
    # pred = bst.predict(xg_eval)
    # print(pred)
