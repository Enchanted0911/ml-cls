import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from main import drop_low_samples


global x_train, x_test, y_train, y_test, x, y


def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(17, 1), padding='same'))
    model.add(Conv1D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(Flatten())
    model.add(Dense(261, activation='softmax'))
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
    return model


def evaluation():
    global x_train, x_test, y_train, y_test, x, y
    # 加载模型用做预测
    json_file = open("./model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 分类准确率
    print("The accuracy of the classification model:")
    scores = loaded_model.evaluate(x_test, y_test, verbose=0)
    print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))

    # 输出预测类别
    predicted = loaded_model.predict(x)
    predicted_label = np.argmax(predicted, axis=1)
    print("predicted label:\n " + str(predicted_label))


def train():
    # 训练分类器
    estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=32, verbose=1)
    estimator.fit(x_train, y_train, validation_data=[x_test, y_test], validation_split=0.2, validation_freq=10)

    # 将其模型转换为json
    model_json = estimator.model.to_json()
    with open("./model.json", 'w') as json_file:
        json_file.write(model_json)  # 权重不在json中,只保存网络结构
    estimator.model.save_weights('model.h5')


def data_generator(data_path):
    global x_train, x_test, y_train, y_test, x, y
    # 载入数据
    df = pd.read_csv(data_path)
    df.rename(columns={'Unnamed: 0': 'vin'}, inplace=True)
    year_df = drop_low_samples(data=df, label_name='productDate', drop_below=5)
    x = year_df.drop(['productDate', 'dateIndex', 'vin'], axis=1)
    y = year_df['productDate']
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    le = LabelEncoder().fit(y)
    year_labels = le.transform(y)
    x = np.expand_dims(x, axis=2)
    y_one_hot = np_utils.to_categorical(year_labels)

    # 划分训练集，测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.3, random_state=0)


if __name__ == '__main__':
    data_path = "./processed_data/processed_data.csv"
    data_generator(data_path)
    # train()
    evaluation()
