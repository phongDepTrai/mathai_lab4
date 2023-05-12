import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import datetime
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def scale_output(data, v_min, v_max):
    t_max = 0.9
    t_min = 0.1
    target = t_min + ((data - v_min) / (v_max - v_min)) * (t_max - t_min)
    return target


def rescale_output(predict, v_min, v_max):
    t_max = 0.9
    t_min = 0.1
    exp = v_min + ((predict - t_min) / (t_max - t_min)) * (v_max - v_min)
    return exp


def feature_engineering(df):
    df_data = df.copy()

    feature = {
        'categorical': {
            'MSSubClass': ['20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90', '120', '150', '160', '180',
                           '190'],
            'MSZoning': ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
            'Alley': ['Grvl', 'Pave', 'None'],
            'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
            'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
            'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards',
                             'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                             'Names', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer',
                             'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
            'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
            'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
            'BldgType': ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'],
            'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
            'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
            'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
            'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc',
                            'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                            'VinylSd', 'Wd Sdng', 'WdShing'],
            'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc',
                            'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                            'VinylSd', 'Wd Sdng', 'WdShing'],
            'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],
            'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
            'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
            'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
            'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
            'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'None'],
            'GarageFinish': ['Fin', 'RFn', 'Unf', 'None'],
            'PavedDrive': ['Y', 'P', 'N'],
            'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'None'],
            'MoSold': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            'YrSold': ['2006', '2007', '2008', '2009', '2010'],
            'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
            'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
        },
        'binary': {
            'Street': ['Pave', 'Grvl'],
            'CentralAir': ['Y', 'N']
        },
        'ordinal': {
            'LotShape': ['None', 'IR3', 'IR2', 'IR1', 'Reg'],
            'Utilities': ['None', 'NoSeWa', 'NoSewr', 'AllPub'],
            'LandSlope': ['None', 'Sev', 'Mod', 'Gtl'],
            'ExterQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'ExterCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
            'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'HeatingQC': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'KitchenQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
            'PoolQC': ['None', 'Fa', 'Ta', 'Gd', 'Ex']
        },
    }

    selected = []
    for cname in df_data.columns:
        # binary
        if cname in feature['binary']:  # Chuy?n các thu?c tính binary thành d?ng 0/1
            default_value = feature['binary'][cname][0]
            feature_name = cname + "_is_" + default_value
            selected.append(feature_name)
            df_data[feature_name] = df_data[cname].apply(lambda x: int(x == default_value))
            # categorical
        elif cname in feature['categorical']:  # Chuy?n các thu?c tính Categorical thành d?ng One-hot vector
            values = feature['categorical'][cname]
            for val in values:
                try:
                    new_name = "{}_{}".format(cname, val)

                    selected.append(new_name)
                    df_data[new_name] = df_data[cname].apply(lambda x: int(x == val))
                except Exception as err:
                    print("One-hot encoding for {}_{}. Error: {}".format(cname, val, err))
        # ordinal
        elif cname in feature['ordinal']:
            new_name = cname + "_ordinal"
            selected.append(new_name)
            df_data[new_name] = df_data[cname].apply(lambda x: int(feature['ordinal'][cname].index(x)))
        # numeric
        else:
            selected.append(cname)

    return df_data[selected]


def fill_missing_data(df):
    df_data = df.copy()
    categoricals = []
    for cname, dtype in df_data.dtypes.items():
        if dtype == 'object':
            categoricals.append(cname)
    # Categorical
    df_data[categoricals] = df_data[categoricals].fillna('None')

    # Numeric
    for cname in df_data.columns:
        if cname not in categoricals:
            df_data[cname] = df_data[cname].fillna(0)
    return df_data


def get_linear_data():
    df_train = pd.read_csv('./train.csv')
    # remove outlier
    df_train = df_train[df_train["GrLivArea"] < 4500]
    # normalize SalePrice
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    # remove missing
    df_train.drop(['PoolQC', 'MiscFeature'], axis=1, inplace=True)
    df_train = df_train[~df_train['Electrical'].isna()]

    # number to categorical
    cols = ["MSSubClass", "YrSold", 'MoSold']
    df_train[cols] = df_train[cols].astype(str)

    # remove high correlation data
    col = ['GarageArea', '1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt']
    df_train.drop(col, axis=1, inplace=True)

    # fill missing
    df_train = fill_missing_data(df_train)

    # create new feature
    df_train['TotalPorchSF'] = df_train['OpenPorchSF'] + df_train['EnclosedPorch'] \
                               + df_train['3SsnPorch'] + df_train['ScreenPorch']
    df_train['TotalBaths'] = df_train['BsmtFullBath'] + df_train['FullBath'] \
                             + 0.5 * (df_train['BsmtHalfBath'] + df_train['HalfBath'])
    df_train['TotalAreaSF'] = df_train['TotalBsmtSF'] + df_train['GrLivArea']
    df_train['Age'] = df_train['YrSold'].astype('int64') - df_train['YearBuilt']

    # feature engineering
    df_train = feature_engineering(df_train)

    # remove zero column
    for col in df_train.columns:
        if not any(df_train[col]):
            df_train.drop([col], axis=1, inplace=True)

    # save y
    y = df_train['SalePrice']
    df_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
    return df_train, y

def get_raw_data():
    df_train = pd.read_csv('./train.csv')
    # fill missing
    df_train = fill_missing_data(df_train)

    # feature engineering
    df_train = feature_engineering(df_train)

    # save y
    y = df_train['SalePrice']
    df_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
    return df_train, y

def linear_regression():
    # Linear regression
    # linear_x, linear_y = get_linear_data()
    linear_x, linear_y = get_raw_data()
    linear_y = np.log(linear_y)
    x_train, x_val, y_train, y_val = train_test_split(linear_x, linear_y, test_size=0.2, random_state=1)
    linear_regression_model = LinearRegression()
    # train
    linear_regression_model.fit(x_train, y_train)
    # valid
    y_predict = linear_regression_model.predict(x_val)

    linear_error = mean_squared_error(y_val, y_predict)
    print("Linear Error: ", linear_error)

    plt.plot(y_predict, y_val, 'ro')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')

    plt.xlabel("Predict (log scale)")
    plt.ylabel("Groundtruth (log scale)")
    plt.savefig("./figs/linear_regression_no_data.png")


def get_model(input_dim, params):
    """
    params =    [[num hidden layer 1, activation layer 1],
                [num hidden layer 2, activation layer 2]],
                ...
    """
    model = Sequential()  # a model consisting of successive layers
    for num_hidden, activation in params:
        model.add(Dense(num_hidden, input_dim=input_dim,
                        kernel_initializer='normal',
                        activation=activation))
        input_dim = num_hidden
    model.add(Dense(1, kernel_initializer='normal'))
    return model


def neural_network():
    # use linear data
    # linear_x, linear_y = get_linear_data()
    linear_x, linear_y = get_raw_data()
    linear_x = np.matrix(linear_x)
    linear_y = np.matrix(linear_y)
    scale_x = MinMaxScaler()
    scale_y = MinMaxScaler()

    scale_x.fit(linear_x)
    linear_x = scale_x.transform(linear_x)

    linear_y = linear_y.reshape((1460, 1))
    scale_y.fit(linear_y)
    linear_y = scale_y.transform(linear_y)

    x_train, x_val, y_train, y_val = train_test_split(linear_x, linear_y, test_size=0.2, random_state=1)

    input_dim = linear_x.shape[1]
    print("input_dim: ", input_dim)

    # gte model 1
    params_1 = [[256, "tanh"], [128, "tanh"], [64, "tanh"]]
    # params_1 = [[256, "relu"], [128, "relu"], [64, "relu"], [32, "relu"], [16, "relu"]]
    model_1 = get_model(input_dim, params_1)
    # model_1.compile(loss='MSE',
    #                 optimizer='sgd')

    # model_1.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adadelta())
    model_1.compile(loss='mean_squared_error', optimizer="sgd")
    print(model_1.summary())
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_1.fit(x_train, y_train,
                epochs=300, batch_size=4,
                validation_data=(x_val, y_val),
                callbacks=[tensorboard_callback])

    y_predict = model_1.predict(x_val)
    y_predict = scale_y.inverse_transform(y_predict)
    y_val = scale_y.inverse_transform(y_val)

    y_predict = np.log(y_predict)
    y_val = np.log(y_val)
    model_1_error = mean_squared_error(y_predict, y_val)
    print("model_1 Error: ", model_1_error)

    plt.plot(y_predict, y_val, 'ro')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')

    plt.xlabel("Predict (log scale)")
    plt.ylabel("Groundtruth (log scale)")
    plt.savefig("./figs/NN_no_data.png")


def main():
    linear_regression()
    # neural_network()


if __name__ == "__main__":
    main()
