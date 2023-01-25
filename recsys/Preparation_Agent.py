import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from helper_functions import Helper
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import multiprocessing

# More ML Models
import xgboost as xgb
import xgboost
import sklearn
import statsmodels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from interpret.glassbox.ebm.ebm import ExplainableBoostingClassifier
import shap as shap

###################################################################################################
# usage agent       ###############################################################################
###################################################################################################
class Usage_Agent:
    import pandas as pd
    def __init__(self, input_df, device):
        self.input = input_df
        self.device = device

    # train test split
    # -------------------------------------------------------------------------------------------
    
    #train start: the day from which training starts
    def get_train_start(self, df):
        import datetime
        start_date = min(df.index) + datetime.timedelta(days=3)
        # determine train_start date 
        return str(start_date)[:10]
    
    def train_test_split(self, df, date, train_start = ''):
        if train_start == '':
            train_start = self.get_train_start(df)

        #restrict number of variables
        select_vars =  [str(self.device) + '_usage', str(self.device)+ '_usage_lag_1', str(self.device)+ '_usage_lag_2',	'active_last_2_days']
        df = df[select_vars]
        #spli train and test
        tomorrow = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        X_train = df.loc[train_start:date, df.columns != str(self.device) + '_usage']
        y_train = df.loc[train_start:date, df.columns == str(self.device) + '_usage']
        X_test  = df.loc[tomorrow, df.columns != str(self.device) + '_usage']
        y_test  = df.loc[tomorrow , df.columns == str(self.device) + '_usage']
        return X_train, y_train, X_test, y_test
    
    # model training and evaluation
    # -------------------------------------------------------------------------------------------
    def fit_Logit(self, X, y, max_iter=100):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=0, max_iter=max_iter).fit(X, y)

    def fit_knn(self, X, y, n_neighbors=10, leaf_size=30):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, algorithm="auto", n_jobs=-1).fit(X, y)

    def fit_random_forest(self, X, y, max_depth=10, n_estimators=500, max_features="sqrt"):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features, n_jobs=-1).fit(X, y)

    def fit_ADA(self, X, y, learning_rate=0.1, n_estimators=100):
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators).fit(X, y)

    def fit_XGB(self, X, y, learning_rate=0.1, max_depth=6, reg_lambda=1, reg_alpha=0):
        import xgboost
        return xgboost.XGBClassifier(verbosity=0, use_label_encoder=False, learning_rate=learning_rate, max_depth=max_depth, reg_lambda=reg_lambda, reg_alpha=reg_alpha).fit(X, y)

    def fit_EBM(self, X, y): 
        from interpret.glassbox.ebm.ebm import ExplainableBoostingClassifier  
        return ExplainableBoostingClassifier().fit(X,y)

    def fit_smLogit(self, X, y):
        import statsmodels
        return statsmodels.api.Logit(y, X).fit(disp=False)
    
    def fit(self, X, y, model_type, **args):
        model = None
        if model_type == "logit":
            model = self.fit_Logit(X, y, **args)
        elif model_type == "ada":
            model = self.fit_ADA(X, y, **args)
        elif model_type == "knn":
            model = self.fit_knn(X, y, **args)
        elif model_type == "random forest":
            model = self.fit_random_forest(X, y, **args)
        elif model_type == "xgboost":
            model = self.fit_XGB(X, y, **args)
        elif model_type == "ebm":
            model = self.fit_EBM(X,y, **args)
        elif model_type == "logit_sm":
            model = self.fit_smLogit(X, y)
        else:
            raise InputError("Unknown model type.")
        return model

    def predict(self, model, X):
        import sklearn
        import statsmodels
        from interpret.glassbox.ebm.ebm import ExplainableBoostingClassifier     
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier
        import xgboost
        import numpy as np
        import pandas
        res = 0
        cols = X.index
        for e in cols:
            if isinstance(X, pd.DataFrame):
                if e in X.columns:
                    res += 1
            if isinstance(X, pd.Series):
                if e in X.index:
                    res += 1
        X = np.array(X).reshape(-1, res)
        if type(model) == sklearn.linear_model.LogisticRegression:
            y_hat = model.predict_proba(X)[:,1]
        elif type(model) == sklearn.neighbors._classification.KNeighborsClassifier:
            y_hat = model.predict_proba(X)[:,1]
        elif type(model) == sklearn.ensemble._forest.RandomForestClassifier:
            y_hat = model.predict_proba(X)[:,1]
        elif type(model) ==  sklearn.ensemble._weight_boosting.AdaBoostClassifier:
            y_hat = model.predict_proba(X)[:,1]
        elif type(model) == xgboost.sklearn.XGBClassifier:
            y_hat = model.predict_proba(X)[:,1]
        elif type(model) == ExplainableBoostingClassifier:
            y_hat = model.predict_proba(X)[:,1]
        elif type(model) == statsmodels.discrete.discrete_model.BinaryResultsWrapper:
            y_hat = model.predict(X)
        else:
            raise InputError("Unknown model type.")

        return y_hat

    def auc(self, y_true, y_hat):
        import sklearn.metrics
        return sklearn.metrics.roc_auc_score(y_true, y_hat)
    
    def evaluate(
            self, df, model_type, train_start = '', predict_start="2014-01-01", predict_end=-1, return_errors=False,
            weather_sel=False, xai=False, **args
    ):
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        dates = pd.DataFrame(df.index)
        dates = dates.set_index(df.index)["last_updated"]
        predict_start = pd.to_datetime(predict_start)
        predict_end = (
            pd.to_datetime(dates.iloc[predict_end])
            if type(predict_end) == int
            else pd.to_datetime(predict_end)
        )
        dates = dates.loc[predict_start:predict_end]
        y_true = []
        y_hat_train = {}
        y_hat_test = []
        y_hat_lime = []
        y_hat_shap = []
        auc_train_dict = {}
        auc_test = []
        xai_time_lime = []
        xai_time_shap = []

        predictions_list = []

        if weather_sel:
            print('Crawl weather data....')
            # Add Weather
            ################################
            from meteostat import Point, Daily
            from datetime import datetime, timedelta

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            start = time[0]
            end = time[len(time) - 1]
            weather = Daily(lough, start, end)
            weather = weather.fetch()

            from sklearn.impute import KNNImputer
            import numpy as np

            headers = weather.columns.values

            empty_train_columns = []
            for col in weather.columns.values:
                if sum(weather[col].isnull()) == weather.shape[0]:
                    empty_train_columns.append(col)
            headers = np.setdiff1d(headers, empty_train_columns)

            imputer = KNNImputer(missing_values=np.nan, n_neighbors=7, weights="distance")
            weather = imputer.fit_transform(weather)
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")

            ################################

        if not xai:
            for date in tqdm(dates.index):
                errors = {}
                try:
                    X_train, y_train, X_test, y_test = self.train_test_split(
                        df, date, train_start
                    )
                    # fit model
                    model = self.fit(X_train, y_train, model_type, **args)
                    # predict
                    y_hat_train.update({date: self.predict(model, X_train)})
                    y_hat_test += list(self.predict(model, X_test))
                    # evaluate train data
                    auc_train_dict.update(
                        {date: self.auc(y_train, list(y_hat_train.values())[-1])}
                    )
                    y_true += list(y_test)

                except Exception as e:
                    errors[date] = e
        else:
            print('The explainability approaches in the Usage Agent are being evaluated for model: ' + str(model_type))
            print('Start evaluation with LIME and SHAP')
            import time
            import lime
            import shap as shap
            from lime import lime_tabular

            for date in tqdm(dates.index):
                errors = {}
                try:
                    X_train, y_train, X_test, y_test = self.train_test_split(
                        df, date, train_start
                    )
                    # fit model
                    model = self.fit(X_train, y_train, model_type)
                    # predict
                    y_hat_train.update({date: self.predict(model, X_train)})
                    y_hat_test += list(self.predict(model, X_test))
                    # evaluate train data
                    auc_train_dict.update(
                        {date: self.auc(y_train, list(y_hat_train.values())[-1])}
                    )
                    y_true += list(y_test)
                    start_time = time.time()

                    if model_type == "xgboost":
                        booster = model.get_booster()

                        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                                           feature_names=X_train.columns,
                                                                           kernel_width=3, verbose=False)

                    else:
                        explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train),
                                                                      mode="classification",
                                                                      feature_names=X_train.columns,
                                                                      categorical_features=[0])

                    if model_type == "xgboost":
                        exp = explainer.explain_instance(X_test, model.predict_proba)
                    else:
                        exp = explainer.explain_instance(data_row=X_test, predict_fn=model.predict_proba)

                    y_hat_lime += list(exp.local_pred)

                    # take time for each day:
                    end_time = time.time()
                    difference_time = end_time - start_time

                    xai_time_lime.append(difference_time)
                    # SHAP
                    # =========================================================================
                    start_time = time.time()

                    if model_type == "logit":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)


                    elif model_type == "ada":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)

                    elif model_type == "knn":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)


                    elif model_type == "random forest":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)

                    elif model_type == "xgboost":
                        explainer = shap.TreeExplainer(model, X_train, model_output='predict_proba')

                    else:
                        raise InputError("Unknown model type.")

                    base_value = explainer.expected_value[1]  # the mean prediction


                    shap_values = explainer.shap_values(
                        X_test)
                    contribution_to_class_1 = np.array(shap_values).sum(axis=1)[1]  # the red part of the diagram
                    shap_prediction = base_value + contribution_to_class_1
                    # Prediction from XAI:
                    y_hat_shap += list([shap_prediction])


                    # take time for each day:
                    end_time = time.time()
                    difference_time = end_time - start_time
                    xai_time_shap.append(difference_time)

                except Exception as e:
                    errors[date] = e

        auc_test = self.auc(y_true, y_hat_test)
        auc_train = np.mean(list(auc_train_dict.values()))
        predictions_list.append(y_true)
        predictions_list.append(y_hat_test)
        predictions_list.append(y_hat_lime)
        predictions_list.append(y_hat_shap)

        # Efficiency
        time_mean_lime = np.mean(xai_time_lime)
        time_mean_shap = np.mean(xai_time_shap)
        print('Mean time nedded by appraoches: ' + str(time_mean_lime) + ' ' + str(time_mean_shap))

        if return_errors:
            return auc_train, auc_test, auc_train_dict, time_mean_lime, time_mean_shap, predictions_list, errors
        else:
            return auc_train, auc_test, auc_train_dict, time_mean_lime, time_mean_shap, predictions_list
        
    # pipeline function: predicting device usage
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, model_type, train_start = '', weather_sel=False):

        if weather_sel:
            # Add Weather
            ################################
            from meteostat import Point, Daily
            from datetime import datetime, timedelta

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            start = time[0]
            end = time[len(time) - 1]
            weather = Daily(lough, start, end)
            weather = weather.fetch()

            from sklearn.impute import KNNImputer
            import numpy as np

            headers = weather.columns.values

            empty_train_columns = []
            for col in weather.columns.values:
                if sum(weather[col].isnull()) == weather.shape[0]:
                    empty_train_columns.append(col)
            headers = np.setdiff1d(headers, empty_train_columns)

            imputer = KNNImputer(missing_values=np.nan, n_neighbors=7, weights="distance")
            weather = imputer.fit_transform(weather)
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")
            ################################

        X_train, y_train, X_test, y_test = self.train_test_split(df, date, train_start)
        model = self.fit(X_train, y_train, model_type)
        return self.predict(model, X_test)


    # pipeline function: predicting device usage
    # -------------------------------------------------------------------------------------------
    def pipeline_xai(self, df, date, model_type, train_start = '', weather_sel=False):

        if weather_sel:
            # Add Weather
            ################################
            from meteostat import Point, Daily
            from datetime import datetime, timedelta

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            start = time[0]
            end = time[len(time) - 1]
            weather = Daily(lough, start, end)
            weather = weather.fetch()

            from sklearn.impute import KNNImputer
            import numpy as np

            headers = weather.columns.values

            empty_train_columns = []
            for col in weather.columns.values:
                if sum(weather[col].isnull()) == weather.shape[0]:
                    empty_train_columns.append(col)
            headers = np.setdiff1d(headers, empty_train_columns)

            imputer = KNNImputer(missing_values=np.nan, n_neighbors=7, weights="distance")
            weather = imputer.fit_transform(weather)
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")
            ################################

        X_train, y_train, X_test, y_test = self.train_test_split(df, date, train_start)
        model = self.fit(X_train, y_train, model_type)
        return self.predict(model, X_test), X_train, X_test, model