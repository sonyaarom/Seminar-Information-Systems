#!/usr/bin/env python3
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

###################################################################################################
# preparation agent ###############################################################################
###################################################################################################
class Preparation_Agent:
#installing dependencies
    import pandas as pd
    #uploading data and simple data wrangling
    def __init__(self, dbfile, shiftable_devices):
        from helper_functions import Helper
        helper = Helper()
        self.input = helper.export_sql(dbfile)
        self.shiftable_devices = shiftable_devices

    def unpacking_attributes(self, df):
        import pandas as pd
        output = df.copy()
        output['shared_attributes']=output['shared_attributes'].apply(lambda x: x.replace('true','True'))
        output['shared_attributes']=output['shared_attributes'].apply(lambda x: x.replace('false','False'))
        output['shared_attributes']=output['shared_attributes'].apply(lambda x: x.replace('null','None'))

        output['shared_attributes']=output['shared_attributes'].apply(lambda dat: dict(eval(dat)))
        df2 = pd.json_normalize(output['shared_attributes'])
        result = pd.DataFrame( pd.concat([output,df2], axis = 1).drop('shared_attributes', axis = 1))
        result = result.dropna(axis = 1, thresh=int(0.95*(len(result.columns))))
        return result

    def access_shiftable_devices(self, df, attrs= 'all'):
        import pandas as pd
        trial = df.copy()
        trial.attributes_id = trial.attributes_id.dropna()
        trial.state= pd.to_numeric(trial['state'], errors='coerce').dropna()
        if attrs == 'all':
            w_data = trial[trial.unit_of_measurement.isin(['W'])]
            w_data = trial[trial.entity_id.isin(self.shiftable_devices)]
            w_data_long = w_data[['entity_id','last_updated','state']]
            w_data_wide = pd.pivot(w_data_long,  index = ['last_updated'], columns = 'entity_id', values = 'state')
        if attrs != 'all':
            w_data = trial[trial.unit_of_measurement.isin(['W']) & trial.attributes_id.isin([attrs])]
            w_data = trial[trial.entity_id.isin(self.shiftable_devices)]
            w_data_long = w_data[['entity_id','last_updated','state']]
            w_data_wide = pd.pivot(w_data_long,  index = ['last_updated'], columns = 'entity_id', values = 'state')
        result = w_data_wide.fillna(0).reset_index()
        return(result)
    
    #basic preprocessing
    # -------------------------------------------------------------------------------------------
    def outlier_truncation(self, series, factor=1.5, verbose=0):
        from tqdm import tqdm
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3-q1
        
        lower_bound = q1 - factor*iqr
        upper_bound = q3 + factor*iqr
        
        output = []
        counter = 0
        for item in (tqdm(series, desc=f'[outlier truncation: {series.name}]') if verbose != 0 else series):
            if item > upper_bound:
                output.append(int(upper_bound))
                counter += 1
            elif item < lower_bound:
                output.append(int(lower_bound))
                counter += 1
            else:
                output.append(item)
        print(f'[outlier truncation: {series.name}]: {counter} outliers were truncated.') if verbose != 0 else None 
        return output
    
    def scale(self, df, features='all', kind='MinMax', verbose=0):
        output = df.copy()
        features = output.select_dtypes(include=['int', 'float']).columns if features == 'all' else features

        if kind == 'MinMax':
            from sklearn.preprocessing import MinMaxScaler
            
            scaler = MinMaxScaler()
            output[features] = scaler.fit_transform(output[features])
            print('[MinMaxScaler] Finished scaling the data.') if verbose != 0 else None
        else:
            raise InputError('Chosen scaling method is not available.')
        return output 

    def get_timespan(self, df, start, timedelta_params):
        df.last_updated = pd.to_datetime(df.last_updated)
        df = df.set_index('last_updated')
        start = pd.to_datetime(start) if type(start) != type(pd.to_datetime('1970-01-01')) else start 
        end = start + pd.Timedelta(**timedelta_params)
        return df[start:end].reset_index()
    
    def truncate(self, df, features='all', factor=1.5, verbose=0):
        import time
        output = df.copy()
        features = df.select_dtypes(include=['int', 'float']).columns if features == 'all' else features

        for feature in features:
            time.sleep(0.2) if verbose != 0 else None
            row_nn = df[feature] != 0                                                                  # truncate only the values for which the device uses energy
            output.loc[row_nn, feature] = self.outlier_truncation(df.loc[row_nn, feature], factor=factor, verbose=verbose) # Truncatation factor = 1.5 * IQR
            print('\n') if verbose != 0 else None
        return output

    
    def plot_consumption(self, df, features='all', figsize='default', threshold=None, title='Consumption'):
        df = df.copy()
        features = [column for column in df.columns if column not in ['Unix', 'Issues']] if features == 'all' else features
        fig, ax = plt.subplots(figsize=figsize) if figsize != 'default' else plt.subplots()
        if threshold != None:
            df['threshold'] = [threshold]*df.shape[0]
            ax.plot(df['threshold'], color = 'tab:red')
        for feature in features:
            ax.plot(df[feature])
        ax.legend(['threshold'] + features) if threshold != None else ax.legend(features)
        ax.set_title(title);
    # feature creation
    # -------------------------------------------------------------------------------------------
    def get_device_usage(self, df, device, threshold):
        return (df.loc[:, device] > threshold).astype('int')

    def get_activity(self, df, active_appliances, threshold):
        import pandas as pd
        active = pd.DataFrame({appliance: df[appliance] > threshold for appliance in active_appliances})
        return active.apply(any, axis = 1).astype('int')

    def get_last_usage(self, series):
        import pandas as pd
        last_usage = []
        for idx in range(len(series)):
            shift = 1
            if pd.isna(series.shift(periods = 1)[idx]):
                shift = None
            else:
                while series.shift(periods = shift)[idx] == 0:
                    shift += 1
            last_usage.append(shift)
        return last_usage

    def get_last_usages(self, df, features):
        import pandas as pd

        output = pd.DataFrame()
        for feature in features:
            output['periods_since_last_'+str(feature)] = self.get_last_usage(df[feature])
        output.set_index(df.index, inplace=True)
        return output


    def get_time_feature(self, df, features='all'):
        import pandas as pd
        functions = {
            'hour': lambda df: df.index.hour, 
            'day_of_week': lambda df: df.index.dayofweek,
            'day_name': lambda df: df.index.day_name().astype('category'),
            'month': lambda df: df.index.month, 
            'month_name': lambda df: df.index.month_name().astype('category'),
            'weekend': lambda df: [int(x in ['Saturday', 'Sunday']) for x in  list(df.index.day_name())]
        }
        if features == 'all':
            output = pd.DataFrame({function[0]: function[1](df) for function in functions.items()})
        else:
            output = pd.DataFrame({function[0]: function[1](df) for function in functions.items() if function[0] in features})
        output.set_index(df.index, inplace=True)
        return output
    
    def get_time_lags(self, df, features, lags):
        import pandas as pd
        output = pd.DataFrame()
        for feature in features:
            for lag in lags:
                output[f'{feature}_lag_{lag}'] = df[feature].shift(periods=lag)
        return output

    #visualising threshold
    # ------------------------------------------------------------------------------------------- 
    def visualize_threshold(self, df, threshold, appliances, figsize=(18,5)):
        import pandas as pd
        # data prep
        for appliance in appliances:
            df[str(appliance) + '_usage'] = self.get_device_usage(df, appliance, threshold)
        df = df.join(self.get_time_feature(df))
        df['activity'] = self.get_activity(df, appliances, threshold)

        # plotting 
        import matplotlib.pyplot as plt

        usage_cols = [column for column in df.columns if str(column).endswith('_usage')]
        columns = ['activity'] + usage_cols

        fig, axes = plt.subplots(1,3, figsize=figsize)

        # hour
        hour = df.groupby('hour').mean()[columns]
        hour.plot(ax=axes[0])
        axes[0].set_ylim(-.1, 1.1);
        axes[0].set_title(f'[threshold: {round(threshold, 4)}] Activity ratio per hour')

        # week 
        usage_cols = [column for column in df.columns if str(column).endswith('_usage')]
        week = df.groupby('day_name').mean()[columns]
        week = week.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        week.plot(ax=axes[1])
        axes[1].set_ylim(-.1, 1.1);
        #axes[1].set_xticklabels(['']+list(week.index), rotation=90)
        axes[1].set_title(f'[threshold: {round(threshold, 4)}] Activity ratio per day of the week')

        # month
        usage_cols = [column for column in df.columns if str(column).endswith('_usage')]
        month = df.groupby('month').mean()[columns]
        month.plot(ax=axes[2])
        axes[2].set_ylim(-.1, 1.1);
        axes[2].set_title(f'[threshold: {round(threshold, 4)}] Activity ratio per month')
    def validate_thresholds(self, df, thresholds, appliances, figsize=(18,5)):

        for threshold in tqdm(thresholds):
            self.visualize_threshold(df, threshold, appliances, figsize)
        time.sleep(0.2)
        print('\n')
    
    #pipelines
    # -------------------------------------------------------------------------------------------
    #pipeline load
    def pipeline_load(self, df, params):
        from helper_functions import Helper
        import pandas as pd
        helper = Helper()
        
        df  = self.unpacking_attributes(self.input)
        df = self.access_shiftable_devices(df)
        
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        # df = self.truncate(df, **params['truncate'],)
        # scaled = self.scale(df, **params['scale'])
        # ignore scaling for now, we would just scale those variables, which does not make sense 
        # Index(['state_id', 'old_state_id', 'attributes_id', 'origin_idx', 'hash'], dtype='object')
        scaled = df.copy()

        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df = df.set_index('last_updated')

        scaled['last_updated'] = pd.to_datetime(scaled['last_updated'])
        scaled = scaled.set_index('last_updated')

        # aggregate
        df = helper.aggregate_load(df, **params['aggregate'])
        scaled = helper.aggregate_load(scaled, **params['aggregate'])

        # Get device usage and transform to energy consumption
        for device in params['shiftable_devices']:
            df[str(device) + '_usage'] = self.get_device_usage(df, device, **params['device'])
            output[device] = df.apply(lambda timestamp: timestamp[device] * timestamp[str(device) + '_usage'], axis = 1)

        return output, scaled, df
    #pipeline usage
    def pipeline_usage(self, df, params):
        from helper_functions import Helper
        import pandas as pd

        helper = Helper()

        df  = self.unpacking_attributes(self.input)
        df = self.access_shiftable_devices(df)
        
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        # df = self.truncate(df, **params['truncate'],)
        # scaled = self.scale(df, **params['scale'])
        # ignore scaling for now, we would just scale those variables, which does not make sense 
        # Index(['state_id', 'old_state_id', 'attributes_id', 'origin_idx', 'hash'], dtype='object')
        scaled = df.copy()
        
        # df['last_updated'] = pd.to_datetime(df['last_updated'])
        # df = df.set_index('last_updated')
        scaled['last_updated'] = pd.to_datetime(scaled['last_updated'])
        scaled = scaled.set_index('last_updated')
        
        # Aggregate to hour level
        scaled = helper.aggregate_load(scaled, **params['aggregate_hour'])

        # Activity feature
        output['activity'] = self.get_activity(scaled, **params['activity'])

        # Get device usage and transform to energy consumption
        for device in params['shiftable_devices']:
            output[str(device) + '_usage'] = self.get_device_usage(scaled, device, **params['device'])

        # aggregate and convert from mean to binary
        output = helper.aggregate(output, **params['aggregate_day'])
        output = output.apply(lambda x: (x > 0).astype('int'))

        # Last usage
        output = output.join(self.get_last_usages(output, output.columns))
        
        # Time features
        output = output.join(self.get_time_feature(output, **params['time']))

        # lags
        output = output.join(self.get_time_lags(output, ['activity'] + [str(device)+'_usage' for device in params['shiftable_devices']], [1,2,3]))
        output['active_last_2_days'] = ((output.activity_lag_1 == 1) | (output.activity_lag_2 == 1)).astype('int')

        # dummy coding
        output = pd.get_dummies(output, drop_first=True)
        return output

    #pipeline activity
    def pipeline_activity(self, df, params):
        from helper_functions import Helper
        import pandas as pd
        helper = Helper()
        df = df.copy()
        import pandas as pd
        output = pd.DataFrame()

        df  = self.unpacking_attributes(self.input)
        df = self.access_shiftable_devices(df)
        # Data cleaning
        # df = self.truncate(df, **params['truncate'],)
        # df = self.scale(df, **params['scale'])
        # ignore scaling for now, we would just scale those variables, which does not make sense 
        # Index(['state_id', 'old_state_id', 'attributes_id', 'origin_idx', 'hash'], dtype='object')

        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df = df.set_index('last_updated')
        # Aggregate to hour level
        df = helper.aggregate_load(df, **params['aggregate'])

        # Activity feature
        output['activity'] = self.get_activity(df, **params['activity'])
        
        ## Time feature
        output = output.join(self.get_time_feature(df, **params['time']))

        # Activity lags
        output = output.join(self.get_time_lags(output, **params['activity_lag']))

        # Dummy coding
        output = pd.get_dummies(output, drop_first=True)

        return output


###################################################################################################
# usage agent #####################################################################################
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
        end_date = min(df.index) + datetime.timedelta(days=3)
        # determine train_start date 
        return str(end_date)[:10]
    
    def train_test_split(self, df, date, train_start=''):
        if train_start == '':
            train_start = self.get_train_start(df)
        
        select_vars =  [str(self.device) + '_usage', 
                        str(self.device)+ '_usage_lag_1', 
                        str(self.device)+ '_usage_lag_2', 'active_last_2_days']
        

            
        # Add weather possibly
        if "temp" in df.columns:
            select_vars.append("temp")
            df["temp"].fillna(method="backfill", inplace=True)
        if "dwpt" in df.columns:
            select_vars.append("dwpt")
            df["dwpt"].fillna(method="backfill", inplace=True)
        if "rhum" in df.columns:
            select_vars.append("rhum")
            df["rhum"].fillna(method="backfill", inplace=True)
        if "wdir" in df.columns:
            select_vars.append("wdir")
            df["wdir"].fillna(method="backfill", inplace=True)
        if "wspd" in df.columns:
            select_vars.append("wspd")
            df["wspd"].fillna(method="backfill", inplace=True)
        
        df = df[select_vars]
        X_train = df.loc[train_start:date, df.columns != str(self.device) + "_usage"]
        y_train = df.loc[train_start:date, df.columns == str(self.device) + "_usage"]
        X_test = df.loc[date, df.columns != str(self.device) + "_usage"]
        y_test = df.loc[date, df.columns == str(self.device) + "_usage"]
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
        res = 3
        cols = ["temp", "dwpt", "rhum", "wdir", "wspd"]
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
    
###################################################################################################
# load agent ######################################################################################
###################################################################################################
class Load_Agent:
    def __init__(self, load_input_df):
        self.input = load_input_df

    # selecting the correct data, identifying device runs, creating load profiles
    # -------------------------------------------------------------------------------------------
    def prove_start_end_date(self, df, date):
        import pandas as pd

        start_date = (df.index[0]).strftime("%Y-%m-%d")
        end_date = date

        if len(df.loc[start_date]) < 24:
            start_date = (pd.to_datetime(start_date) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            df = df[start_date:end_date]
        else:
            df = df[:end_date]
        
        if end_date not in df:
            return df

        if len(df.loc[end_date]) < 24:
            end_new = (pd.to_datetime(end_date) - pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            df = df[:end_new]
        else:
            df = df[:end_date]
        return df

    def df_yesterday_date(self, df, date):
        import pandas as pd

        yesterday = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return df[:yesterday]

    def load_profile_raw(self, df, shiftable_devices):
        import pandas as pd

        hours = []
        for hour in range(1, 25):
            hours.append("h" + str(hour))
        df_hours = {}

        for idx, appliance in enumerate(
            shiftable_devices
        ):
            df_hours[appliance] = pd.DataFrame(index=None, columns=hours)
            column = df[appliance]

            for i in range(len(column)):

                if (i == 0) and (column[0] > 0):
                    df_hours[appliance].loc[0, "h" + str(1)] = column[0]

                elif (column[i - 1] == 0) and (column[i] > 0):
                    for j in range(0, 24):
                        if (i + j) < len(column):
                            if column[i + j] > 0:
                                df_hours[appliance].loc[i, "h" + str(j + 1)] = column[
                                    i + j
                                ]
        return df_hours

    def load_profile_cleaned(self, df_hours):
        import numpy as np

        for app in df_hours.keys():
            for i in df_hours[app].index:
                for j in df_hours[app].columns:
                    if np.isnan(df_hours[app].loc[i, j]):
                        df_hours[app].loc[i, j:] = 0
        return df_hours

    def load_profile(self, df_hours, shiftable_devices):
        import pandas as pd

        hours = df_hours[shiftable_devices[0]].columns
        loads = pd.DataFrame(columns=hours)

        for app in df_hours.keys():
            app_mean = df_hours[app].apply(lambda x: x.mean(), axis=0)
            for hour in app_mean.index:
                loads.loc[app, hour] = app_mean[hour]

        loads = loads.fillna(0)
        return loads

    # evaluating the performance of the load agent
    # -------------------------------------------------------------------------------------------
    def get_true_loads(self, shiftable_devices):
        true_loads = self.load_profile_raw(self.input, shiftable_devices)
        true_loads = self.load_profile_cleaned(true_loads)
        for device, loads in true_loads.items():
            true_loads[device].rename(
                index=dict(enumerate(self.input.index)), inplace=True
            )
        return true_loads

    def evaluate(self, shiftable_devices, date, metric="mse", aggregate=True, evaluation=False):
        from tqdm import tqdm
        import pandas as pd
        import numpy as np
        tqdm.pandas()

        if metric == "mse":
            import sklearn.metrics

            metric = sklearn.metrics.mean_squared_error

        true_loads = self.get_true_loads(shiftable_devices)

        scores = {}
        if not evaluation:
            for device in shiftable_devices:
                true_loads[device] = self.prove_start_end_date(true_loads[device], date)
                scores[device] = true_loads[device].progress_apply(
                    lambda row: metric(
                        row.values,
                        self.pipeline(
                            self.input, str(row.name)[:10], [device]
                        ).values.reshape(
                            -1,
                        ),
                    ),
                    axis=1,
                )
        else:
            for device in shiftable_devices:
                true_loads[device] = self.prove_start_end_date(true_loads[device], date)
                scores[device] = {}
                for idx in tqdm(true_loads[device].index):
                    date = str(idx)[:10]
                    y_true = true_loads[device].loc[idx, :].values
                    try:
                        y_hat = (df.loc[date][device].values.reshape(-1,))
                    except KeyError:
                        try:
                            y_hat = self.pipeline(
                                self.input, date, [device]
                            ).values.reshape(
                                -1,
                            )
                        except:
                            y_hat = np.full(24, 0)
                    scores[device][idx] = metric(y_true, y_hat)
                scores[device] = pd.Series(scores[device], dtype='float64')

        if aggregate:
            scores = {device: scores_df.mean() for device, scores_df in scores.items()}
        return scores

    # pipeline function: creating typical load profiles
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, shiftable_devices):
        df = self.prove_start_end_date(df, date)
        df = self.df_yesterday_date(df, date)
        df_hours = self.load_profile_raw(df, shiftable_devices)
        df_hours = self.load_profile_cleaned(df_hours)
        loads = self.load_profile(df_hours, shiftable_devices)
        return loads

###################################################################################################
# activity agent ##################################################################################
###################################################################################################
class Activity_Agent:
    def __init__(self, activity_input_df):
        self.input = activity_input_df

    # train test split
        # -------------------------------------------------------------------------------------------
    def get_Xtest(self, df, date, time_delta='all', target='activity'):
        import pandas as pd
        from helper_functions import Helper

        helper = Helper()
        
        if time_delta == 'all':
            output = df.loc[pd.to_datetime(date):, df.columns != target]
        else:
            df = helper.get_timespan(df, date, time_delta)
            output = df.loc[:, df.columns != target]
        return output

    def get_ytest(self, df, date, time_delta='all', target='activity'):
        import pandas as pd
        from helper_functions import Helper

        helper = Helper()
        
        if time_delta == 'all':
            output = df.loc[pd.to_datetime(date):, target]
        else:
            output = helper.get_timespan(df, date, time_delta)[target]
        return output

    def get_Xtrain(self, df, date, start=-30, target='activity'):
        import pandas as pd

        if type(start) == int:
            start = pd.to_datetime(date) + pd.Timedelta(days= start)
            start = pd.to_datetime('2013-11-01') if start < pd.to_datetime('2013-11-01') else start
        else:
            start = pd.to_datetime(start)
        end = pd.to_datetime(date) + pd.Timedelta(seconds=-1)
        return df.loc[start:end, df.columns != target]

    def get_ytrain(self, df, date, start=-30, target='activity'):
        import pandas as pd

        if type(start) == int:
            start = pd.to_datetime(date) + pd.Timedelta(days= start)
            start = pd.to_datetime('2013-11-01') if start < pd.to_datetime('2013-11-01') else start
        else:
            start = pd.to_datetime(start)
        end = pd.to_datetime(date) + pd.Timedelta(seconds=-1)
        return df.loc[start:end, target]
    
    #train start: the day from which training starts
    def get_train_start(self, df):
        import datetime
        end_date = min(df.index) + datetime.timedelta(days=3)
        # determine train_start date 
        return str(end_date)[:10]

    def train_test_split(self, df, date, train_start=-30, test_delta='all', target='activity'):
        if train_start == '':
            train_start = self.get_train_start(df)
        X_train = self.get_Xtrain(df, date, start=train_start, target=target)
        y_train = self.get_ytrain(df, date, start=train_start, target=target)
        X_test = self.get_Xtest(df, date, time_delta=test_delta, target=target)
        y_test = self.get_ytest(df, date, time_delta=test_delta, target=target)
        X_test = X_test.fillna(0)
        X_train = X_train.fillna(0)
        y_test = y_test.fillna(0)
        y_train = y_train.fillna(0)
        return X_train, y_train, X_test, y_test

    # model training and evaluation
    # -------------------------------------------------------------------------------------------
    def fit_Logit(self, X, y, max_iter=100):
        return LogisticRegression(random_state=0, max_iter=max_iter).fit(X, y)

    def fit_knn(self, X, y, n_neighbors=10, leaf_size=30):
        return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, algorithm="auto", n_jobs=-1).fit(X, y)

    def fit_random_forest(self, X, y, max_depth=10, n_estimators=500, max_features="sqrt"):
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features, n_jobs=-1).fit(X, y)

    def fit_ADA(self, X, y, learning_rate=0.1, n_estimators=100):
        return AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators).fit(X, y)

    def fit_XGB(self, X, y, learning_rate=0.1, max_depth=6, reg_lambda=1, reg_alpha=0):
        return xgboost.XGBClassifier(verbosity=0, use_label_encoder=False, learning_rate=learning_rate, max_depth=max_depth, reg_lambda=reg_lambda, reg_alpha=reg_alpha).fit(X, y)

    def fit_EBM(self, X, y): 
        return ExplainableBoostingClassifier().fit(X,y)

    def fit_smLogit(self, X, y):
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

        y_hat = pd.Series(y_hat, index=X.index)

        return y_hat

    def auc(self, y_true, y_hat):
        import sklearn.metrics
        return sklearn.metrics.roc_auc_score(y_true, y_hat)
    
    def plot_model_performance(self, auc_train, auc_test, ylim="default"):
        import matplotlib.pyplot as plt

        plt.plot(list(auc_train.keys()), list(auc_train.values()))
        plt.plot(list(auc_train.keys()), list(auc_test.values()))
        plt.xticks(list(auc_train.keys()), " ")
        plt.ylim(ylim) if ylim != "default" else None

    def evaluate(
            self, df, split_params, model_type, 
        predict_start='2013-11-30', 
        predict_end=-1, return_errors=False,
            xai=False, weather_sel=False, **args
        ):
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        dates = (
            pd.DataFrame(df.index)
            .set_index(df.index)["last_updated"]
            .apply(lambda date: str(date)[:10])
            .drop_duplicates()
        )
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
            print("Crawl weather data....")
            # Add Weather
            ################################
            from meteostat import Point, Hourly
            from datetime import datetime
            
            #### need to be coded differently!!####################################
            lough = Point(52.766593, -1.223511)
            
            time = df.index.to_series(name="time").tolist()
            weather = Hourly(lough, time[0], time[len(df) - 1])
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
            for date in tqdm(dates):
                errors = {}
                try:
                    X_train, y_train, X_test, y_test = self.train_test_split(
                        df, date, **split_params
                    )

                    # fit model
                    model = self.fit(X_train, y_train, model_type, **args)

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
            print('The explainability approaches in the Activity Agent are being evaluated for model: ' + str(model_type))
            print('Start evaluation with LIME and SHAP')
            import time
            import lime
            from lime import lime_tabular
            import shap as shap

            for date in tqdm(dates):
                errors = {}
                try:
                    X_train, y_train, X_test, y_test = self.train_test_split(
                        df, date, **split_params
                    )

                    # fit model
                    model = self.fit(X_train, y_train, model_type)

                    # self.predict uses predict_proba i.e. we get probability estimates and not classes
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

                    for local in range(len(X_test)):

                        if model_type == "xgboost":
                            exp = explainer.explain_instance(X_test.iloc[local, :].values, model.predict_proba)
                        else:
                            exp = explainer.explain_instance(data_row=X_test.iloc[local], predict_fn=model.predict_proba)

                        y_hat_lime += list(exp.local_pred)


                    # take time for each day:
                    end_time = time.time()
                    difference_time = end_time - start_time

                    xai_time_lime.append(difference_time)
                    # SHAP
                    # ==============================================================================
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

                    elif model_type == "logit_sm":
                        explainer = shap.TreeExplainer(model.predict, X_train_summary)

                    else:
                        raise InputError("Unknown model type.")

                    base_value = explainer.expected_value[1]  # the mean prediction

                    for local in range(len(X_test)):

                        shap_values = explainer.shap_values(
                            X_test.iloc[local, :])

                        contribution_to_class_1 = np.array(shap_values).sum(axis=1)[1]
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
    
    
    # pipeline function: predicting user activity
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, model_type, split_params, weather_sel=False):

        if weather_sel:

            # Add Weather
            ################################
            from meteostat import Point, Hourly
            from datetime import datetime

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            weather = Hourly(lough, time[0], time[len(df) - 1])
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

        # train test split
        X_train, y_train, X_test, y_test = self.train_test_split(
            df, date, **split_params
        )

        # fit model
        model = self.fit(X_train, y_train, model_type)

        # predict
        return self.predict(model, X_test)

    # pipeline function: predicting user activity with xai
        # -------------------------------------------------------------------------------------------
    def pipeline_xai(self, df, date, model_type, split_params, weather_sel=False):

        if weather_sel:

            # Add Weather
            ################################
            from meteostat import Point, Hourly
            from datetime import datetime

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            weather = Hourly(lough, time[0], time[len(df) - 1])
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

        # train test split
        X_train, y_train, X_test, y_test = self.train_test_split(
            df, date, **split_params
        )

        # fit model
        model = self.fit(X_train, y_train, model_type)

        # predict
        return self.predict(model, X_test), X_train, X_test, model
    
    
###################################################################################################
# price agent #####################################################################################
###################################################################################################
class Price_Agent(): 

    def return_day_ahead_prices(self, date, timezone = 'Europe/Brussels'):
        import pandas as pd
        import datetime
        from entsoe import EntsoePandasClient
        import pytz
        date = pd.to_datetime(date,format= '%Y-%m-%d')
        current_timezone = pytz.timezone(timezone)
        date = current_timezone.localize(date)
        start = (date - datetime.timedelta(days= 3)).normalize()
        end = (date + datetime.timedelta(days = 3)).normalize()
        country_code = 'DE_LU'
        client = EntsoePandasClient(api_key='6f67ccf4-edb3-4100-a850-969c73688627')
        df = client.query_day_ahead_prices(country_code = country_code, start = start, end = end)
        range = pd.date_range(start=date, freq="H", periods=48)
        df = df.loc[range]
        return df
    
    
    
###################################################################################################
# recommendation agent ############################################################################
###################################################################################################    
# The Original Recommendation Agent
# ===============================================================================================
class Recommendation_Agent:
    def __init__(
        self, activity_input, usage_input, load_input, price_input, shiftable_devices, model_type = 'random forest'):
        self.activity_input = activity_input
        self.usage_input = usage_input
        self.load_input = load_input
        self.price_input = price_input
        self.shiftable_devices = shiftable_devices
        self.Activity_Agent = Activity_Agent(activity_input)
        # create dictionary with Usage_Agent for each device
        self.Usage_Agent = {
            name: Usage_Agent(usage_input, name) for name in shiftable_devices
        }
        self.Load_Agent = Load_Agent(load_input)
        self.Price_Agent = Price_Agent()
        self.model_type = model_type

    # calculating costs
    # -------------------------------------------------------------------------------------------
    def electricity_prices_from_start_time(self, date):
        import pandas as pd

        prices_48 = self.Price_Agent.return_day_ahead_prices(date)
        prices_from_start_time = pd.DataFrame()
        for i in range(24):
            prices_from_start_time["Price_at_H+" + str(i)] = prices_48.shift(-i)
        # delete last 24 hours
        prices_from_start_time = prices_from_start_time[:-24]
        return prices_from_start_time

    def cost_by_starting_time(self, date, device, evaluation=False):
        import numpy as np
        import pandas as pd

        # get electriciy prices following every device starting hour with previously defined function
        prices = self.electricity_prices_from_start_time(date)
        # build up table with typical load profile repeated for every hour (see Load_Agent)
        if not evaluation:
            device_load = self.Load_Agent.pipeline(
                self.load_input, date, self.shiftable_devices
            ).loc[device]
        else:
            # get device load for one date
            device_load = evaluation["load"][date].loc[device]
        device_load = pd.concat([device_load] * 24, axis=1)
        # multiply both tables and aggregate costs for each starting hour
        costs = np.array(prices) * np.array(device_load)
        costs = np.sum(costs, axis=0)
        # return an array of size 24 containing the total cost at each staring hour.
        return costs

    # creating recommendations
    # -------------------------------------------------------------------------------------------
    def recommend_by_device(
        self,
        date,
        device,
        activity_prob_threshold,
        usage_prob_threshold,
        evaluation=False,
        weather_sel=False
    ):
        import numpy as np

        # add split params as input
        # IN PARTICULAR --> Specify date to start training
        split_params = {
            "train_start": "",
            "test_delta": {"days": 1, "seconds": -1},
            "target": "activity",
        }
        # compute costs by launching time:
        costs = self.cost_by_starting_time(date, device, evaluation=evaluation)
        # compute activity probabilities
        if not evaluation:
            if weather_sel:
                activity_probs = self.Activity_Agent.pipeline(self.activity_input, date, self.model_type, split_params, weather_sel=True)
            else:
                activity_probs = self.Activity_Agent.pipeline(self.activity_input, date, self.model_type, split_params)
        else:
            # get activity probs for date
            activity_probs = evaluation["activity"][date]

        # set values above threshold to 1. Values below to Inf
        # (vector will be multiplied by costs, so that hours of little activity likelihood get cost = Inf)
        activity_probs = np.where(activity_probs >= activity_prob_threshold, 1, float("Inf"))

        # add a flag in case all hours have likelihood smaller than threshold
        no_recommend_flag_activity = 0
        if np.min(activity_probs) == float("Inf"):
            no_recommend_flag_activity = 1

        # compute cheapest hour from likely ones
        best_hour = np.argmin(np.array(costs) * np.array(activity_probs))

        # compute likelihood of usage:
        if not evaluation:
            usage_prob = self.Usage_Agent[device].pipeline(self.usage_input, date, self.model_type, split_params["train_start"])
        else:
            # get usage probs
            name = ("usage_" + device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            usage_prob = evaluation[name][date]


        no_recommend_flag_usage = 0
        if usage_prob < usage_prob_threshold:
            no_recommend_flag_usage = 1

        return {
            "recommendation_date": [date],
            "device": [device],
            "best_launch_hour": [best_hour],
            "no_recommend_flag_activity": [no_recommend_flag_activity],
            "no_recommend_flag_usage": [no_recommend_flag_usage],
            "recommendation": [
                best_hour
                if (no_recommend_flag_activity == 0 and no_recommend_flag_usage == 0)
                else np.nan
            ],
        }


    def visualize_recommendation_by_device(self, dict):
        import datetime
        recommendation_date = str(dict['recommendation_date'][0])
        recommendation_date = datetime.datetime.strptime(recommendation_date, '%Y-%m-%d')
        best_launch_hour = dict['best_launch_hour'][0]
        recommendation_date = recommendation_date.replace(hour=best_launch_hour)
        recommendation_date = recommendation_date.strftime(format = "%d.%m.%Y %H:%M")
        device = dict['device'][0]
        if (dict['no_recommend_flag_activity'][0]== 0 and dict['no_recommend_flag_usage'][0]==0) == True:
            return print('You have one recommendation for the following device: ' + str(device) + '\nPlease use it on ' + recommendation_date[0:10] + ' at '+ recommendation_date[11:]+'.')

    # pipeline function: create recommendations
    # -------------------------------------------------------------------------------------------
    def pipeline(self, date, activity_prob_threshold, usage_prob_threshold, evaluation=False, weather_sel=False):
        import pandas as pd

        recommendations_by_device = self.recommend_by_device(
            date,
            self.shiftable_devices[0],
            activity_prob_threshold,
            usage_prob_threshold,
            evaluation=evaluation,
        )
        recommendations_table = pd.DataFrame.from_dict(recommendations_by_device)

        for device in self.shiftable_devices[1:]:
            if weather_sel:
                recommendations_by_device = self.recommend_by_device(
                    date,
                    device,
                    activity_prob_threshold,
                    usage_prob_threshold,
                    evaluation=evaluation,
                    weather_sel=True
                )
            else:
                recommendations_by_device = self.recommend_by_device(
                    date,
                    device,
                    activity_prob_threshold,
                    usage_prob_threshold,
                    evaluation=evaluation,
                )
            recommendations_table = recommendations_table.append(
                pd.DataFrame.from_dict(recommendations_by_device)
            )
        return recommendations_table

    def visualize_recommendation(self, recommendations_table, price):
        import datetime
        for i in range(len(recommendations_table)):
            date_and_time = recommendations_table.recommendation_date.iloc[i] + ':' + str(recommendations_table.best_launch_hour.iloc[i])

            date_and_time = datetime.datetime.strptime(date_and_time, '%Y-%m-%d:%H')

            date_and_time_show = date_and_time.strftime(format = "%d.%m.%Y %H:%M")
            date_and_time_price = date_and_time.strftime(format = "%Y-%m-%d %H:%M:%S")
            price = price.filter(like=date_and_time_price, axis=0)['Price_at_H+0'].iloc[0]
            output = print('You have a recommendation for the following device: ' + str(recommendations_table.device.iloc[i]) + '\n\nPlease use the device on the ' + date_and_time_show[0:10] + ' at ' + date_and_time_show[11:] + ' oclock because it costs you only ' + str(price) + ' €.\n')
            if (recommendations_table.no_recommend_flag_activity.iloc[i]==0 and recommendations_table.no_recommend_flag_usage.iloc[i]==0) == True:
                return output
            else:
                return
