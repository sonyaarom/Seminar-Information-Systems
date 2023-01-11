class Helper:

    @staticmethod
    def read_txt(filename):
        fl = open(filename, 'r')
        print(fl.read())
        fl.close


    def get_timespan(self, df, start, timedelta_params):
        import pandas as pd

        start = pd.to_datetime(start) if type(start) != type(pd.to_datetime('1970-01-01')) else start 
        end = start + pd.Timedelta(**timedelta_params)
        return df[start:end]


    @staticmethod
    def load_txt(filename):
        fl = open(filename, 'r')
        output = fl.read()
        fl.close
        return output


    def get_column_labels(self, filename):
        columns = {}
        readme = self.load_txt(filename)
        temp = readme[readme.find('\nHouse'):]

        for house in range(1, 22):
            cols = {}
            temp = readme[readme.find('\nHouse '+str(house)):]
    
            for idx in range(10):
                start = temp.find(str(idx)+'.')+2
                stop = temp.find(',') if temp.find(',') < temp.find('\n\t') else temp.find('\n\t')
                cols.update({'Appliance'+str(idx):temp[start:stop]})
                temp = temp[stop+1:]

            columns.update({house: cols})
        return columns


    def load_household(self, REFIT_dir, house_id):
        import pandas as pd

        data_sets = {id:f'CLEAN_House{id}.csv' for id in range(1,22)}
        filename = REFIT_dir + data_sets[house_id]

        readme = REFIT_dir + 'REFIT_Readme.txt'
        columns = self.get_column_labels(readme)

        house = pd.read_csv(filename)
        house.rename(columns=columns[house_id], inplace=True)
        house.set_index(pd.DatetimeIndex(house['Time']), inplace=True)
        return house

    def aggregate(self, df, resample_param):
        return df.resample(resample_param).mean().copy()

    def aggregate_load(self, df, resample_param = '60T'):
        import numpy as np
        output = df.copy()
        output = output.resample(resample_param).mean()
        output = output.replace(np.nan, 0)
        return output


    def plot_consumption(self, df, features='all', figsize='default', threshold=None, title='Consumption'):
        import matplotlib.pyplot as plt

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

    def create_day_ahead_prices_df(self, FILE_PATH, filename):
        import pandas as pd
        electricity_prices1 = pd.read_csv(FILE_PATH + filename)
        electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
        electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.replace("2015", "2013")

        electricity_prices2 = pd.read_csv(FILE_PATH + filename)
        electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
        electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.replace("2015", "2014")

        electricity_prices3 = pd.read_csv(FILE_PATH + filename)
        electricity_prices3["MTU (UTC)"] =  electricity_prices3["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]

        electricity_prices = pd.concat([electricity_prices1, electricity_prices2, electricity_prices3])
        electricity_prices.columns = ["Time", "Price"]
        electricity_prices = electricity_prices.set_index(pd.DatetimeIndex(electricity_prices['Time']), drop = True)
        electricity_prices = electricity_prices["Price"]
        return electricity_prices
    
    
    
    def concat_household_scores(self, agent_scores):
        import pandas as pd
        df_names = list(list(agent_scores.values())[0].keys())
        output = {}
        for name in df_names:
            output[name] = pd.concat([scores[name] for household, scores in agent_scores.items()])
        return pd.concat(output, axis=1)
    
    
    def shiftable_device_legend(self, EXPORT_PATH):
        from os import walk
        import json
        import pandas as pd
        # get config files stored at the export path
        _, _, filenames = next(walk(EXPORT_PATH))
        config_files = [file for file in filenames if file.find('config.json') != -1]

        legend_shiftable_devices = pd.DataFrame()
        for config_file in config_files:
            config = json.load(open(EXPORT_PATH+config_file, 'r'))
            household_id = config['data']['household']
            devices = config['user_input']['shiftable_devices']
            i = 0
            for device in devices:
                legend_shiftable_devices.loc[household_id, i] = device
                i += 1

        legend_shiftable_devices.sort_index(inplace=True)        
        legend_shiftable_devices.columns.name = 'device'
        legend_shiftable_devices.index.name = 'household'
        return legend_shiftable_devices
    
    
    def export_sql(self, file):
        import sqlite3
        import pandas as pd  
        with sqlite3.connect(file) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM states")
            states = cur.fetchall()
        from_states_db = []
        for result in states:
            result = list(result)
            from_states_db.append(result)
        columns = ["state_id","entity_id","state","attributes","event_id","last_changed","last_updated","old_state_id","attributes_id","context_id","context_user_id","context_parent_id","origin_idx"]
        states_df = pd.DataFrame(from_states_db, columns = columns)

        with sqlite3.connect(file) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM state_attributes WHERE last_updated > DATETIME('now', '-30 day')")
            state_attributes = cur.fetchall()
        from_state_attributes_db = []
        for result in state_attributes:
            result = list(result)
            from_state_attributes_db.append(result)
        columns = ["attributes_id","hash","shared_attributes"]
        state_attributes_df = pd.DataFrame(from_state_attributes_db, columns = columns)

        output = pd.merge(states_df, state_attributes_df, how= "left", on = 'attributes_id')
        return output
