import re, time
from ast import literal_eval

import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor():
    def __init__(self,
                 load_path='./data/final_matched2.csv',
                 do_shuffle=True,
                 train_data_ratio=0.9,
                 random_seed=42):
        self.data = pd.read_csv(load_path, sep='┷', engine='python')
        self.title = self.data['title'].values.tolist()
        self.option_name = self.data['option_name'].values.tolist()
        self.result_target = None

        self.load_path = load_path
        self.do_shuffle = do_shuffle
        self.train_size = train_data_ratio
        self.random_seed = random_seed

    def __remove_special_tokens(self, col:list) -> list:
        new_col = []
        for idx, i in enumerate(col):
            if idx > 0 and isinstance(i, str):
                i = re.sub('[^가-힣a-zA-Z0-9 +-]', ' ', i).strip()
                new_col.append(i)
            else:
                new_col.append('')
        return new_col
    
    def __regularize_spacing(self, col:list) -> list:
        new_col = []
        for i in col:
            if isinstance(i, str):
                i = ' '.join(i.split()).strip()
                new_col.append(i)
            else:
                new_col.append('')
        return new_col

    def _clean_title_and_option_name_columns(self):
        self.title = self.__remove_special_tokens(self.title)
        self.title = self.__regularize_spacing(self.title)
        self.option_name = self.__remove_special_tokens(self.option_name)
        self.option_name = self.__regularize_spacing(self.option_name)

    def _result_target_to_nested_list(self):
        result_target = self.data['result_target'].values.tolist()
        new_result_target = []
        for idx, t in enumerate(result_target):
            if idx > 0:
                t = literal_eval(t)['sku'][0] 
                new_result_target.append(t)
            else:
                new_result_target.append('output')
        self.result_target = new_result_target

    def renew_dataframe(self):
        input_list = []
        self._clean_title_and_option_name_columns()
        self._result_target_to_nested_list()
        for idx, (t, o) in enumerate(zip(self.title, self.option_name)):
            if idx > 0:
                input_list.append(t + '|' + o)
            else:
                input_list.append('input')
        self.data['input'] = input_list
        self.data['output'] = self.result_target
        self.data = self.data.drop(columns=['index', 'title', 'option_name', 'result_target'])

    def shuffle_dataframe(self, shuffle=True):
        if shuffle:
            self.data.drop(index=0).reset_index(drop=True)
            self.data = self.data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        else:
            pass

    def save_dataframe_to_csv(self, df:pd.DataFrame, infix:str):
        save_path = self.load_path.replace('.csv', '.{}.csv'.format(infix))
        df.to_csv(save_path, sep=',', index=False)


    def print_dataframe_statistics(self, df:pd.DataFrame):
        print('컬럼명: {}'.format(df.columns.tolist()))
        print('행 개수: {}\n열 개수: {}'.format(*df.shape))
        print('전체 NaN 개수: {}'.format(df.isnull().sum().sum()))

    def preprocess(self):
        print('process raw data.......', end='\r')
        self.renew_dataframe()
        print('shuffle cleaned data...', end='\r', flush=True)
        self.shuffle_dataframe(self.do_shuffle)
        print('split data.............', end='\r', flush=True)
        train_df, test_df = train_test_split(self.data, train_size=self.train_size, random_state=self.random_seed)
        print('saving.................', end='\r', flush=True)
        self.save_dataframe_to_csv(train_df, 'train')
        self.save_dataframe_to_csv(test_df, 'test')
        print('saving........completed')

        print('\n[원본 데이터]')
        self.print_dataframe_statistics(self.data)
        print('\n[정제된 학습 데이터]')
        self.print_dataframe_statistics(train_df)
        print('\n[정제된 추론 데이터]')
        self.print_dataframe_statistics(test_df)

if __name__ == '__main__':
    start = time.time()
    proc = Preprocessor(
        load_path='./data/final_matched2.csv',
        do_shuffle=True,
        train_data_ratio=0.9,
        random_seed=42)
    proc.preprocess()
    end = time.time()

    print('Total time consumed: {:.2f}'.format(end - start))
        



    

# data['input'] = data['title'] + data['option_name'].apply(lambda x: ' ' + ('' if isinstance(x, float) else x))
# data['output'] = data['result_target'].apply(lambda x: literal_eval(x)['sku'][0])
# data.drop(columns=['title', 'option_name', 'result_target'], inplace=True)

# data.to_csv('C:\\Users\\builton\\Desktop\\final.mtch.mod.bt.csv')
