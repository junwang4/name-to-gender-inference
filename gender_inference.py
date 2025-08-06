import time
import os
import unidecode
import pandas as pd
import numpy as np
from sklearn import metrics

from nameparser import HumanName
import fire
import joblib

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn import metrics


def create_features(df, name_indexes=(0,), ngrams=(1,2,3,4,5,6,7,8), drop_column_parsed=True):
    # name_indexes: 0: given, 1: middle,  2: surname
    def gen_feature(str, dir='l', num=2):
        str = unidecode.unidecode(str)
        if len(str)<num:
            return ''
        else:
            char = str[(num-1):num] if dir=='l' else str[-1] if num==1 else str[-num:1-num]
            return char

    for ni in name_indexes:
        for dir in ('l', 'r',): # l: from left; r: from right
            for num in ngrams:
                feature_name = f'f{ni}_{dir}_{num}'
                df[feature_name] = df['parsed'].apply(lambda x: gen_feature(x[ni], dir, num))
                if 0:
                    print('----', feature_name, len(df[feature_name].unique()))
                    print(df[feature_name].value_counts()[:35], '\n')

    if 'gender' in df.columns:
        df.drop(['gender'], axis=1, inplace=True)
    if drop_column_parsed:
        df.drop(['parsed'], axis=1, inplace=True)


class Name2Gender:
    def __init__(self):
        self.model_folder = './models'

    def get_model_folder(self):
        return self.model_folder

    def get_fpath_of_lgb_model(self):
        return f'{self.get_model_folder()}/model_lgb.m'

    def get_fpath_of_label_encoder(self):
        return f'{self.get_model_folder()}/label_encoder.m'

    def predict_gender(self, fpath_in):
        def parse_human_name_to_list(x):
            hn = HumanName( unidecode.unidecode(x.lower()) )
            return [hn.first, hn.middle, hn.last]

        def read_and_create_features_for_fullnames(nrows=None):
            df_raw = pd.read_csv(fpath_in, nrows=nrows)
            print(df_raw.shape)
            df_test = df_raw.copy()
            df_test['parsed'] = df_test.name.apply(parse_human_name_to_list)
            print('... creating features ...')
            create_features(df_test)
            print(df_test.iloc[0])
            df_test = df_test.drop(["name"], axis=1)
            print('\n- done with feature creating\n')
            return df_test, df_raw

        nrows = None
        df, df_out = read_and_create_features_for_fullnames(nrows=nrows)
        print('- num of names to predict:', len(df))

        enc = joblib.load(self.get_fpath_of_label_encoder())
        model = joblib.load(self.get_fpath_of_lgb_model())

        X_test_vec = enc.transform(df)
        y_pred_proba = model.predict_proba(X_test_vec)

        df_out['male_confidence'] = y_pred_proba[:,1]

        fpath_out = self.get_fpath_of_prediction(fpath_in)
        df_out.to_csv(fpath_out, index=False, float_format="%.2f")
        print('\n- prediction result saved to:', fpath_out)

    def read_data_for_evaluation(self, fpath):
        df_test = pd.read_csv(fpath, na_filter=False)
        df_test.fillna('', inplace=True)

        # "first_name","middle_name","last_name","full_name","gender","origin"
        df_test = df_test[df_test.gender.isin(('f', 'm'))]
        df_test['gender'] = df_test['gender'].apply(lambda x: int(x=='m'))

        print(df_test.gender.value_counts())
        print()

        df_test_raw = df_test.copy().reset_index(drop=True)
        df_test_raw['gender'] = df_test_raw['gender'].astype(str)

        df_test['parsed'] = df_test.apply(lambda x: [x['first_name'], x['middle_name'], x['last_name']], axis=1)

        y_test = df_test['gender']
        create_features(df_test)

        for col in ["id", "name", "first_name", "middle_name", "last_name", "full_name", "origin"]:
            if col in df_test:
                df_test = df_test.drop(col, axis=1)
        return y_test, df_test, df_test_raw


    def my_transform(self, df, enc, fit=False):
        columns_of_num = [col for col in df.columns if col.startswith('num')]
        # in case there are numerical features, we can hstack() them with one-hot categorical ones

        df_num = None
        if len(columns_of_num) == 0:
            df_cat = df
        else:
            df_num = df[columns_of_num]
            df_cat = df.drop(columns_of_num, axis=1)

        if fit:
            vec_cat = enc.fit_transform(df_cat)
        else:
            vec_cat = enc.transform(df_cat)
        if df_num is None:
            return enc, vec_cat
        else:
            return enc, sparse.hstack((vec_cat, df_num.values))

    def evaluate_model(self, input):
        print(f'- loading pretrained models from {self.get_fpath_of_lgb_model()} \n')
        enc = joblib.load(self.get_fpath_of_label_encoder())
        model = joblib.load(self.get_fpath_of_lgb_model())

        y_test, X_test, df_test_raw = self.read_data_for_evaluation(input)
        print()
        print(X_test[:1])
        print()
        print(df_test_raw[:1])

        _, X_test_vec = self.my_transform(X_test, enc, fit=False)
        print("X_test_vec.shape: ", X_test_vec.shape, '\n')

        y_pred = model.predict(X_test_vec)
        print(metrics.classification_report(y_test, y_pred, digits=4))
        print(metrics.confusion_matrix(y_test, y_pred))

        y_pred_proba = model.predict_proba(X_test_vec)
        print(y_pred_proba.shape)
        df_proba = pd.DataFrame(y_pred_proba, columns=['c0', 'c1'])
        df_out = pd.concat((df_test_raw, df_proba), axis=1)

        df_out.to_csv(self.get_fpath_of_prediction(input), float_format="%.4f", index=False)

    def get_fpath_of_prediction(self, input):
        return input.replace('.csv', '.pred.csv')


    def advanced_error_analysis(self, input, TH_M=0.602, TH_F=0.543):
        #TH_M, TH_F = 0.602, 0.543  # for comparing with gender.io results
        #TH_M, TH_F = 0.82, 0.78  # 0.125 rejection rate, precision_min = 0.97, recall_min = 0.95
        #TH_M, TH_F = 0.80, 0.80  # as a baseline reference
        #TH_M, TH_F = 0.602, 0.543  # for comparing with gender.io results

        if not os.path.exists(self.get_fpath_of_prediction(input)):
            self.evaluate_model(input)

        print('read prediction data from:', self.get_fpath_of_prediction(input))
        df = pd.read_csv(self.get_fpath_of_prediction(input))


        df['pred'] = df['c1'].apply(lambda x: 1 if x>=TH_M else 0 if x<=1-TH_F else 2)
        df_male = df[df.gender==1]
        df_female = df[df.gender==0]

        print('\n\n*******************\n*\n* REJECTION RATE ANALYSIS \n*\n*******************\n')
        rejection_rate_male = len(df_male[df_male.pred==2])/len(df_male)
        rejection_rate_female = len(df_female[df_female.pred==2])/len(df_female)
        rejection_rate_both = len(df[df.pred==2])/len(df)
        print(f'- rejection rate (M): {rejection_rate_male:.4f}')
        print(f'- rejection rate (F): {rejection_rate_female:.4f}')
        print(f'- rejection rate/all: {rejection_rate_both:.4f}')
        
        print(f'\n\n- remove those rejected cases ...\n')

        print()
        print(metrics.confusion_matrix(df.gender, df.pred))
        print()
        df = df[df.pred != 2]  # remove rejected names
        print(f'- number of names after rejection: {len(df)}\n')
        print(metrics.classification_report(df.gender, df.pred, digits=3, labels=[0, 1]))

        print(f'- TH_M = {TH_M:.3f}   TH_F = {TH_F:.3f}')

        # estimate gender-api.com's performance
        def get_ytrue_ypred(cm):
            y_true = []
            y_pred = []
            for true_idx in range(cm.shape[0]):
                for pred_idx in range(cm.shape[1]):
                    cnt = cm[true_idx, pred_idx]
                    y_true.extend([true_idx] * cnt)
                    y_pred.extend([pred_idx] * cnt)
            return np.array(y_true), np.array(y_pred)

        gender_api_cm = [ [1750, 172, 46],
                          [110, 3573, 128],
                          [0, 0, 0]]
        y_true, y_pred = get_ytrue_ypred(np.array(gender_api_cm))
        print('\n\n*************************\n*\n* Performance of Gender API (gender-api.com)\n*\n*************************\n')
        print(metrics.confusion_matrix(y_true, y_pred))
        print(metrics.classification_report(y_true, y_pred, digits=3, labels=[0, 1]))


#--------------------
#
def run(task='', input='', confidence_male_threshold=0.602, confidence_female_threshold=0.543):
    n2g = Name2Gender()
    if task == 'predict':
        n2g.predict_gender(input)
    elif task == 'evaluate':
        n2g.evaluate_model(input)
    elif task == 'advanced_error_analysis':
        n2g.advanced_error_analysis(input, confidence_male_threshold, confidence_female_threshold)


if __name__ == '__main__':
    tic = time.time()
    fire.Fire(run)
    print(f'\n- time used: {time.time() - tic:.2f} seconds\n')
