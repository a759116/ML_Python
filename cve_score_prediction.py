#!/usr/bin/env python
# coding: utf-8

# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression


# ## Pre-processing and data preparation


# function to create training and test data
def simple_split(data,y,length,split_mark=0.7):
    if split_mark > 0. and split_mark < 1.0:
        n = int(length*split_mark)
    else:
        n = int(split_mark)
    X_train = data[:n].copy()
    X_test = data[n:].copy()
    y_train = y[:n].copy()
    y_test = y[n:].copy()
    
    return X_train, X_test, y_train, y_test


# load data
data = pd.read_csv("data/cve.csv")  
#data.head()
# split data into training and test sets
X_train, X_test, y_train, y_test = simple_split(data.summary,data.cvss,len(data))


# depends on cve description analysis, stop words will remove noise
stop_words = text.ENGLISH_STOP_WORDS.union(['22'])
# will convert text to number vector
vectorizer = CountVectorizer(stop_words=stop_words)
# convert text into word features
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


feature_names = vectorizer.get_feature_names()
# print("Number of features: {}".format(len(feature_names)))
# print("First 20 features: {}\n".format(feature_names[:20]))
# print("Middle 20 features: {}\n".format(feature_names[len(feature_names)//2 - 20:len(feature_names)//2]))
# print("Last 20 features: {}\n".format(feature_names[len(feature_names) - 20:]))


# ## Multivariate Regression Model

# ### Helper methods to build and evaluate liner model

# Helper methods to build and evaluate liner model
from sklearn.linear_model import LinearRegression

def train_liner_regression_model(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    return lr_model

def evaluate_liner_regression_model(lr_model, X_train, y_train, X_test, y_test):
    
    model_score={}
    
    y_pred = lr_model.predict(X_test)
    
    # The mean squared error
    mse = mean_squared_error(y_test, y_pred)
    model_score['mean_squared_error']=round(mse,4)
    # print('Mean squared error: %.2f'% mse)
    # The root mean squared error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_score['root_mean_squared_error']=round(rmse,4)
    # print('Root Mean squared error: %.2f'% rmse)
    # The coefficient of determination: 1 is perfect prediction
    coef_determination = r2_score(y_test, y_pred)
    model_score['coef_determination']=round(coef_determination,4)
    # print('Coefficient of determination: %.2f'% coef_determination)
    
    # R-squared value: R-squared is between 0 and 1, 
    # Higher values are better because it means that more variance is explained by the model.
    #rsq = lr_model.score(X_train, y_train)
    rsq = lr_model.score(X_test, y_test)
    model_score['R-squared']=round(rsq,4)
    # print('R-squared value: %.2f'% rsq)
    
    return y_pred,model_score 


# ### Helper Method for Feature selection

def select_features(X_train, y_train, X_test, score_func=f_regression, number_of_features='all'):
    # configure to select all features
    fs = SelectKBest(score_func, k=number_of_features)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    
    return X_train_fs, X_test_fs, fs


# ### Train and Evaluate Linear Regression Model with different set of features

def eval_model_for_features(score_func=f_regression, number_of_features='all'):
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, score_func,number_of_features)
    lr_model = train_liner_regression_model(X_train_fs, y_train)
    y_pred, model_score = evaluate_liner_regression_model(lr_model, X_train_fs, y_train, X_test_fs, y_test)
    
    return lr_model, fs, y_pred, model_score


# ### Main
import matplotlib.pyplot as plt

# def load_data(path)
# split data
# def extract_features_from_text(X_train,X_test,stop_words)

# Unforseen CVEs: will be used for score prediction
nvd_last_20_cves = pd.read_csv("data/nvd_last_20_cves.csv")

model_score_list = []
model_selected_features = {}

for i in range (0, 6):
    number_of_features=i*100
    if number_of_features == 0:
        lr_model,fs,y_pred,model_score = eval_model_for_features(score_func=f_regression,
                                                                 number_of_features='all')
        model_score['number_of_features'] = 'all'
        print("Number of features: All \n")
    else:
        lr_model,fs,y_pred,model_score = eval_model_for_features(score_func=f_regression,
                                                             number_of_features=number_of_features)
        model_score['number_of_features'] = number_of_features
        model_selected_features[str(number_of_features)]= fs.get_support(True)
        # print("Number of features: {}\n".format(number_of_features))
    
    # predict socres for unforeseen vulnerabilities
    for index, row in nvd_last_20_cves.iterrows():
        # print(row['cve_id'],row['summary'])
        model_score[row['cve_id']+'_predict']=round(lr_model.predict(fs.transform(
            vectorizer.transform([row['summary']])))[0],4)
        model_score[row['cve_id']+'_v2_actual']=row['cvss_score_20']
        model_score[row['cve_id']+'_v3_actual']=row['cvss_score_31']
        model_score[row['cve_id']+'_v2_diff']=model_score[row['cve_id']+'_predict']-row['cvss_score_20']
        model_score[row['cve_id']+'_v3_diff']=model_score[row['cve_id']+'_predict']-row['cvss_score_31']
    
    model_score_list.append(model_score)
    
    
    # print("Scatter Plot: Actual vs. Prediction")
    #sns.scatterplot(x=y_test,y=y_pred)
    # plt.scatter(y_test, y_pred, color='red')
    # plt.show()

model_score_df = pd.DataFrame.from_dict(model_score_list)
model_score_df

model_feature_list = {}
for key, values in model_selected_features.items():
    # print("{} features selected \n".format(key))
    feature_list = []
    for val in values:
        feature_list.append(feature_names[val])
    model_feature_list[key]=feature_list
    # print("{} features list length \n".format(len(feature_list)))
model_feature_df = pd.DataFrame.from_dict(model_feature_list,orient='index').transpose()
model_feature_df.to_csv("selected_features_f.csv",index=False)

model_score_df.to_csv("model_scores_f.csv",index=False)


v2_diff_cols=[col for col in model_score_df.columns if 'v2_diff' in col]
v2_diff_cols.append('number_of_features')
model_v2_diff=model_score_df[v2_diff_cols]
model_v2_diff.set_index('number_of_features',inplace=True)
#model_v2_diff
for index, row in model_v2_diff.iterrows():
    print("For {} features the difference between predicted and actual V2 scores\n{}\n{}\n".format(index,
                row.describe(),
                row.plot(kind="hist")))
    plt.show()


v3_diff_cols=[col for col in model_score_df.columns if 'v3_diff' in col]
v3_diff_cols.append('number_of_features')
model_v3_diff=model_score_df[v3_diff_cols]
model_v3_diff.set_index('number_of_features',inplace=True)
#model_v3_diff
for index, row in model_v3_diff.iterrows():
    print("For {} features the difference between predicted and actual V3 scores\n{}\n{}\n".format(index,
                row.describe(),
                row.plot(kind="hist")))
    plt.show()

model_pred=model_score_df[model_score_df['number_of_features']==300]
pred_cols=[col for col in model_score_df.columns if 'CVE' in col]
model_pred=model_pred[pred_cols]
diff_cols=[col for col in model_score_df.columns if 'diff' in col]
model_pred=model_pred.drop(diff_cols, axis=1)
model_pred=model_pred.T
model_pred.to_csv("model_pred.csv")
