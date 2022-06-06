# from turtle import clear
# from selenium import webdriver

# DRIVER_PATH = '/Users/mdjavedulferdous/Desktop/chromedriver'
# driver = webdriver.Chrome(executable_path=DRIVER_PATH)
# driver.get('https://www.amazon.com/s?k=apple+watch&i=electronics&crid=3MR9PVJ318602&sprefix=%2Celectronics%2C328&ref=nb_sb_ss_recent_1_0_recent')
# print(driver.page_source)
# driver.save_screenshot('screenshot.png')
import numpy as np
from sklearn import model_selection
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score,average_precision_score,confusion_matrix, recall_score, accuracy_score, classification_report, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import pandas as pd
def evaluation_process():
    path = '/Users/mdjavedulferdous/Documents/Dataset/New Dataset/pageList_version_six.csv'
    csv_df = pd.read_csv(path)
    X = csv_df[['NumOfButton','NumOfLinks','commonURL','NumberOfValues','is_page']]
    y = csv_df[['pageClass']]
    filtered_one = csv_df[csv_df['pageClass'] != 0] 
    filtered_zero = csv_df[csv_df['pageClass'] == 0] 
    f_o = filtered_zero.sample(n=209,replace=False)
    z = pd.concat([filtered_one, f_o], axis=0)
    x_2 = z[['NumOfButton','NumOfLinks','commonURL','NumberOfValues','is_page']]
    y_2 = z[["pageClass"]]

    return x_2, y_2
    
def train_classifier(X,y):
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    p_class_0, p_class_1, r_class_0, r_class_1 = [], [], [], []
    
    for train_index, test_index in KFold().split(X):
        #print('Iteration:',iteration)
        #print("No of Training Instance: ",(NoY - len(test_index)))
        #print("No of Testing Instance: ",(NoY - len(train_index)))
        #print("Train Index: ", train_index)
        #print("Test Index: ", test_index)
        
        model = clf
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        # save the model to disk
        filename = '/Users/mdjavedulferdous/Desktop/TiiS/Code/finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        
        # load the model from disk
        # loaded_model = pickle.load(open(filename, 'rb'))
        # result = loaded_model.score(X_test, y_test)
        # print(loaded_model.predict(X_test))

        y_score = model.predict(X_test)
        cv_results = model_selection.cross_val_score(model, X_train, y_train).mean()
        print(precision_score(y_test, y_score, average=None))
        # precision = precision_score(y_test, y_score, average=None)
        # p_class_0.append(precision[0])
        # p_class_1.append(precision[1])
        
        # recall = recall_score(y_test, y_score, average=None)
        # #print(recall)
        # r_class_0.append(recall[0])
        # r_class_1.append(recall[1])
        target_names = ['Class 0', 'Class 1']
        print(classification_report(y_test, y_score, target_names=target_names))

    return p_class_0,p_class_1, r_class_0, r_class_1, cv_results


def main():
    
    x_2, y_2 = evaluation_process()
    p_class_0,p_class_1, r_class_0, r_class_1, acc = train_classifier(x_2,y_2)
    # print("========================================================")
    # print('Average class 0 precision = {:.1f}'.format((sum(p_class_0)/len(p_class_0))*100))
    # print('Average class 1 precision = {:.1f}'.format((sum(p_class_1)/len(p_class_1))*100))
    # print('Average class 0 recall = {:.1f}'.format((sum(r_class_0)/len(r_class_0))*100))
    # print('Average class 1 recall = {:.1f}'.format((sum(r_class_1)/len(r_class_1))*100))
    # print('Average accuracy = {:.1f}'.format(acc*100))

    # print("========================================================")

if __name__ == "__main__":
    main()