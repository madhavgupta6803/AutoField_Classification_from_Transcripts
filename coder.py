import pandas as pd
import numpy as np
import os
import glob
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer()

from sklearn.preprocessing import LabelEncoder
# Used for encoding the data in the field and Trascript column
label_encoder_y_train = LabelEncoder()
label_encoder_x_train = LabelEncoder()
#y_train_encoded = label_encoder.fit_transform(y_train_labels)
datab = [] # Used to store list of dataframes used for training
# datab_val = []
# datab_val_w_ann = []
def concatenate_elements_if_gt_one(input_list): # No longer useful in this case
    l = []*1
    # Check if the number of elements in the list is greater than 1
    if len(input_list) > 1:
    # Concatenate the elements to form a single entity
        concatenated_entity = " ".join(input_list)
    #print(concatenated_entity)
        l.append(concatenated_entity) 
        return l
    elif len(input_list) == 1:
        # Return the element as is
        return input_list
    else:
        l = ['2019']
        #l.append[2019]
        return l



# Changing the column names form the original values to these to maintain uniformity in training set
new_column_names = [
    "start_index", "end_index", "x_top_left", "y_top_left",
    "x_bottom_right", "y_bottom_right", "transcript", "field"
]

# Changing the column names form the original values to these to maintain uniformity in testing set
new_column_names_val = [
    "start_index", "end_index", "x_top_left", "y_top_left",
    "x_bottom_right", "y_bottom_right", "transcript"
]
# def rename_columns_in_csv(new_column_names):
#     df.columns = new_column_names

# Specify the folder path containing the CSV files
folder_path = 'dataset/dataset/train/boxes_transcripts_labels'

# Use glob to get a list of CSV file paths in the folder
csv_files = glob.glob(folder_path + '/*.tsv')
for csv_file in csv_files:
    dff = pd.read_csv(csv_file, sep='\t')
    #dff.columns = new_column_names
    # for column in dff.columns:
    #     print(column)
    if(dff.shape[1]<9): # Checking and cleaning the dataframes if columns are merged then remove that dataframe 
        del dff
        continue
    else:
        # One column is made in dff when importing from csv so removing that column
        dff = dff.drop(columns=['Unnamed: 0'])
        dff.columns = new_column_names
        datab.append(dff)
        #print(datab)
    #print(dff.head())
# Merging all the dataframes to form a new dataframe for training
combined_dff = pd.concat(datab, ignore_index = True)

folder_path_val = 'dataset/dataset/val_original/boxes_transcripts'

# Code to convert testing files but they are loaded in loop not together
"""# Use glob to get a list of CSV file paths in the folder
csv_files = glob.glob(folder_path_val + '/*.tsv')
for csv_file in csv_files:
    # df_val= pd.read_csv(csv_file)
    # df_val.to_csv(csv_file, sep='\t')
    dff_val = pd.read_csv(csv_file, sep='\t')
    #dff.columns = new_column_names
    # for column in dff.columns:
    #     print(column)
    if(dff_val.shape[1]<8):
        print(csv_file)
        del dff_val
        continue
    else:
        #print('okk')
        dff_val = dff_val.drop(columns=['Unnamed: 0'])
        dff_val.columns = new_column_names_val
        datab_val.append(dff_val)
    #print(dff.head())
combined_dff_val = pd.concat(datab_val, ignore_index = True)"""

folder_path_val_w_nn = 'dataset/dataset/val_w_ann/boxes_transcripts_labels'

# # Use glob to get a list of CSV file paths in the folder
"""csv_files = glob.glob(folder_path_val_w_nn + '/*.tsv')
for csv_file in csv_files:
    df_val_w_ann= pd.read_csv(csv_file)
    df_val_w_ann.to_csv(csv_file, sep='\t')
    dff_val_w_ann = pd.read_csv(csv_file, sep='\t')
    #dff.columns = new_column_names
    # for column in dff.columns:
    #     print(column)
    if(dff_val_w_ann.shape[1]<9):
        del dff_val_w_ann
        continue
    else:
        dff_val_w_ann = dff_val_w_ann.drop(columns=['Unnamed: 0'])
        dff_val_w_ann.columns = new_column_names
        dff_val_w_ann.to_csv(csv_file, sep='\t')
        datab_val_w_ann.append(dff_val_w_ann)
    #print(dff.head())
combined_dff_val_w_ann = pd.concat(datab_val_w_ann, ignore_index = True)
print(combined_dff)"""
    # for index, row in dff.iterrows():
    #     datab.append(dff[[row[0], row[1], row[2], row[3], row[4], row[5]]])
    #     datain.append(dff(row[6])) # Input is all the remaining quantities
    #     dataout.append(dff[row[7]]) # Output is the field column
        #X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
        # field = row[7].lower()
        # transcript = row[6]
# Using RandomForest for classification into 15 classes available
from sklearn.ensemble import RandomForestClassifier
y_train_labels = combined_dff['field'] # We want the model to predict the field 
print((y_train_labels)) 
y_train_encoded = label_encoder_y_train.fit_transform(y_train_labels) # Converting them to labels to give to the classifier
print(label_encoder_y_train.classes_)

X_train_trans = combined_dff['transcript'] # Only this column needs to be encoded as it has all types of values from strings to numbers to special characters
X_train_rem = combined_dff.drop(columns = ['field', 'transcript']).to_numpy() # Rest of the input_X is numerical

#print(X_train_rem)
#print(X_train_trans)
X_train_trans_encoded = label_encoder_x_train.fit_transform(X_train_trans).reshape(-1, 1)# Reshaping so that it can be combined with rest of X_input
#X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_train_combined = np.hstack((X_train_rem, X_train_trans_encoded))# Final training X_input 

#print(X_train_combined)
output_folder = 'dataset/dataset/output/'   # Replace with your desired output folder path

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

csv_files = glob.glob(folder_path_val + '/*.tsv')
for csv_file in csv_files:
    #df_val= pd.read_csv(csv_file)
    #df_val.to_csv(csv_file, sep='\t')
    dff_val = pd.read_csv(csv_file, sep='\t')
    #dff.columns = new_column_names
    # for column in dff.columns:
    #     print(column)
    if(dff_val.shape[1]<8): # Converting each .tsv val file into dataframe and then performing the similar things to give X_test_Input to the model
        print(csv_file)
        del dff_val
        continue
    else:
        #print('okk')
        dff_val = dff_val.drop(columns=['Unnamed: 0'])
        dff_val.columns = new_column_names_val
    X_test_trans = dff_val['transcript']
    X_test_rem = dff_val.drop(columns = ['transcript']).to_numpy()
    label_encoder_x_test = LabelEncoder()# Initializing the labelEncoder to be used in next line
    X_test_trans_encoded = label_encoder_x_test.fit_transform(X_test_trans).reshape(-1, 1)
    X_test = np.hstack((X_test_rem, X_test_trans_encoded))
    model = RandomForestClassifier(n_estimators=20) 
    model.fit(X_train_combined, y_train_encoded) # Applying the RF Classifier with 20 estimators
    y_test_pred = model.predict(X_test) # Predicting output which will be encoded
    y_test_pred_decoded = label_encoder_y_train.inverse_transform(y_test_pred) # Converting the encoded data to original labels
    dff_val['field'] = y_test_pred_decoded # Making a new column and adding all the predicted values there
    #print(dff_val)
    output_file = os.path.join(output_folder, os.path.basename(csv_file))
    dff_val.to_csv(output_file, sep='\t', index=False)
    dff_val.to_csv(csv_file, sep='\t', index=False) # Saving the output_file




"""count =0 
# for element in y_test_pred:
#     if(element >0):
#         count+=1
# print(count)
y_test_pred_decoded = label_encoder_y_train.inverse_transform(y_test_pred)
# for element in y_test_pred_decoded:
#     if(element != '!'):
#         print(element)
print(y_test_pred_decoded)"""


            








