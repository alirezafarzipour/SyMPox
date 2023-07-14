import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import gradio as gr
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
# from PIL import Image
print('Gradio version: ', gr.__version__)


# define dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Oversample with SMOTE and random undersample for imbalanced dataset
# define pipeline
over = SMOTE(sampling_strategy=0.5, random_state=27) #8
under = RandomUnderSampler(sampling_strategy=0.5, random_state=376) #1
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X, y = pipeline.fit_resample(X, y)

# split dataset to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=354)

# define model and make prediction on test set
classifier = XGBClassifier(learning_rate=0.0991, gamma=0, n_estimators = 80)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# confusion matrix & accuracy score
cm = confusion_matrix(y_test, y_pred)    
print('\n', cm)
print('\naccuracy:   ', accuracy_score(y_test, y_pred))


# creating a graphic environment
header = dataset.columns[1:-1]

def predict(symptoms):
    new_sample = [0]*len(header)
    for i in range(len(header)):
        for j in range(len(symptoms)):
            if header[i] == symptoms[j]:
                new_sample[i] = 1
    pred = classifier.predict(np.array(new_sample).reshape(1, len(header)))
    
    if pred == 1:
      return  'positive.png', '❌ Regrettably, there exists a possibility of Monkeypox.'
      
    else:
      return  'negative.png', '✔️ Fortunately, you have been spared from the presence of Monkeypox. It is recommended to consult with a medical professional for further confirmation.'
    
    
title='🐒🦠 Detection of Monkeypox Cases Based on Symptoms'
des='1. Kindly choose the symptoms you are experiencing from the provided section below and proceed by clicking the <b>Submit</b> button. The outcome will be displayed in the <u>output</u> section for your reference\n2. If you wish to reset the selected symptoms, simply click the <b>Clear</b> button.'

out_text = gr.inputs.Textbox(label="Result")
out_image = gr.inputs.Image(label=" ")
demo = gr.Interface(fn=predict, inputs=gr.CheckboxGroup(list(header)), outputs=[out_image, out_text], allow_flagging="never",
                   title=title, description=des)

if __name__ == "__main__":
    demo.launch()
