# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:15:09 2020

@author: NutchapolD
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from ECG_function import *

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__)
app = dash.Dash()
server = app.server

# 1D CNN
X_test=np.load('data/ECG_3_X_test.npy')
y_test=np.load('data/ECG_3_y_test.npy')
idx0=np.where(np.argmax(y_test,axis=1)==0)[0]
idx1=np.where(np.argmax(y_test,axis=1)==1)[0]
idx2=np.where(np.argmax(y_test,axis=1)==2)[0]
'''
best_model = tf.keras.models.load_model('best_model/1D_CNN_model_ECG_3')
l=y_test.shape[0]
dict_size=10
cm_dict={}
for i in range(dict_size):
    chosen=np.random.choice(l,int(l/2))
    chosen_X_test=X_test[chosen,:,:]
    chosen_y_test=y_test[chosen,:]
    chosen_y_pred=best_model.predict(chosen_X_test)
    cm = confusion_matrix(np.argmax(chosen_y_test,axis=1), np.argmax(chosen_y_pred,axis=1))
    cm_dict[i]=cm
'''
# wavelet CNN 
X_test_w=np.load('data/X_test_ECG_3_cwt_mexh.npy')
y_test_w=np.load('data/y_test_ECG_3_cwt_mexh.npy')
best_model_w = tf.keras.models.load_model('wavelet_CNN_model_ECG_3')
l=y_test_w.shape[0]
dict_size_w=1
cm_dict_w={}
for i in range(dict_size_w):
    chosen=np.random.choice(l,int(l/2))
    chosen_X_test=X_test_w[chosen,:,:]
    chosen_y_test=y_test_w[chosen,:]
    chosen_y_pred=best_model_w.predict(chosen_X_test)
    cm = confusion_matrix(np.argmax(chosen_y_test,axis=1), np.argmax(chosen_y_pred,axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm=np.around(cm,3)
    cm_dict_w[i]=cm




#cm=np.array([[254,   2,   3],
#       [  0, 248,  25],
#       [  0,  27, 240]])
app.layout = html.Div([
    html.H1(children='ECG Prediction',
            style={'textAlign': 'center'}
        ),
    html.Div(children='This dashboard demonstrates plot examples of three types of ECG. Normal hearth rate (NHR) of healthy patient, congestive heart failure (CHF), and arrhythmia (ATH). The overall performance of the algorithm is shown below in form of a onfusion matrix'),
    html.Button('plot another random sample', id='plot-data-1', n_clicks=0),  
    html.Div(children='ECG response of a healthy person',
            style={'textAlign': 'center'}
        ),
    dcc.Graph(id='ECG-1'),
    html.Button('plot another random sample', id='plot-data-2', n_clicks=0),  
    html.Div(children='ECG response of a patient with congestive heart failure',
            style={'textAlign': 'center'}
        ),
    dcc.Graph(id='ECG-2'),
    html.Button('plot another random sample', id='plot-data-3', n_clicks=0),  
    html.Div(children='ECG response of a patient with arrhythmia',
            style={'textAlign': 'center'}
        ),
    dcc.Graph(id='ECG-3'),
    #html.Button('run 1D CNN model', id='run-model', n_clicks=0),  
    #dcc.Graph(id='confusion-matrix'),
    html.Button('run wavelet CNN model', id='run-model-w', n_clicks=0),  
    dcc.Graph(id='confusion-matrix-w'),
    
])
        
@app.callback(
    dash.dependencies.Output('ECG-1', 'figure'),
    [dash.dependencies.Input('plot-data-1', 'n_clicks')])
def update_output_1(n_clicks):
    #np.random.seed(n_clicks)
    chosen0=np.random.choice(len(idx0),1)[0]
    n=1000
    df=pd.DataFrame()
    df['x']=np.arange(n)
    df['y']=X_test[idx0[chosen0],:n,:]
    return { 
            'data': [
                dict(
                    x=df['x'],
                    #x=np.arange(n).reshape(1,-1)[0],
                    y=df['y'],
                    name='ecg'
                ) 
            ],
            
            'layout': dict(
                test='NSR',
                xaxis={'title': 'time'},
                yaxis={'title': 'ECG response'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
                height = 200,
            )
    }
    
@app.callback(
    dash.dependencies.Output('ECG-2', 'figure'),
    [dash.dependencies.Input('plot-data-2', 'n_clicks')])
def update_output_2(n_clicks):
    #np.random.seed(n_clicks)
    chosen1=np.random.choice(len(idx1),1)[0]
    n=1000
    df=pd.DataFrame()
    df['x']=np.arange(n)
    df['y']=X_test[idx1[chosen1],:n,:]
    return { 
            'data': [
                dict(
                    x=df['x'],
                    #x=np.arange(n).reshape(1,-1)[0],
                    y=df['y'],
                    name='ecg'
                ) 
            ],
            
            'layout': dict(
                test='NSR',
                xaxis={'title': 'time'},
                yaxis={'title': 'ECG response'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
                height = 200,
            )
    }
    
@app.callback(
    dash.dependencies.Output('ECG-3', 'figure'),
    [dash.dependencies.Input('plot-data-3', 'n_clicks')])
def update_output_3(n_clicks):
    #np.random.seed(n_clicks)
    chosen2=np.random.choice(len(idx2),1)[0]
    n=1000
    df=pd.DataFrame()
    df['x']=np.arange(n)
    df['y']=X_test[idx2[chosen2],:n,:]
    return { 
            'data': [
                dict(
                    x=df['x'],
                    #x=np.arange(n).reshape(1,-1)[0],
                    y=df['y'],
                    name='ecg'
                ) 
            ],
            
            'layout': dict(
                test='NSR',
                xaxis={'title': 'time'},
                yaxis={'title': 'ECG response'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
                height = 200,
            )
    }

'''    
@app.callback(
    dash.dependencies.Output('confusion-matrix', 'figure'),
    [dash.dependencies.Input('run-model', 'n_clicks')])
def update_output_confusion_matrix(n_clicks):
    cm_idx=np.random.choice(dict_size,1)[0]
    cm=cm_dict[cm_idx]
    return confusion_dash(cm, "1D CNN")
'''

@app.callback(
    dash.dependencies.Output('confusion-matrix-w', 'figure'),
    [dash.dependencies.Input('run-model-w', 'n_clicks')])
def update_output_confusion_matrix(n_clicks):
    cm_idx=np.random.choice(dict_size_w,1)[0]
    cm=cm_dict_w[cm_idx]
    return confusion_dash(cm, "Wavelet CNN")

#if __name__ == '__main__':
#    app.run_server(debug=True)

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
