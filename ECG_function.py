# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:33:12 2020

@author: NutchapolD
"""
import numpy as np
from keras.utils import to_categorical

def confusion_dash(cm, kind):
    return {
                "data": [
            {
              "type": "heatmap",
              "x": [
                "NHR",
                "CHF",
                "ATH"
              ],
              "y": [
                "ATH",
                "CHF",
                "NHR"
              ],
              "z": [
                [
                  cm[2,0],
                  cm[2,1],
                  cm[2,2]
                ],
                [
                  cm[1,0],
                  cm[1,1],
                  cm[1,2]
                ],
                [
                  cm[0,0],
                  cm[0,1],
                  cm[0,2]
                ]
              ]
            }
          ],
          "layout": {
            "title": "Confusion Matrix of "+kind,
            "height": 500,
            "width" : 500,
            "xaxis": {
              "title": "Predicted Class"
            },
            "yaxis": {
              "title": "True Class"
            },
            "annotations": [
              {
                "x": "NHR",
                "y": "NHR",
                "font": {
                  "color": "white"
                },
                "text": str(cm[0,0]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "CHF",
                "y": "NHR",
                "font": {
                  "color": "white"
                },
                "text": str(cm[0,1]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "ATH",
                "y": "NHR",
                "font": {
                  "color": "white"
                },
                "text": str(cm[0,2]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "NHR",
                "y": "CHF",
                "font": {
                  "color": "white"
                },
                "text": str(cm[1,0]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "CHF",
                "y": "CHF",
                "font": {
                  "color": "white"
                },
                "text": str(cm[1,1]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "ATH",
                "y": "CHF",
                "font": {
                  "color": "white"
                },
                "text": str(cm[1,2]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "NHR",
                "y": "ATH",
                "font": {
                  "color": "white"
                },
                "text": str(cm[2,0]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "CHF",
                "y": "ATH",
                "font": {
                  "color": "white"
                },
                "text": str(cm[2,1]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              },
              {
                "x": "ATH",
                "y": "ATH",
                "font": {
                  "color": "white"
                },
                "text": str(cm[2,2]),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
              }
            ]
          },
          "frames": [],
            }