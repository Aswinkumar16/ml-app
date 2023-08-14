import pandas as pd
import base64
import os
import io
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib
import shap

import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots

import mpld3 as mpl
import seaborn as sns

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, f1_score

