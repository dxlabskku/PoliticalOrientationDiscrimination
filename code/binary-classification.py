import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from math import log
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from khaiii import KhaiiiApi
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

# 크롤링 자료 불러오기
data = pd.read_excel("daum_news_crawling(기존자료).xlsx", index_col = 0, engine = 'openpyxl')

# 보수 메이저 언론사로 p_data 구성
p_data = data[(data['source'] == '조선일보') | (data['source'] == '중앙일보') | (data['source'] == '동아일보')]
p_data = p_data.reset_index()
p_data

# 진보 메이저 언론사로 n_data 구성
n_data = data[(data['source'] == '한겨레') | (data['source'] == '경향신문')]
n_data = n_data.reset_index()
n_data
