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


# 보수 메이저 언론사로 p_data 구성 / 진보 메이저 언론사로 n_data 구성
p_data = data[(data['source'] == '조선일보') | (data['source'] == '중앙일보') | (data['source'] == '동아일보')]
p_data = p_data.reset_index()
n_data = data[(data['source'] == '한겨레') | (data['source'] == '경향신문')]
n_data = n_data.reset_index()


# 'title' 열에서 '국방외교'라는 단어가 있으면 해당 행을 유지하고 '외교'라는 단어가 있으면 해당 행을 삭제
p_data = p_data[p_data['title'].str.contains('국방외교') | ~p_data['title'].str.contains('외교')]
p_data = p_data.reset_index()
n_data = n_data[p_data['title'].str.contains('국방외교') | ~n_data['title'].str.contains('외교')]
n_data = n_data.reset_index()


# 기사 or 내용이동일한 항목 통합
p_data_merge = p_data.drop_duplicates(subset = 'title').reset_index(drop = True)
p_data_merge = p_data.drop_duplicates(subset = 'contents').reset_index(drop = True)
n_data_merge = n_data.drop_duplicates(subset = 'title').reset_index(drop = True)
n_data_merge = n_data.drop_duplicates(subset = 'contents').reset_index(drop = True)


# 긱 언론사별 라벨 부여
p_data_final = p_data_merge.loc[:, ['source', 'contents']]
def assign_label(p_data_final):
    if p_data_final['source'] == '조선일보':
        return 0
    elif p_data_final['source'] == '중앙일보':
        return 1
    elif p_data_final['source'] == '동아일보':
        return 2
p_data_final['label'] = p_data_final.apply(assign_label, axis=1)
p_data_final['contents'] = p_data_final['contents'].fillna("")

n_data_final = n_data_merge.loc[:, ['source', 'contents']]
def assign_label(n_data_final):
    if n_data_final['source'] == '한겨레':
        return 3
    elif n_data_final['source'] == '경향신문':
        return 4
n_data_final['label'] = n_data_final.apply(assign_label, axis=1)
n_data_final['contents'] = n_data_final['contents'].fillna("")


# re 함수 활용 특수문자, 영어, 한자, 숫자, HTML 태그 등 제
def clean_text(texts):
    review = texts
    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\♣\▲\ⓒ\■\[\]\“\”\☞\‘\’\▶\·\…\〃\<\>"]', '', review) #remove punctuation
    review = re.sub(r'\d+','', review)# remove number
    review = re.sub(r'\s+', ' ', review) #remove extra space
    review = re.sub(r'<[^>]+>','',review) #remove Html tags
    review = re.sub(r'\s+', ' ', review) #remove spaces
    review = re.sub(r'^\s+', '', review) #remove space from start
    review = re.sub(r'\s+$', '', review) #remove space from the end
    review = re.sub(r'[A-Za-z]+', '', review)
    review = re.sub(r'[\u4e00-\u9fff]+', '', review)
    review = re.sub(r'\S+\s+기자', '', review)
    review = re.sub(r'\S+\s+선임기자', '', review)
    review = re.sub(r'\S+\s+군사전문기자', '', review)
    review = re.sub(r'\S+기자', '', review)
    review = re.sub(r'\S+선임기자', '', review)
    review = re.sub(r'\S+군사전문기자', '', review)
    return review
  
p_data_final['clean_contents'] = p_data_final['contents'].apply(lambda x : clean_text(x))
n_data_final['clean_contents'] = n_data_final['contents'].apply(lambda x : clean_text(x))


# 추가 언론사 특정 가능 문구 제거
remove_source = ['조선일보', '중앙일보', '동아일보', '한겨레', '경향신문', '오마이뉴스', '동아닷컴']
remove_add = ['무단 전재 및 재배포 금지', '기사', '어제 못본', '명장면이 궁금하다면', '오늘의', '김상호', '절친이 되어 주세요', 
              '신문구독주주신청', '페이스북카카오톡사설칼럼신문', '무단전재 및 재배포 금지']

pattern1 = '|'.join(remove_source)
p_data_final['clean_contents'] = p_data_final['clean_contents'].str.replace(pattern1, '', regex = True)
n_data_final['clean_contents'] = n_data_final['clean_contents'].str.replace(pattern1, '', regex = True)

pattern2 = '|'.join(remove_add)
p_data_final['clean_contents'] = p_data_final['clean_contents'].str.replace(pattern2, '', regex = True)
n_data_final['clean_contents'] = n_data_final['clean_contents'].str.replace(pattern2, '', regex = True)


# Khaiii 활용 형태소 분석
api = KhaiiiApi()
tqdm.pandas()

def analyze_morphs(text):
    if text.strip() == '':
        return []
    analyzed = api.analyze(text)
    morphs = []
    for word in analyzed:
        for morph in word.morphs:
            morphs.append((morph.lex, morph.tag))
    return morphs

p_data_final['morphs_contents'] = p_data_final['clean_contents'].progress_apply(analyze_morphs)
n_data_final['morphs_contents'] = n_data_final['clean_contents'].progress_apply(analyze_morphs)


# 일반명사(NNG), 고유명사(NNP), 의존명사(NNB), 동사(VV), 형용사(VA) 만 사용
p_data_final['filtered_contents'] = p_data_final['morphs_contents'].apply(lambda x: [(word, pos) for word, pos in x if pos in ['NNG', 'NNP', 'NNB', 'VV', 'VA']])
n_data_final['filtered_contents'] = n_data_final['morphs_contents'].apply(lambda x: [(word, pos) for word, pos in x if pos in ['NNG', 'NNP', 'NNB', 'VV', 'VA']])

p_data_final['filtered_contents'] = p_data_final['filtered_contents'].apply(lambda x: [word for word, _ in x])
n_data_final['filtered_contents'] = n_data_final['filtered_contents'].apply(lambda x: [word for word, _ in x])


# 전처리된 토큰 문장화
p_data_final['filtered_contents_str'] = p_data_final['filtered_contents'].apply(lambda x: ' '.join(x))
n_data_final['filtered_contents_str'] = n_data_final['filtered_contents'].apply(lambda x: ' '.join(x))


# 보수 언론사 데이터 / 진보언론사 데이터 통합 
sum_data_final = pd.concat([p_data_final, n_data_final])


# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
x = tfidf_vectorizer.fit_transform(sum_data_final['filtered_contents_str'])
y = sum_data_final['label']


# train-test set 구분
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1)

# Under-sampling 적용(미적용시 주석처리)
under_sampler = RandomUnderSampler()
x_train, y_train = under_sampler.fit_resample(x_train, y_train)

# SMOTE 적용(미적용시 주석처리)
smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)


# LR, RF, XGB, 앙상블 모델 적용 결과 산출
# Logistic Regression 모델
lr_model = LogisticRegression(C = 10.0, max_iter = 1000)

# Random Forest 모델
rf_model = RandomForestClassifier(max_depth = None, n_estimators = 200)

# XGBoost 모델
xgb_model = XGBClassifier(use_label_encoder = False, eval_metric = 'error', learning_rate = 0.2, max_depth = 7, n_estimators = 200)



# 예측 결과의 가중 평균 구하기
s_voting = VotingClassifier(estimators=[('lr', lr_model), ('rf', rf_model), ('XGB', xgb_model)], voting='soft', weights = [10, 1, 10], flatten_transform = True)
s_voting_fit = s_voting.fit(x_train, y_train)
t_pred = s_voting_fit.predict(x_test)
    
print("\n=================== 앙상블 ==================")
print(classification_report(y_test, t_pred, digits=3))


# K-Fold crossvalidation(Under-sampling)
kf = KFold(n_splits = 5)
accuracies = []

for train_index, val_index in kf.split(x_train):
    X_train_fold, X_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    under_sampler = RandomUnderSampler()
    X_train_fold_resampled, y_train_fold_resampled = under_sampler.fit_resample(X_train_fold, y_train_fold)

    s_voting_fit = s_voting.fit(X_train_fold_resampled, y_train_fold_resampled)

    y_val_pred = s_voting_fit.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, y_val_pred)
    accuracies.append(accuracy)

average_accuracy = np.mean(accuracies)
print(f"Average accuracy: {average_accuracy:.3f}")


# K-Fold crossvalidation(SMOTE)
kf = KFold(n_splits = 5)
accuracies = []

for train_index, val_index in kf.split(x_train):
    X_train_fold, X_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    smote = SMOTE()
    X_train_fold_resampled, y_train_fold_resampled = smote.fit_resample(X_train_fold, y_train_fold)

    s_voting_fit = s_voting.fit(X_train_fold_resampled, y_train_fold_resampled)

    y_val_pred = s_voting_fit.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, y_val_pred)
    accuracies.append(accuracy)

average_accuracy = np.mean(accuracies)
print(f"Average accuracy: {average_accuracy:.3f}")
