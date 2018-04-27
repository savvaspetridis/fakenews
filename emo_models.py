# written by Savvas

import pandas as pd
import numpy as np
#from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_score, recall_score, f1_score

import nltk
import string
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

vec = CountVectorizer()

df = pd.read_csv("liar_dataset/train.tsv", delimiter="\t", header= None)
# print 'len(DF): ' + str(df)
lbw = []
corpus =[]
s2 = set()

# liwc = np.array(ml)
train_file = open('liar_dataset/train.tsv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n')
for line in train_file:
	line = line.split('\t');
	corpus.append(line[2]);
	lbw.append(line[1]);
	s2.add(line[1]);

'''
test_file = pd.read_csv("liar_dataset/test.tsv", delimiter="\t", header= None)
lbw2 = []
test_corpus = []
for i in range(0,len(test_file)):
	test_corpus.append(test_file[2].iloc[i]);
	lbw2.append(test_file[1].iloc[i]);
'''
lbw2 = []
valid_corpus = []
valid_file = open('liar_dataset/valid.tsv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n')
for line in valid_file:
	line = line.split('\t');
	valid_corpus.append(line[2]);
	lbw2.append(line[1]);

lb =dict()
for i, w in enumerate(sorted(s2)):
	lb[w] = i

train_six_labels =[]
for i in lbw:
    train_six_labels.append(lb[i])

valid_six_labels = [];
for i in lbw2:
	valid_six_labels.append(lb[i]);

train_bin_labels_file = open('train_binary_labels','r');
train_bin_labels_file = train_bin_labels_file.read();
train_binary_labels = train_bin_labels_file.split('\n');

valid_bin_labels_file = open('valid_binary_labels','r');
valid_bin_labels_file = valid_bin_labels_file.read();
valid_binary_labels = valid_bin_labels_file.split('\n');

# ============= TRAINING ==========================================
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ==============
# METADATA 
# ==============

train_metadata = [];
train_meta = open('metadata_train.csv');
train_meta = train_meta.read();
train_meta = train_meta.split('\n');
for line in train_meta:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		train_metadata.append(line);

# ==============
# BINARY FILES
# ==============

# EMOTION
train_emotion_bin = []; 
train_file = open('train_files/emo_train_bin.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n')
for line in train_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		train_emotion_bin.append(line);

# SENTISTRENGTH
train_senti_bin = []; 
train_file = open('train_files/sent_train_bin.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n')
for line in train_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		train_senti_bin.append(line);

# MANNER ADVERBS
train_manner_adverbs_bin = [];
train_file = open('train_files/manner_adverbs_train_bin.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n');
for line in train_file:
	if line != '':
		# line = line.split(',');
		line = map(int,line);
		train_manner_adverbs_bin.append(line);

# SUPERLATIVES
train_superlatives_bin = [];
train_file = open('train_files/superlatives_train_bin.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n');
for line in train_file:
	if line != '':
		line = map(int,line);
		train_superlatives_bin.append(line);

# ==============
# COUNT FILES
# ==============

train_emotion_cnt = []; 
train_file = open('train_files/emo_train_cnt.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n')
for line in train_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		train_emotion_cnt.append(line);

# SENTISTRENGTH
train_senti_cnt = []; 
train_file = open('train_files/train_sent.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n')
for line in train_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		train_senti_cnt.append(line);

# MANNER ADVERBS
train_manner_adverbs_cnt = [];
train_file = open('train_files/manner_adverbs_train_cnt.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n');
for line in train_file:
	if line != '':
		# line = line.split(',');
		line = map(int,line);
		train_manner_adverbs_cnt.append(line);

# SUPERLATIVES
train_superlatives_cnt = [];
train_file = open('train_files/superlatives_train_cnt.csv', 'r');
train_file = train_file.read();
train_file = train_file.split('\n');
for line in train_file:
	if line != '':
		line = map(int,line);
		train_superlatives_cnt.append(line);

# ============= VALIDATION ==========================================
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ==============
# METADATA 
# ==============

valid_metadata = [];
valid_meta = open('metadata_valid.csv');
valid_meta = valid_meta.read();
valid_meta = valid_meta.split('\n');
for line in valid_meta:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		valid_metadata.append(line);

# ==============
# BINARY FILES
# ==============

# EMOTION
valid_emotion_bin = []; 
valid_file = open('valid_files/emo_valid_bin.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n')
for line in valid_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		valid_emotion_bin.append(line);

# SENTISTRENGTH
valid_senti_bin = []; 
valid_file = open('valid_files/sent_valid_bin.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n')
for line in valid_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		valid_senti_bin.append(line);

# MANNER ADVERBS
valid_manner_adverbs_bin = [];
valid_file = open('valid_files/manner_adverbs_valid_bin.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n');
for line in valid_file:
	if line != '':
		line = map(int,line);
		valid_manner_adverbs_bin.append(line);

# SUPERLATIVES
valid_superlatives_bin = [];
valid_file = open('valid_files/superlative_valid_bin.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n');
for line in valid_file:
	if line != '':
		line = map(int,line);
		valid_superlatives_bin.append(line);

# ==============
# COUNT FILES
# ==============

valid_emotion_cnt = []; 
valid_file = open('valid_files/emo_valid_cnt.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n')
for line in valid_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		valid_emotion_cnt.append(line);

# SENTISTRENGTH
valid_senti_cnt = []; 
valid_file = open('valid_files/sent_valid_cnt.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n')
for line in valid_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		valid_senti_cnt.append(line);

# MANNER ADVERBS
valid_manner_adverbs_cnt = [];
valid_file = open('valid_files/manner_adverbs_valid_cnt.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n');
for line in valid_file:
	if line != '':
		line = line.split(',');
		line = map(int,line);
		valid_manner_adverbs_cnt.append(line);

# SUPERLATIVES
valid_superlatives_cnt = [];
valid_file = open('valid_files/superlatives_valid_cnt.csv', 'r');
valid_file = valid_file.read();
valid_file = valid_file.split('\n');
for line in valid_file:
	if line != '':
		line = map(int,line);
		valid_superlatives_cnt.append(line);


# ===============================================================
#
#				EXPERIMENTS
#
# ===============================================================

# train_emotion_bin
# train_senti_bin
# train_manner_adverbs_bin
# train_superlatives_bin
# train_emotion_cnt
# train_senti_cnt
# train_manner_adverbs_cnt
# train_superlatives_cnt
# valid_emotion_bin
# valid_senti_bin
# valid_manner_adverbs_bin
# valid_superlatives_bin
# valid_emotion_cnt
# valid_senti_cnt
# valid_manner_adverbs_cnt
# valid_superlatives_cnt

# ========== TRAINING DATA ==============
x_train_unigrams = vec.fit_transform(corpus).toarray();
x_train_unigrams_metadata = np.hstack([train_metadata,x_train_unigrams]);
# Binary
x_train_unigrams_emotion_bin = np.hstack([train_emotion_bin,x_train_unigrams]);
x_train_unigrams_senti_bin = np.hstack([train_senti_bin,x_train_unigrams]);
x_train_unigrams_superlatives_manner_bin = np.hstack([train_manner_adverbs_bin,train_superlatives_bin,x_train_unigrams]);
# Count
x_train_unigrams_emotion_cnt = np.hstack([train_emotion_cnt,x_train_unigrams]);
x_train_unigrams_senti_cnt = np.hstack([train_senti_cnt,x_train_unigrams]);
x_train_unigrams_superlatives_manner_cnt = np.hstack([train_manner_adverbs_cnt,train_superlatives_cnt,x_train_unigrams]);
# All
x_train_all_bin = np.hstack([train_manner_adverbs_bin,train_superlatives_bin,train_senti_bin,train_emotion_bin,train_metadata,x_train_unigrams]);
x_train_all_cnt = np.hstack([train_manner_adverbs_cnt,train_superlatives_cnt,train_senti_cnt,train_emotion_cnt,train_metadata,x_train_unigrams]);

# ========== VALIDATION DATA ==============
x_valid_unigrams = vec.transform(valid_corpus).toarray();
x_valid_unigrams_metadata = np.hstack([valid_metadata,x_valid_unigrams]);
# Binary
x_valid_unigrams_emotion_bin = np.hstack([valid_emotion_bin,x_valid_unigrams]);
x_valid_unigrams_senti_bin = np.hstack([valid_senti_bin,x_valid_unigrams]);
x_valid_unigrams_superlatives_manner_bin = np.hstack([valid_manner_adverbs_bin,valid_superlatives_bin,x_valid_unigrams]);
# Count
x_valid_unigrams_emotion_cnt = np.hstack([valid_emotion_cnt,x_valid_unigrams]);
x_valid_unigrams_senti_cnt = np.hstack([valid_senti_cnt,x_valid_unigrams]);
x_valid_unigrams_superlatives_manner_cnt = np.hstack([valid_manner_adverbs_cnt,valid_superlatives_cnt,x_valid_unigrams]);
# All 
x_valid_all_bin = np.hstack([valid_manner_adverbs_bin,valid_superlatives_bin,valid_senti_bin,valid_emotion_bin,valid_metadata,x_valid_unigrams]);
x_valid_all_cnt = np.hstack([valid_manner_adverbs_cnt,valid_superlatives_cnt,valid_senti_cnt,valid_emotion_cnt,valid_metadata,x_valid_unigrams]);


training_data_sets = [x_train_unigrams_metadata, x_train_unigrams_emotion_bin, x_train_unigrams_senti_bin, x_train_unigrams_superlatives_manner_bin, x_train_unigrams_emotion_cnt, x_train_unigrams_senti_cnt, x_train_unigrams_superlatives_manner_cnt, x_train_all_bin, x_train_all_cnt];
names = ['uni + metadata','uni + emotion_bin', 'uni + senti_bin', 'uni + superlatives_bin + manner_bin', 'uni + emotion_cnt', 'uni + senti_cnt', 'uni + superlatives_cnt + manner_cnt', 'uni + all_bin', 'uni + all_cnt'];
validation_data_sets = [x_valid_unigrams_metadata, x_valid_unigrams_emotion_bin, x_valid_unigrams_senti_bin, x_valid_unigrams_superlatives_manner_bin, x_valid_unigrams_emotion_cnt, x_valid_unigrams_senti_cnt, x_valid_unigrams_superlatives_manner_cnt, x_valid_all_bin, x_valid_all_cnt];

y_train = train_six_labels; 
y_valid = valid_six_labels;
logreg = LogisticRegression()

'''
logreg_file = open('logreg_results','w'); 

print '';
print '================ LOGISTIC REGRESSION MODELS ================'
print '================== Six-way Classification =================='
print ''

i = 0;
while i < len(training_data_sets):
	x_train = training_data_sets[i];
	x_valid = validation_data_sets[i];
	logreg.fit(x_train, y_train);
	y_valid_predict = logreg.predict(x_valid);
	print names[i] + ':'
	#print classification_report(y_valid,y_valid_predict);
	# score = logreg.score(x_valid,y_valid)
	# print(names[i] + ': ' + str(score));
	logreg_file.write(names[i] + ': \n');
	logreg_file.write(str(classification_report(y_valid,y_valid_predict)));
	i = i + 1;

print ''
print '================== Binary Classification =================='
print ''

y_train = train_binary_labels; 
y_valid = valid_binary_labels;

i = 0;
while i < len(training_data_sets):
	x_train = training_data_sets[i];
	x_valid = validation_data_sets[i];
	logreg.fit(x_train, y_train);
	y_valid_predict = logreg.predict(x_valid);
	print names[i] + ':'
	# print classification_report(y_valid,y_valid_predict);
	logreg_file.write(names[i] + ': \n');
	logreg_file.write(str(classification_report(y_valid,y_valid_predict)));
	# print(names[i] + ': ' + str(logreg.score(x_valid,y_valid)));
	i = i + 1;
'''


# training_data_sets = [x_train_unigrams_metadata, x_train_unigrams_superlatives_manner_bin, x_train_unigrams_senti_cnt, x_train_unigrams_superlatives_manner_cnt, x_train_all_bin, x_train_all_cnt];
# names = ['uni + metadata','uni + superlatives_bin + manner_bin', 'uni + senti_cnt', 'uni + superlatives_cnt + manner_cnt', 'uni + all_bin', 'uni + all_cnt'];
# validation_data_sets = [x_valid_unigrams_metadata, x_valid_unigrams_superlatives_manner_bin, x_valid_unigrams_senti_cnt, x_valid_unigrams_superlatives_manner_cnt, x_valid_all_bin, x_valid_all_cnt];

svm_file = open('svm_results','w'); 
svm_file.write('============== SIX WAY ============== \n \n');

supvec = SVC(kernel='linear')
print '';
print '=================== SUPPORT VECTOR MODELS =================='
print '================== Six-way Classification =================='
print ''

y_train = train_six_labels; 
y_valid = valid_six_labels;

i = 0;
while i < len(training_data_sets):
	x_train = training_data_sets[i];
	x_valid = validation_data_sets[i];
	supvec.fit(x_train, y_train);
	score = supvec.score(x_valid,y_valid);
	y_valid_predict = supvec.predict(x_valid);
	svm_file.write(names[i] + ': ' + str(score) + '\n');
	svm_file.write(str(classification_report(y_valid,y_valid_predict)));
	# print(names[i] + ': ' + str(supvec.score(x_valid,y_valid)));
	i = i + 1;

print ''
print '================== Binary Classification =================='
print ''

svm_file.write('============== BINARY ==============\n \n');

y_train = train_binary_labels; 
y_valid = valid_binary_labels;

i = 0;
while i < len(training_data_sets):
	x_train = training_data_sets[i];
	x_valid = validation_data_sets[i];
	supvec.fit(x_train, y_train);
	score = supvec.score(x_valid,y_valid);
	y_valid_predict = supvec.predict(x_valid);
	svm_file.write(names[i] + ': ' + str(score) + '\n');
	svm_file.write(str(classification_report(y_valid,y_valid_predict)));
	# print(names[i] + ': ' + str(supvec.score(x_valid,y_valid)));
	i = i + 1;

print("EOF")
