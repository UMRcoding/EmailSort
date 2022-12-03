import os
import re
from email import policy, parser
from html import unescape
from collections import Counter
import urlextract
import nltk
from nltk.corpus import stopwords  # 去英文停用词
import pandas
from sklearn.model_selection import train_test_split  # 随机划分样本，增加结果准确性
from sklearn import preprocessing, metrics, naive_bayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1 读取数据集
INDEX_PATH = os.path.join('delay', 'index').replace("\\","/")# 从性能考虑，先使用较小的数据集进行训练
DATA_PATH = os.path.join('data').replace("\\","/")  # 数据文件夹路径
labels = []
filenames = []


def create_dataset(index_path):
    with open(index_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split(' ')
            labels.append(line[0])
            filenames.append(line[1].strip('\n').split('/')[-1])
create_dataset(INDEX_PATH)


def load_email(filename, file_path):
    with open(os.path.join(file_path, filename), 'rb') as f:
        return parser.BytesParser(policy=policy.default).parse(f)


raw_emails = [load_email(name, DATA_PATH) for name in filenames]
print(len(raw_emails))
print(raw_emails[1].get_content().strip())  # 查看邮件文本内容


# 2 数据预处理
# 构造函数获取邮件的结构类型及其计数
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return 'multipart({})'.format(', '.join([
            get_email_structure(sub_email)
            for sub_email in payload]))
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub(r'<[aA]\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub(r'<img\s.*?>', ' IMAGE ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)
def email_to_text(email):
    html = None
    # walk()打印出一封具有多部分结构之信息的每个部分的MIME类型
    for part in email.walk():
        ctype = part.get_content_type()
        if ctype not in ('text/plain', 'text/html'):
            continue
        try:
            content = part.get_content()
        except LookupError:
            content = str(part.get_payload())
        if ctype == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


print(structures_counter(raw_emails).most_common())  # 显示邮件包含的类型

# 3 分词
stopwords_list = stopwords.words('english')  # 去英文停用词
token = nltk.stem.SnowballStemmer('english')
alpha = 'abcdefghigklmnopqrstuvwxyz'
for single in alpha:
    stopwords_list.append(single)
extractor = urlextract.URLExtract()


def word_split(email):
    text = email_to_text(email) or ' '
    text = text.lower()
    text = re.sub(r'\W+', ' ', text, flags=re.M)
    urls = list(set(extractor.find_urls(text)))
    urls.sort(key=lambda item: len(item), reverse=True)
    for url in urls:
        text = text.replace(url, "URL")
    text = re.sub(r'\d+(?:\.\d*[eE]\d+)?', 'NUMBER', text)
    content = list(nltk.word_tokenize(text))
    all_words = []
    for word in content:
        if word not in stopwords_list:
            word = token.stem(word)
            all_words.append(word)
    return all_words


all_emails = [word_split(data) for data in raw_emails]
print(all_emails[0])  # 查看分词结果

# 4 特征提取
# 创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = all_emails
trainDF['label'] = labels

# 将数据集分为测试集和训练集
train_data, test_data, train_label, test_label = train_test_split(trainDF['text'], trainDF['label'], random_state=0)

# label编码为目标变量,即从字符串转为一个数字
encoder = preprocessing.LabelEncoder()
train_label = encoder.fit_transform(train_label)
test_label = encoder.fit_transform(test_label)

# 4.1 计数特征向量
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
text=trainDF['text'].map(' '.join)
train_data_test=train_data.map(' '.join)
test_data_test=test_data.map(' '.join)
count_vect.fit(text)
xtrain_count = count_vect.transform(train_data_test)  # 训练集特征向量
xtest_count = count_vect.transform(test_data_test)  # 测试集特征向量

# 4.2 TF-IDF特征向量
# 4.2.1 词语级
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(text)
xtrain_tfidf = tfidf_vect.transform(train_data_test)
xtest_tfidf = tfidf_vect.transform(test_data_test)
# 4.2.2 多词语级
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
tfidf_vect_ngram.fit(text)
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_data_test)
xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_data_test)
# 4.2.3 词性级
tfidf_vect_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
tfidf_vect_char.fit(text)
xtrain_tfidf_char = tfidf_vect_char.transform(train_data_test)
xtest_tfidf_char = tfidf_vect_char.transform(test_data_test)


# 5 分类任务
# 创建分类器
def train_model(classifier, train_feature, test_feature):
    classifier.fit(train_feature, train_label)
    prediction = classifier.predict(test_feature)
    acc = metrics.accuracy_score(prediction, test_label)
    prec = metrics.precision_score(prediction, test_label)
    rec = metrics.recall_score(prediction, test_label)
    f1 = metrics.f1_score(prediction, test_label)
    return acc, prec, rec, f1


# 5.1 朴素贝叶斯多项式模型
# 5.1.1 计数特征向量
accuracy, precision, recall, f1_socre = train_model(naive_bayes.MultinomialNB(), xtrain_count, xtest_count)
print("NB, Count Vectors: ", accuracy)
# 5.1.2 词语级别tf-idf特征向量
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, xtest_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy)
# 5.1.3 多词语级别tf-idf特征向量
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, xtest_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy)
# 5.1.4 词性级别tf-idf特征向量
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_char, xtest_tfidf_char)
print("NB, CharLevel Vectors: ", accuracy)

# 5.2 朴素贝叶斯伯努利模型
# 5.2.1 计数特征向量
accuracy, precision, recall, f1_socre = train_model(naive_bayes.BernoulliNB(), xtrain_count, xtest_count)
print("LR, Count Vectors: ", accuracy)
# 5.2.2 词语级别tf-idf特征向量
accuracy = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf, xtest_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)
# 5.2.3 多词语级别tf-idf特征向量
accuracy = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf_ngram, xtest_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)
# 5.2.4 词性级别tf-idf特征向量
accuracy = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf_char, xtest_tfidf_char)
print("LR, CharLevel Vectors: ", accuracy)
