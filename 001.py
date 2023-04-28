import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def __init__(self):
        self.clf = None  # 分类器
        self.vocab = None  # 词汇表

    def preprocess(self, text):
        """
        数据预处理，将文本转换为小写并去除数字和标点符号
        """
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text):
        """
        特征提取，将文本分词，并仅保留名词、动词、形容词和副词
        """
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        tokens = [word for word, tag in pos_tags if tag.startswith('N') or tag.startswith('V') or tag.startswith('J') or tag.startswith('R')]
        return tokens

    def train(self, X, y):
        """
        训练模型
        """
        self.clf = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
        self.clf.fit(X, y)
        self.vocab = self.clf.named_steps['vect'].get_feature_names()

    def predict(self, X):
        """
        预测新数据
        """
        preds = self.clf.predict_proba(X)
        max_probs = np.amax(preds, axis=1)
        return [self.clf.classes_[np.argmax(pred)] for pred in preds]

if __name__ == '__main__':
    # 加载数据
    data = pd.read_csv('data.csv')
    X = data['text'].tolist()
    y = data['label'].tolist()

    # 数据预处理和特征提取
    nb = NaiveBayes()
    X = [nb.preprocess(text) for text in X]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型并预测测试集数据
    nb.train(X_train, y_train)
    preds = nb.predict(X_test)

    # 打印分类结果和准确率
    print('Predictions:', preds)
    print('Accuracy:', sum(np.array(preds) == np.array(y_test)) / len(y_test))
