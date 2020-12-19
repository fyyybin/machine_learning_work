# -*- coding: utf-8 -*-
from sklearn.multiclass import OneVsRestClassifier  # 结合SVM的多分类组合辅助器
import sklearn.svm as svm  # SVM辅助器
import jieba
from numpy import *
import os
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import CountVectorizer  # 词频矩阵


def readFile(path):
    with open(path, 'r', errors='ignore', encoding='gbk') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        file.close()
        return content

def saveFile(path, result):
    with open(path, 'w', errors='ignore', encoding='gbk') as file:
        file.write(result)
        file.close()

def segText(inputPath):
    data_list = []
    label_list = []
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + "/" + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            # print( " ".join(cutResult))
            label_list.append(eachDir)
            data_list.append(" ".join(cutResult))
    return data_list, label_list

def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList

def getTFIDFMat(train_data, train_label, stopWordList):  # 求得TF-IDF向量
    class0 = ''
    class1 = ''
    class2 = ''
    class3 = ''
    for num in range(len(train_label)):
        if train_label[num] == '体育':
            class0 = class0 + train_data[num]
        elif train_label[num] == '女性':
            class1 = class1 + train_data[num]
        elif train_label[num] == '文学出版':
            class2 = class2 + train_data[num]
        elif train_label[num] == '校园':
            class3 = class3 + train_data[num]
    train = [class0, class1, class2, class3]
    vectorizer = CountVectorizer(stop_words=stopWordList,
                                 min_df=0.5)  # 其他类别专用分类，该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    cipin = vectorizer.fit_transform(train)
    tfidf = transformer.fit_transform(cipin)  # if-idf中的输入为已经处理过的词频矩阵
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    train_cipin = vectorizer.transform(train_data)
    train_arr = transformer.transform(train_cipin)
    clf = model.fit(train_arr, train_label)

    while 1:
        print('请输入需要预测的文本:')
        a = input()
        sentence_in = [' '.join(jieba.cut(a))]
        b = vectorizer.transform(sentence_in)
        c = transformer.transform(b)
        prd = clf.predict(c)
        print('预测类别：', prd[0])

if __name__ == '__main__':
    data, label = segText('data')
    stopWordList = getStopWord('stop/stopword.txt')  # 获取停用词表
    getTFIDFMat(train_data=data, train_label=label, stopWordList=stopWordList)