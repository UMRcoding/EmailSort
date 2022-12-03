#!/usr/bin/env python
# coding: utf-8

import os
import jieba
import re

"""获取中文停用词表"""
def get_stop_words():
    # 读取停用词文件
    stopwords_filepath = 'D:\\Study\\WorkSpace\\PyCharm\\liuheng\\Word02\\data\\stopwords_1208.txt'
    wordslist = [line.strip() for line in open(stopwords_filepath, 'r', encoding='utf-8').readlines()]
    return wordslist;


"""批量获取邮件内容"""
def get_content(filePath):
    filenames = os.listdir(filePath)
    content = []
    for filename in filenames:
        for line in open(filePath + filename):
            content.append(line[:len(line) - 1])
    return content


"""批量获取邮件的词语群，过滤非中文，删除停用词，创建词语字典"""
def get_email_words(content, stopList):
    wordDict = {}
    # pattern = re.compile(r'[^\u4e00-\u9fa5]')
    # zhContent = re.sub(pattern, '', str(content))
    # allWordsList = list(jieba.cut(content))
    for word in content:
        word = jieba.cut(word)
        if word not in stopList and word.strip() != '' and word != None:
            if word not in wordDict.keys():
                wordDict[word] = 1
            else:
                wordDict[word] += 1
    return wordDict


"""计算每封邮件词语的分类影响，并提取影响值最大的前30个，返回每封邮件是垃圾邮件的贝叶斯概率"""
def get_lable_words(spamDict, normDict, filePath, spamFileNum, normFileNum):
    testResult = {}
    filenames = os.listdir(filePath)
    for filename in filenames:
        content = []
        influenceDict = {}
        for line in open(filePath + filename):
            content.append(line[:len(line) - 1])
        words = get_email_words(content, stopList)
        for word in words:
            if word not in spamDict.keys() and word not in normDict.keys():
                influenceDict[word] = 0.4
            else:
                if word in spamDict.keys() and word in normDict.keys():
                    pspam = spamDict[word] / spamFileNum
                    pnorm = normDict[word] / normFileNum
                if word in spamDict.keys() and word not in normDict.keys():
                    pspam = spamDict[word] / spamFileNum
                    pnorm = 0.01
                if word not in spamDict.keys() and word in normDict.keys():
                    pspam = 0.01
                    pnorm = normDict[word] / normFileNum
                p = pspam / (pspam + pnorm)
                influenceDict[word] = p
        influenceDict = dict(sorted(list(influenceDict.items())[:30], key=lambda d: d[1], reverse=True))
        p = calBayes(influenceDict)
        if p > 0.9:
            testResult[filename] = 1
        else:
            testResult[filename] = 0
    return testResult


"""计算一封邮件是垃圾邮件的贝叶斯概率"""
def calBayes(influenceDict):
    pnorm = 1
    pspam = 1
    for word, influence in influenceDict.items():
        pspam *= influence
        pnorm *= (1 - influence)
    p = pspam / (pspam + pnorm)
    return p


"""计算分类准确率"""
def testAccuracy(testResult):
    rightCount = 0
    errorCount = 0
    for filename, result in testResult.items():
        if (int(filename) < 1000 and result == 0) or (int(filename) > 1000 and result == 1):
            rightCount += 1
        else:
            errorCount += 1
    return rightCount / (rightCount + errorCount)

def readwords_list(filepath):
    '''
    读取文件函数，以行的形式读取词表，返回列表
    :param filepath: 路径地址
    :return:
    '''
    wordslist = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return wordslist

def create_dataset(index_path):
    labels_list = []
    filename_list = []
    result = []
    try:
        wordslist = readwords_list(index_path)
        for item in wordslist: # 先去重
            if item not in result:
                result.append(item)
        for item in result:
            line = item.split(' ')
            labels_list.append(line[0])
            filename_list.append(line[1].strip('\n').split('/')[-1])
        return labels_list, filename_list
    except Exception as e:
        print("\033[1;31m文件读取出错:\033[0m" + index_path)
        print(e)

def type_sort(labels_list, filename_list):
    '''
     # 根据结果将邮件分类
    :param labels_list:
    :param filename_list:
    :return:
    '''
    ham_list = []
    spam_list  = []
    other_list = []
    for i in range(0,len(labels_list)):
        if labels_list[i] == 'ham':
            ham_list.append(filename_list[i])
        else:
            other_list.append(filename_list[i])

        if labels_list[i] == 'spam':
            spam_list.append(filename_list[i])
        else:
            other_list.append(filename_list[i])
    return ham_list, spam_list

def readwords_str(filepath):
    '''
    读取文件函数，以行的形式读取词表，返回str
    UTF-8, GBK 高阶的 GB18030, windows-1252
    :param filepath: 路径地址
    :return:
    '''
    try:
        try:
            with open(filepath, 'r', encoding='UTF-8') as f:
                str = f.read()
            return str
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='GB18030') as f:
                    str = f.read()
                return str
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='windows-1252', errors='ignore') as f:
                    str = f.read()
                return str
    except Exception as e:
        print('文件读取出错' + filepath)
        print(e)

def GetPathContent(normFileList,spamFileList):
    '''
    根据名称集合查找各自对应内容，返回集合
    :param normFileList:
    :param spamFileList:
    :return:
    '''
    ham_content_list = []
    spam_content_list = []
    for item in normFileList:
        itempath = base_path + 'data\\' + item
        ham_content = readwords_str(itempath)
        if(len(ham_content)<50):
            print("\033[1;31m数据读取为{},字节流过小，可能为异常数据~~ URL:\033[0m".format(len(ham_content)) + itempath)
        else:
            ham_content_list.append(ham_content)
    for item in spamFileList:
        itempath = base_path + 'data\\' + item
        spam_content = readwords_str(itempath)
        if (len(spam_content) < 50):
            print("\033[1;31m数据读取为{},字节流过小，可能为异常数据~~ URL:\033[0m".format(len(spam_content)) + itempath)
        else:
            spam_content_list.append(spam_content)
    return ham_content_list,spam_content_list


base_path = 'E:\\大学课程\\大四\\信息内容安全\\trec07p\\'
influenceDict = {}

spamFileNum = 200   #7775
normFileNum = 200   #7063

stopList = get_stop_words()
delay_path = base_path + 'delay\\index'
delay_labels_list, delay_filename_list = create_dataset(delay_path)

# 根据结果将邮件分类
delay_norm_filelist, delay_spamfile_list = type_sort(delay_labels_list, delay_filename_list)
normContent, spamContent= GetPathContent(delay_norm_filelist, delay_spamfile_list)

normDict = get_email_words(normContent, stopList)
spamDict = get_email_words(spamContent, stopList)


testContent_path = base_path + 'data'
testContent = get_content(testContent_path)
testDict = get_email_words(testContent, stopList)

testResult = get_lable_words(spamDict, normDict, "./data/test/", spamFileNum, normFileNum)
result = testAccuracy(testResult)
print(result)