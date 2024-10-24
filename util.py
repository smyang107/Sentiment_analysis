# -*- coding: utf-8 -*-
import jieba
import re
from sklearn import metrics
import numpy as np

def results_report(total_oushi,hs_oushi,df2):
    accuracy_list_0 = []
    accuracy_list_1 = []
    accuracy_list_2 = []
    accuracy_list = []
    for (i,results_i) in enumerate(total_oushi):
        y_test = np.array(df2['label']).astype('float64')
        y_pred = np.array(results_i).astype('float64')
        print(f'相似度：{hs_oushi[i]}')
        results_report = metrics.classification_report(y_test,y_pred,digits=3,output_dict=True)
        accuracy_list_0.append(round(results_report['0.0']['precision'],3))
        accuracy_list_1.append(round(results_report['1.0']['precision'],3))
        accuracy_list_2.append(round(results_report['2.0']['precision'],3))
        print(metrics.classification_report(y_test,y_pred,digits=4))
        print("准确率:", metrics.accuracy_score(y_test, y_pred))
        accuracy_list.append(round(metrics.accuracy_score(y_test, y_pred),3))
        print("*"*50)
    return accuracy_list_0,accuracy_list_1,accuracy_list_2,accuracy_list


import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker

def pic_chinese(hs,list_0,list_1,list_2,list_3,pic_name):
    c = (
        Line(init_opts=opts.InitOpts(renderer='svg'))
        .add_xaxis([str(x) for x in list(np.round(hs,2))]) #X轴
        .add_yaxis("积极",list_0,symbol="diamond",symbol_size=10) #Y轴
        .add_yaxis("消极",list_1,symbol="rect",symbol_size=10) #Y轴
        .add_yaxis("中性",list_2,symbol="circle",symbol_size=10) #Y轴
        .add_yaxis("均值",list_3,symbol="triangle",symbol_size=10,
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False,position='right', font_size= 24),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="category", name="相似度阈值",name_location='center',name_gap=40,axislabel_opts=opts.LabelOpts(font_size = 20),name_textstyle_opts=opts.TextStyleOpts(font_size = 20)),
            yaxis_opts=opts.AxisOpts(name="准确率",name_location='middle',name_gap=40,axislabel_opts=opts.LabelOpts(font_size = 20),name_textstyle_opts=opts.TextStyleOpts(font_size = 20)),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(is_show= True,textstyle_opts=opts.TextStyleOpts(font_size = 24))
        )
    )
    c.render_notebook() # 显示
    c.render(f'./tubiao/{pic_name}_chinese.html')
    
    

stopwords = []
with open("../data/stopwords_copy.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())


def load_corpus(path):
    """
    加载语料库
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing(content)
            data.append((content, int(seniment)))
    return data


def load_corpus_bert(path):
    """
    加载语料库
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing_bert(content)
            data.append((content, int(seniment)))
    return data


def processing(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    # 分词
    words = [w for w in jieba.lcut(text) if w.isalpha()]
    # 对否定词`不`做特殊处理: 与其后面的词进行拼接
    while "不" in words:
        index = words.index("不")
        if index == len(words) - 1:
            break
        words[index: index+2] = ["".join(words[index: index+2])]  # 列表切片赋值的酷炫写法
    # 用空格拼接成字符串
    result = " ".join(words)
    return result


def processing_bert(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    ##text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    return text