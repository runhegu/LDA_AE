import pandas as pd
import pandas
import os

import re

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np
import warnings
import time

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
# support chinese
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

result = {}


def compute_onehot(data, string):
    if string == "梯户比例":
        data[data["梯户比例"].isna()]["梯户比例"] = "梯户比例暂无数据"
    elif string == "房屋用途":
        data.dropna(axis=0, how="all", subset=["房屋用途"], inplace=True)
    elif string == "装修情况":
        data.dropna(axis=0, how="all", subset=["装修情况"], inplace=True)

    # 创建OneHotEncoder对象
    encoder = OneHotEncoder(drop="first")

    # 拟合并转换数据
    encoded_data = encoder.fit_transform(pd.DataFrame(data[string])).toarray()
    # 将编码后的数据转换为DataFrame
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

    return encoded_df


def compute_avg_timeprice(data):
    group_data = data.groupby("挂牌时间")
    avg_time_weight = {}
    weight = []
    for key, values in group_data:
        avg_time_weight[key] = np.average([float(v) for v in values["单价"]])
    for index, rows in data.iterrows():
        weight.append(avg_time_weight[rows["挂牌时间"]])

    return weight


def compute_avg_price(data):
    ldata = data[["location", "单价"]]
    group_data = ldata.groupby("location")
    avg_price_weight = {}
    weight = []
    for key, values in group_data:
        avg_price_weight[key] = np.average([float(v) for v in values["单价"]])
    for index, rows in data.iterrows():
        weight.append(avg_price_weight[rows["location"]])
    return weight


def compute_corefsell_tdidf_corf(data):
    all_string = []

    for index, row in data.iterrows():
        if pd.isna(row["核心卖点"]):
            all_string.append("")
        else:
            all_string.append(row["核心卖点"])

    with open("./stopwords.txt", mode="r", encoding="UTF8") as f:
        stopwords = [item.strip() for item in f.readlines()]
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(all_string)
    weight = []

    for doc in tfidf_matrix:
        doc_array = doc.toarray().ravel()
        weight.append(np.average(doc_array))

    return weight


def compute_title_tdidf_corf(data):
    all_string = []

    for index, row in data.iterrows():
        if pd.isna(row["标题"]):
            all_string.append("")
        else:
            all_string.append(row["标题"])

    with open("./stopwords.txt", mode="r", encoding="UTF8") as f:
        stopwords = [item.strip() for item in f.readlines()]
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(all_string)
    weight = []

    for doc in tfidf_matrix:
        doc_array = doc.toarray().ravel()
        weight.append(np.average(doc_array))

    return weight


def house_type_get_num(str):
    numbers = re.findall(r'\d+', str)
    return numbers


def wash_all_data(data):
    use_col = ["房屋用途", "装修情况", "房本备件", "建筑结构", "房屋朝向", "建筑类型", "产权所属", "房屋年限"]
    print("正在初始化数据")
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["location"] = data.loc[:, "地段"] + data.loc[:, "小区名"]
    data["location"] = data["location"].str.replace(" ", "")
    data["location"] = data["location"].str.replace("\'", "") \
        .str.replace("[", "").str.replace("]", "").str.replace(",", "")
    data.drop_duplicates(inplace=True)
    data.drop(inplace=True, columns=["地段", "小区名"])
    data["挂牌时间"] = pd.to_datetime(data["挂牌时间"], errors='coerce')
    data.dropna(axis=0, inplace=True, how="all")
    data.dropna(axis=0, inplace=True, subset=["挂牌时间", "房屋用途", "装修情况", "房本备件", "建筑类型"])
    data["总价格"] = data["总价格"].str.replace("万", "")
    data.loc[data["配备电梯"] == "暂无数据", "配备电梯"] = 0
    data.loc[data["配备电梯"] == "无", "配备电梯"] = 0
    data.loc[data["配备电梯"] == "有", "配备电梯"] = 1
    data.loc[data["配备电梯"].isna(), "配备电梯"] = 0
    pattern = r'\d+(\.\d+)?'
    for index, rows in data.iterrows():
        data.loc[index, "建筑面积"] = re.search(pattern, rows["建筑面积"]).group()
    data["建筑面积"] = data["建筑面积"].astype("float64")
    data.loc[data["上次交易"] == "暂无数据", "上次交易"] = 0
    data["上次交易"] = pd.to_datetime(data["上次交易"], errors='coerce')
    data["上次交易"] = data["上次交易"].astype("int64") / 10**9
    data["挂牌时间"] = data["挂牌时间"].astype("int64") / 10**9
    data["单价"] = data["单价"].astype("int64")
    # data["总价格"] = data["总价格"].astype("float64")
    data.drop(["总价格"], axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    for item in use_col:
        data = pd.concat([data, compute_onehot(data, string=item)], axis=1)
    print(data.columns)
    print(data)
    print("初始化完成")
    return data


def get_all():
    original_data = pd.DataFrame(
        columns=["总价格", "单价", "地段", "小区名", "挂牌时间", "核心卖点", "房屋户型", "配备电梯",
                 "建筑面积",
                 "产权所属",
                 "房本备件",
                 "上次交易",
                 "装修情况",
                 "房屋用途",
                 "梯户比例",
                 "建筑结构",
                 "房屋朝向",
                 "建筑类型",
                 "房屋年限",
                 "标题",
                 ])
    root_path = "./data"
    dirlist = os.listdir(root_path)
    for dir in dirlist:
        dir_path = root_path + r"/" + dir
        filelist = os.listdir(dir_path)
        for file in filelist:
            file_path = dir_path + r"/" + file
            try:
                data = pd.read_csv(file_path,
                                   usecols=["总价格", "单价", "地段", "小区名", "挂牌时间", "核心卖点", "房屋户型",
                                            "配备电梯",
                                            "建筑面积",
                                            "产权所属",
                                            "房本备件",
                                            "上次交易",
                                            "装修情况",
                                            "房屋用途",
                                            "梯户比例",
                                            "建筑结构",
                                            "房屋朝向",
                                            "建筑类型",
                                            "房屋年限",
                                            "标题",
                                            ])
                original_data = pd.concat([original_data, data], axis=0)
            except pandas.errors.ParserError as e1:
                print(file + "打开出错")
                continue
            except ValueError as e2:
                print(file + "没有对应的数据")
                continue
    original_data.reset_index(drop=True, inplace=True)
    return original_data


def get_loc_data(data):
    loc = pd.read_csv("./all_loc_match.csv")[["location", "latlon"]]
    data = pd.merge(data, loc, how="left", on="location")
    loc_series = data["latlon"].str.split(",", expand=True)
    data["lon"] = loc_series.iloc[:, 0].astype("float64")
    data["lat"] = loc_series.iloc[:, 1].astype("float64")
    return data


def get_unlabel_data():
    data = wash_all_data(get_all())
    changzhou_city = {'320402': '天宁区', '320404': '钟楼区', '320405': '戚墅堰区', '320411': '新北区',
                      '320412': '武进区',
                      '320481': '溧阳市', '320482': '金坛市'}

    # %%
    # 对房屋户型权重计算
    print("开始计算房屋户型分类:")
    house_type_weight = []
    # 计算各个屋子的权重系数
    house_type_list = {}
    for index, item in tqdm(data[["单价", "房屋户型"]].iterrows()):
        if pd.isna(item["房屋户型"]):
            continue
        nums = str([int(n) for n in house_type_get_num(item["房屋户型"])])
        price = int(item["单价"])
        if nums not in house_type_list:
            house_type_list[nums] = [price, ]
        else:
            house_type_list[nums].append(price)
    print("开始计算房屋户型权重系数:")
    sum = 0
    num = 0
    for key, values in house_type_list.items():
        for v in values:
            sum += v
            num += 1
        house_type_list[key] = sum / num
        sum = 0
        num = 0
    print("计算完成")
    print("开始计算房屋户型权重:")
    for index, item in tqdm(data[["单价", "房屋户型"]].iterrows()):
        price = int(item["单价"])
        if pd.isna(item["房屋户型"]):
            # 未填值则默认一室一厅一厨一卫
            house_type_weight.append(price)
        else:
            nums = str([int(n) for n in house_type_get_num(item["房屋户型"])])
            house_type_weight.append(house_type_list[nums])
    data["房屋户型weight"] = house_type_weight

    weight = compute_corefsell_tdidf_corf(data)
    data["核心卖点weight"] = weight

    avg_price_weight = compute_avg_price(data)
    data["地区平均单价"] = avg_price_weight

    data["同一时间单价"] = compute_avg_timeprice(data)

    data["标题weight"] = compute_title_tdidf_corf(data)

    # %%
    print("开始整合数据集")
    match_data = pd.read_csv("./all_loc_match.csv", usecols=["location", "adcode"])
    cat_data = data.merge(match_data, how="left", on="location")
    cat_data = pd.concat([cat_data, compute_onehot(cat_data, string="梯户比例")], axis=1)
    cat_data.dropna(axis=0, inplace=True, how="all", subset=["adcode"])
    cat_data.reset_index(drop=True, inplace=True)
    cat_data = get_loc_data(cat_data)
    cat_data.dropna(axis=0, subset=["latlon"], inplace=True)
    return cat_data

# get train_data.csv file
def get_traindata():
    del_col = ["房屋用途", "装修情况", "房本备件", "建筑结构", "房屋朝向", "建筑类型", "产权所属", "核心卖点",
               '房屋户型', '梯户比例', 'location', '房屋年限', "标题", "latlon"]
    train_data = get_unlabel_data()
    train_data["label"] = [None for i in range(len(train_data))]
    train_data["clabel"] = [None for i in range(len(train_data))]
    group_data = train_data.groupby("location")
    for key, values in tqdm(group_data, leave=False):
        # 初始化价格走势标签
        y = pd.DataFrame(columns=["location", "index", "label", "clabel"])
        if len(values["单价"]) < 2:
            result[key] = 3
            train_data = train_data[train_data["location"] != key]
        else:
            group_sorted_data = values[[
                "单价",
                "挂牌时间",
            ]].sort_values(by="挂牌时间", axis=0, ascending=True)
            v0 = group_sorted_data["单价"].iloc[0]
            f_index = 0
            i = 0
            for index, g in group_sorted_data.iterrows():
                if i == 0:
                    f_index = index
                    i += 1
                    continue
                v = g["单价"]
                if v > v0:
                    y.loc[len(y)] = {"location": key, "index": f_index, "clabel": 1, "label": v-v0}
                    v0 = v
                    f_index = index
                elif v0 == v:
                    y.loc[len(y)] = {"location": key, "index": f_index, "clabel": 0, "label": v-v0}
                    v0 = v
                    f_index = index
                elif v < v0:
                    y.loc[len(y)] = {"location": key, "index": f_index, "clabel": 2, "label": v-v0}
                    v0 = v
                    f_index = index
            for index, row in y.iterrows():
                train_data.loc[row["index"], "label"] = row["label"]
                train_data.loc[row["index"], "clabel"] = row["clabel"]
    train_data.drop(labels=del_col, axis=1, inplace=True)
    print(train_data)
    train_data.to_csv("train_data.csv")


if __name__ == '__main__':
    nowtime = time.time()
    get_traindata()
    print("本程序运算时间为{}秒".format(time.time() - nowtime))
