import math

import numpy as np
import pandas as pd

from keras.api.layers import Input, Dense, Dropout, GaussianDropout, GaussianNoise
from keras.api.models import Model
from keras.api.optimizers import RMSprop
from keras.api.losses import mean_squared_error
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans, Birch, MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score,normalized_mutual_info_score, adjusted_rand_score,fowlkes_mallows_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA, LatentDirichletAllocation

# data we use
# The clabel, label and adcode column are used in the pre-experiment and are not related to the current experiment
data = pd.read_csv("train_data.csv").drop(["Unnamed: 0", 'clabel', 'label', 'adcode'], axis=1)

data["挂牌时间"] = pd.to_datetime(data["挂牌时间"] * 10 ** 9)
data["year"] = data["挂牌时间"].dt.year
data["month"] = data["挂牌时间"].dt.month

data.drop(["挂牌时间", '上次交易'], inplace=True, axis=1)
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
print(data.shape, data.columns)


# # resampling
# fraction = 0.3
# sample_data = data.sample(frac=fraction, random_state=42)

# Normalization
wash_data = data.copy(deep=True)
# wash_data = sample_data.copy(deep=True)
minmaxscaler = MinMaxScaler()
wash_data = minmaxscaler.fit_transform(wash_data)

# # noise adding
# noise_mean = 0.5
# noise_std = 0.15
# n_rate = 0.1
# for i in tqdm.tqdm(range(0, math.floor(len(wash_data)*n_rate))):
#     wash_data = np.vstack((wash_data, [abs(i) for i in np.random.normal(noise_mean, noise_std, size=wash_data.shape[1])]))
# np.random.shuffle(wash_data)

test_data = wash_data.copy()

# LDA
decomposistion = LatentDirichletAllocation(n_components=64, random_state=0, verbose=1)
wash_data = decomposistion.fit_transform(wash_data)

# Encoder
input_dim = wash_data.shape[1]
encoding_dim = 32  # 可以根据需要调整编码维度

# Encoder1
input_img = Input(shape=(input_dim,))
# noise = GaussianNoise(1)(input_img)
# dropout = GaussianDropout(0.2)(noise)
# encoded = Dense(encoding_dim, activation='relu')(dropout)
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# # Encoder2
# input_img1 = Input(shape=(test_data.shape[1],))
# # noise1 = GaussianNoise(1)(input_img1)
# encoded1 = Dense(encoding_dim, activation='relu')(input_img1)
# decoded1 = Dense(test_data.shape[1], activation='sigmoid')(encoded1)

autoencoder = Model(input_img, decoded, name="autoencoder")
# onlyautoencoder = Model(input_img1, decoded1, name="onlyautoencoder")
print(autoencoder.summary())

encoder = Model(input_img, encoded, name="encoder")
print(encoder.summary())

# encoder1 = Model(input_img1, encoded1, name="encoder1")

# compile encoder
autoencoder.compile(optimizer=RMSprop(learning_rate=0.01), loss=mean_squared_error)
# onlyautoencoder.compile(optimizer=RMSprop(learning_rate=0.01), loss='mean_squared_error')

# train encoder
autoencoder.fit(wash_data, wash_data,
                epochs=200,
                batch_size=256,
                shuffle=True,
                validation_data=(wash_data, wash_data)
                )
# onlyautoencoder.fit(test_data, test_data,
#                     epochs=200,
#                     batch_size=256,
#                     shuffle=True,
#                     validation_data=(test_data, test_data)
#                     )

# get features
encoded_imgs = encoder.predict(wash_data)
# onlyencoded_imgs = encoder1.predict(test_data)

# test best n_clusters
# ss = []
# ch = []
# db = []
# 70类聚类效果最好
# for i in tqdm.tqdm(range(65, 75)):
    # # # 使用KMeans进行聚类
    # AEmodel = KMeans(n_clusters=3, random_state=0).fit(encoded_imgs)
    # model = KMeans(n_clusters=3, random_state=0).fit(wash_data)
    # db.append(davies_bouldin_score(encoded_imgs, cluster_labelsAE))
    # ch.append(calinski_harabasz_score(encoded_imgs, cluster_labelsAE))
    # ss.append(silhouette_score(encoded_imgs, cluster_labelsAE))

# plt.figure(figsize=(10, 10))
# plt.plot(range(65, 75), ss, c="green", label="ss")
# plt.plot(range(65, 75), db, c="red", label="db")
# plt.plot(range(65, 75), ch, c="blue", label="ch")
# plt.legend()
# # plt.savefig("result逼近")
# plt.show()

# max_value = max(ch)
# max_indices = [i for i, x in enumerate(ch) if x == max_value]
# print(max_indices)
# min_value = min(db)
# min_indices = [i for i, x in enumerate(db) if x == min_value]
# print(min_indices)

AEmodel = KMeans(n_clusters=70).fit(encoded_imgs)
# "LDA+AE+KMeans"
# model = KMeans(n_clusters=70).fit(wash_data)
# "LDA+KMeans"
KMeans_model = KMeans(n_clusters=70).fit(test_data)
# "KMeans"
# AE_KMeans_model = KMeans(n_clusters=70).fit(onlyencoded_imgs)
# "AE+KMeans"

# 获取聚类标签  
cluster_labelsAE = AEmodel.labels_
# cluster_labels = model.labels_
cluster_labelsONLY = KMeans_model.labels_
# cluster_labels_AE_DBSCAN = AE_KMeans_model.labels_


# result
print(f"AutoENCODER silhouette_score: {silhouette_score(encoded_imgs, cluster_labelsAE)}")
print(f"AutoENCODER calinski_harabasz_score: {calinski_harabasz_score(encoded_imgs, cluster_labelsAE)}")
print(f"AutoENCODER davies_bouldin_score: {davies_bouldin_score(encoded_imgs, cluster_labelsAE)}")

# # print(f"method2LDA+DBSCAN silhouette_score:{silhouette_score(wash_data, cluster_labels)}")
# # print(f"method2LDA+DBSCAN calinski_harabasz_score:{calinski_harabasz_score(wash_data, cluster_labels)}")
# # print(f"method2LDA+DBSCAN davies_bouldin_score:{davies_bouldin_score(wash_data, cluster_labels)}")
#
print(f"onlyDBSCAN silhouette_score:{silhouette_score(test_data, cluster_labelsONLY)}")
print(f"onlyDBSCAN calinski_harabasz_score:{calinski_harabasz_score(test_data, cluster_labelsONLY)}")
print(f"onlyDBSCAN davies_bouldin_score:{davies_bouldin_score(test_data, cluster_labelsONLY)}")

# print(f"only_rand_score:{adjusted_rand_score(label['Class'].tolist(), cluster_labelsONLY)}")
# print(f"LDA_rand_score:{adjusted_rand_score(label['Class'].tolist(), cluster_labelsAE)}")

# print(f"rand_score:{adjusted_rand_score(cluster_labelsAE, cluster_labelsONLY)}")

# print(f"AEDBSCAN silhouette_score:{silhouette_score(onlyencoded_imgs, cluster_labels_AE_DBSCAN)}")
# print(f"AEDBSCAN calinski_harabasz_score:{calinski_harabasz_score(onlyencoded_imgs, cluster_labels_AE_DBSCAN)}")
# print(f"AEDBSCAN davies_bouldin_score:{davies_bouldin_score(onlyencoded_imgs, cluster_labels_AE_DBSCAN)}")


# analysis
c = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
cname = list(c.values())


# ST
# data1 = data.copy(deep=True)
# data2 = data.copy(deep=True)
#
# column1 = data1.columns.to_list().copy()
# column1.append("clusterAE")
# column2 = data2.columns.to_list().copy()
# column2.append("clusteronly")
# mean_dataframe1 = pd.DataFrame(columns=column1)
# mean_dataframe2 = pd.DataFrame(columns=column2)
#
# data1["clusterAE"] = cluster_labelsAE
# data2["clusteronly"] = cluster_labelsONLY
#
# groupdata1 = data1.groupby(["clusterAE"])
# groupdata2 = data2.groupby(["clusteronly"])
# for key, values in groupdata1:
#     mean_dataframe1.loc[len(mean_dataframe1)] = values.mean().to_dict()
# for key, values in groupdata2:
#     mean_dataframe2.loc[len(mean_dataframe2)] = values.mean().to_dict()
#
# pd.set_option('display.max_columns', None)
# mean_dataframe1.drop(["clusterAE"], axis=1, inplace=True)
# mean_dataframe2.drop(["clusteronly"], axis=1, inplace=True)
# # print(mean_dataframe1)
# # print(mean_dataframe2)
# cname = list(c.values())
# minmaxscaler_plt1 = MinMaxScaler()
# wash_mean_dataframe1 = minmaxscaler_plt1.fit_transform(mean_dataframe1)
# minmaxscaler_plt2 = MinMaxScaler()
# wash_mean_dataframe2 = minmaxscaler_plt2.fit_transform(mean_dataframe2)
# # plt.subplot(121)
# # plt.title("LDA+AE")
# # for i, j in enumerate(wash_mean_dataframe1):
# #     plt.plot(j, c=cname[i+20], label=i, linestyle='none', marker='o')
# # plt.subplot(122)
# # plt.title("only")
# # for i, j in enumerate(wash_mean_dataframe2):
# #     plt.plot(j, c=cname[i + 20], label=i, linestyle='none', marker='o')
# # # plt.legend()
# # plt.show()
# # pd.DataFrame(wash_mean_dataframe1).to_excel("mean_result1", engine="openpyxl")

# def eucliDist(A, B):
#     return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))

# min_dic = 100000
# min_i = 0
# min_j = 0
# for index, row in pd.DataFrame(wash_mean_dataframe1).iterrows():
#     for index2, row2 in pd.DataFrame(wash_mean_dataframe1).iterrows():
#         v = eucliDist(row.to_list(), row2.to_list())
#         if v < min_dic and v != 0:
#             min_dic = v
#             min_i = index
#             min_j = index2
# print(min_i, min_j, min_dic)


# PCA
color = ["red", "green", "blue", "black"]

# # 3D_PDA
# visiualize1 = PCA(n_components=3, random_state=0)
# visiual_data1 = visiualize1.fit_transform(test_data)
# visiual_data1=pd.DataFrame(visiual_data1)
# visiual_data1["label"] = cluster_labelsONLY
# k = 0
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection='3d')
# for key, data in visiual_data1.groupby(["label"]):
#     # if k == 2:
#     #     break
#     ax.scatter(data.iloc[:, 0],data.iloc[:, 1], data.iloc[:,2], s=30, c=cname[k], cmap="jet", marker="^")
#     k += 1
# # for index, row in visiual_data1.iterrows():
# #     if k == 3000:
# #         break
# #     ax.scatter(row[0], row[1], row[2], s=30, c=cname[int(row["label"])], cmap="jet", marker="^")
# #     k += 1
# plt.show()
#

# 2D_PCA
visiualize1 = PCA(n_components=2, random_state=0)
visiual_data1 = visiualize1.fit_transform(test_data)
visiual_data1 = pd.DataFrame(visiual_data1)
visiual_data1["label"] = cluster_labelsONLY
k = 0
fig = plt.figure(figsize=(10, 10))
for key, data in visiual_data1.groupby(["label"]):
    plt.scatter(data.iloc[:, 0],data.iloc[:, 1], c=cname[k], cmap="jet", marker="^")
    k += 1
plt.show()
