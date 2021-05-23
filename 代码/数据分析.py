'''
班级：大数据1801
姓名：叶际荣
学号: 201806140014
'''
# 导入相关的库
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, \
    recall_score, f1_score, cohen_kappa_score
from matplotlib import rcParams
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, \
    median_absolute_error, r2_score
from sklearn.metrics import roc_curve


# 数据处理
def data_processing(data1, data2):
    print("表一形状：", data1.shape)
    print("表二形状：", data2.shape)
    data = pd.concat(([data1, data2]))  # 对数据表拼接
    data.columns
    print("拼接后表的形状：", data.shape)  # 查看data形状
    # print("查看各字段的空值", data.isnull().sum())  # 查看各字段的空值
    data = data.drop_duplicates(subset=['isbn'], keep='first')  # 对表进行去重 因为每本书只有唯一的isbn
    print("去重后表的形状：", data.shape)
    f_data = data[['price', 'price1']]
    data = data[['commnet', 'price', 'price1']]
    print("提取需要的数据后表的形状：", data.shape)
    data = data.dropna(subset=['commnet', 'price', 'price1'])
    print("除去commnet，和price以及price1为空的表：", data.shape)
    print("查看各值的最大值:\n", data.max())
    print("查看各值的最小值:\n", data.min())
    print("查看价格大于1000的数量;\n", pd.value_counts(data['price'] > 1000))
    data = data[data['price'] < 1000]  # 筛选price>1000的值
    print("筛选price大于1000的值后：", data.shape)

    # 标准差标准化commnet,price
    def StandardScaler(data):
        data = (data - data.mean()) / data.std()
        return data

    std_data = pd.concat([StandardScaler(data['commnet']),
                          StandardScaler(data['price']),
                          StandardScaler(data['price1'])], axis=1)

    km_model = KMeans(n_clusters=3).fit(std_data)  # 聚类
    km_model.labels_  # 查看聚类结果
    km_model.cluster_centers_  # 查看聚类中心
    # get_radar(km_model, data)  # 绘制聚类中心的雷达图
    # get_cluster(data, std_data)  # 构建并评价聚类模型
    get_classification(f_data)  # 构建并评价分类模型
    # get_Return(data)  # 构建并评价回归模型


# 绘制聚类中心的雷达图
def get_radar(km_model, data):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    labels = data.columns
    # 数据个数
    k = 3
    plot_data = km_model.cluster_centers_
    # 指定颜色
    color = ['b', 'g', 'r']
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    plot_data = np.concatenate((plot_data, plot_data[:, [0]]), axis=1)
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure()
    # polar参数
    ax = fig.add_subplot(111, polar=True)
    for i in range(len(plot_data)):
        # 画线
        ax.plot(angles, plot_data[i], 'o-', color=color[i], label='特征' + str(i), linewidth=2)

    ax.set_rgrids(np.arange(0.01, 7, 0.5), np.arange(-1, 7, 0.5), fontproperties="SimHei")
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    plt.legend(loc=4)
    plt.savefig('../聚类中心的雷达图.png')
    plt.show()


# scikit估计器构建k-Means构建聚类模型
def get_cluster(data, std_data):
    k_data = data[['price', 'price1']]
    scale = MinMaxScaler().fit(k_data)
    data_scale = scale.transform(k_data)
    kmeans = KMeans(n_clusters=3, random_state=123).fit(data_scale)
    print(kmeans)
    # 用TSNE降维速度太慢
    tsne = TSNE(n_components=2, init='random', random_state=177).fit(k_data)
    df = pd.DataFrame(tsne.embedding_)
    df['labels'] = kmeans.labels_
    df1 = df[df['labels'] == 0]
    df2 = df[df['labels'] == 1]
    df3 = df[df['labels'] == 2]
    plt.figure(figsize=(9, 6))
    plt.plot(df1[0], df1[1], 'bo', df2[0], df2[1], 'g>', df3[0], df3[1], 'r*')
    # plt.plot(df2[0], df2[1], 'g>')
    # plt.plot(df3[0], df3[1], 'r*')
    plt.savefig('../聚类结果可视化.png')
    # plt.show()
    # 评价聚类模型（轮廓系数评价法）
    silhouettteScore = []
    for i in range(2, 15):
        # 构建并训练模型
        kmeans = KMeans(n_clusters=i, random_state=123).fit(std_data)
        score = silhouette_score(std_data, kmeans.labels_)
        silhouettteScore.append(score)
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 15), silhouettteScore, linewidth=1.5, linestyle="-")
    plt.savefig('../轮廓系数评价法结果可视化.png')
    plt.show()


# 构建并评价分类模型
def get_classification(f_data):
    # 分类
    cancer_data1 = f_data
    index1 = f_data['price'] <= 50  # price大于或等于50的为一类
    index2 = f_data['price'] > 50  # price小于50的为一类
    cancer_data1.loc[index1, 'label'] = 0
    cancer_data1.loc[index2, 'label'] = 1
    print('查看price大于或等于50的数量:', sum(index1))  # 查看price大于或等于50的数量
    print('查看price小于50的数量：', sum(index2))  # 查看price小于50的数量
    cancer_data = cancer_data1[['price', 'price1']]
    cancer_target = cancer_data1['label']
    # 将数据划分为训练集测试集
    cancer_data_train, cancer_data_test, cancer_target_train, cancer_target_test = \
        train_test_split(cancer_data, cancer_target, test_size=0.2, random_state=22)
    # 数据标准化
    stdScaler = StandardScaler().fit(cancer_data_train)
    cancer_trainStd = stdScaler.transform(cancer_data_train)
    cancer_testStd = stdScaler.transform(cancer_data_test)

    # 建立SVM模型
    svm = SVC().fit(cancer_trainStd, cancer_target_train)
    print('建立的SVM模型为：\n', svm)
    cancer_target_pred = svm.predict(cancer_testStd)
    print('预测前20个结果为：\n', cancer_target_pred[:20])
    true = np.sum(cancer_target_pred == cancer_target_test)
    print('预测对的结果数目为：', true)
    print('预测错的的结果数目为：', cancer_target_test.shape[0] - true)
    print('预测结果准确率为：', true / cancer_target_test.shape[0])

    # 求出ROC曲线的x轴和y轴
    fpr, tpr, thresholds = \
        roc_curve(cancer_target_test, cancer_target_pred)
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 1)  # 设定x轴的范围
    plt.ylim(0.0, 1.1)  # 设定y轴的范围
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Postive Rate')
    plt.plot(fpr, tpr, linewidth=2, linestyle="-", color='red')
    plt.savefig('../ROC曲线结果可视化.png')
    plt.show()
    # print('使用SVM预测数据的准确率为：',
    #       accuracy_score(cancer_data_test, cancer_target_pred, average='micro'))
    # print('使用SVM预测数据的精确率为：',
    #       precision_score(cancer_data_test, cancer_target_pred, average='micro'))
    # print('使用SVM预测数据的召回率为：',
    #       recall_score(cancer_data_test, cancer_target_pred, average='micro'))
    # print('使用SVM预测数据的F1值为：',
    #       f1_score(cancer_data_test, cancer_target_pred, average='micro'))
    # print('使用SVM预测数据的Cohen’s Kappa系数为：',
    #       cohen_kappa_score(cancer_data_test, cancer_target_pred))


# 构建并评价回归模型
# 建立线性回归模型
def get_Return(data):
    X = data[['price']]
    y = data[['price1']]
    # 将数据划分为训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=125)
    clf = LinearRegression().fit(X_train, y_train)
    print('建立的LinearRegression模型为：\n', clf)

    # 预测训练集结果
    y_pred = clf.predict(X_test)
    print('预测前20个结果为：\n', y_pred[:20])
    rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(10, 6))  # 设定空白画布，并制定大小
    # 用不同的颜色表示不同数据
    plt.plot(range(y_test.shape[0]), y_test, color="blue", linewidth=1.5, linestyle="-")
    plt.plot(range(y_test.shape[0]), y_pred, color="red", linewidth=1.5, linestyle="-.")
    plt.legend(['真实值', '预测值'])
    plt.savefig('../回归模型可视化.png')
    plt.show()  # 显示图片
    # 评价
    print('线性回归模型的平均绝对误差为：', mean_absolute_error(y_test, y_pred))
    print('线性回归模型的均方误差为：', mean_squared_error(y_test, y_pred))
    print('线性回归模型的中值绝对误差为：', median_absolute_error(y_test, y_pred))
    print('线性回归模型的可解释方差值为：', explained_variance_score(y_test, y_pred))
    print('线性回归模型的R方值为：', r2_score(y_test, y_pred))


if __name__ == '__main__':
    data1 = pd.read_csv('F:/大三上/大作业/python数据分析/数据文件.csv', encoding='gbk')  # 导入数据
    data2 = pd.read_csv('F:/大三上/大作业/python数据分析/数据文件1.csv', encoding='gbk')
    data_processing(data1, data2)
