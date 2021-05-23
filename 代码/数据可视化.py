'''
班级：大数据1801
姓名：叶际荣
学号: 201806140014
'''
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Bar, Line

plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

data1 = pd.read_csv('F:/大三上/大作业/python数据分析/数据文件.csv', encoding='gbk')  # 导入数据
data2 = pd.read_csv('F:/大三上/大作业/python数据分析/数据文件1.csv', encoding='gbk')
data = pd.concat(([data1, data2]))  # 对数据表拼接
print('价格大于于100的数量', sum(data['price'] > 100))
print('价格小于100的数量', sum(data['price'] < 100))

# 书本价格区间直方图
price = data['price']
price1 = []
price2 = []
price3 = []
price4 = []
price5 = []
price6 = []
price7 = []
price8 = []
price9 = []
price10 = []
price11 = []
for p in price:
    if p < 10:
        price1.append(p)
    elif 10 <= p < 20:
        price2.append(p)
    elif 20 <= p < 30:
        price3.append(p)
    elif 30 <= p < 40:
        price4.append(p)
    elif 40 <= p < 50:
        price5.append(p)
    elif 50 <= p < 60:
        price6.append(p)
    elif 60 <= p < 70:
        price7.append(p)
    elif 70 <= p < 80:
        price8.append(p)
    elif 80 <= p < 90:
        price9.append(p)
    elif 90 <= p < 100:
        price10.append(p)
    else:
        price11.append(p)
all_price = ['10以下', '10-20', '20-30', '30-40', '40-50',
             '50-60', '60-70', '70-80', '80-90', '90-100', '100以上']
p_values = [len(price1), len(price2), len(price3), len(price4), len(price5),
            len(price6), len(price7), len(price8), len(price9), len(price10), len(price11)]
plt.figure(figsize=(8, 7))
plt.bar(all_price, p_values, width=0.75, color='red')
plt.xlabel('价格（元）')
plt.ylabel('书本数量')
plt.savefig('../书本价格区间直方图.png')
plt.title('书本价格区间直方图')
data.columns
plt.show()
# groupby_data = data.groupby(data['pack'])
# price_mean = groupby_data['price'].mean()
# plt.plot(price_mean, linewidth=2.5)
# plt.show()

# 书本价格区间散点图
plt.figure(figsize=(8, 7))
plt.scatter(all_price, p_values, color='red', marker='o')
plt.xlabel('价格（元）')
plt.ylabel('书本数量')
plt.savefig('../书本价格区间散点图.png')
plt.title('书本价格区间散点图')
plt.show()

#  书本包装折线图
# a = data['pack'].value_counts()
pack = data['pack']
pack1 = []
pack2 = []
pack3 = []
pack4 = []
pack5 = []
pack6 = []
pack7 = []
for p in pack:
    if p == '简裝本':
        pack2.append(p)
    elif p == '简裝':
        pack2.append(p)
    elif p == '平装-胶订':
        pack3.append(p)
    elif p == '平装胶订':
        pack3.append(p)
    elif p == '平装':
        pack4.append(p)
    elif p == '精装':
        pack5.append(p)
    elif p == '精装本':
        pack5.append(p)
    else:
        pack1.append(p)
all_pack = ['其他', '简裝本', '平装胶订', '平装', '精装']
p_pack = [len(pack1), len(pack2), len(pack3), len(pack4), len(pack5)]
plt.figure(figsize=(8, 7))
plt.plot(all_pack, p_pack, color='red')
plt.xlabel('包装样式')
plt.ylabel('书本数量')
plt.savefig('../书本包装折线图.png')
plt.title('书本包装折线图')
plt.show()
# 查看各出版社书籍的数量
b = data['publisher'].value_counts()
print('查看各出版社书籍的数量:\n', b)

# pyecharts
# 书本价格区间直方图
bar = Bar()
bar.add_xaxis(all_price)
bar.add_yaxis('书本数量', p_values)
bar.render('../书本价格区间直方图.html')


# 书本包装折线图
c = Line()
c.add_xaxis(all_pack)
c.add_yaxis('书本数量', p_pack)
c.render('../书本包装折线图.html')
