import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

# 每类随机生成点的个数。
points_sum = 100

c1_x = []
c1_y = []
c2_x = []
c2_y = []

# 分别在(0, 0)点附近和(8, 8)点附近生成2类随机数据。
for _ in range(points_sum):
    c1_x.append(np.random.normal(0, 2))
    c1_y.append(np.random.normal(0, 2))
    c2_x.append(np.random.normal(8, 2))
    c2_y.append(np.random.normal(8, 2))

data = np.array([c1_x+c2_x,c1_y+c2_y]).transpose()
#print(data.shape)
classification = pd.utils.generate_label_matrix([0]*points_sum+[1]*points_sum)[0]
#print(classification.shape)
model = pd.nn.Network()
model.add(pd.nn.Dense(2,input_dimension=2),name='dense')
model.loss('svm')

model.optimizer('gradient descent', rate=0.01, consistent=True)
model.add_plugin('variable_print', pd.nn.VariableMonitorPlugin('dense'))
model.regularization('l2', 0.01)  # 使用L2正则化。
model.train(data, classification, epochs=1000)

# 获取W和B的训练结果。
W,B = model.get_layer('dense').variables()
w_data = W.get_value().transpose()
b_data = B.get_value().transpose()
print(w_data)
print(b_data)

# 计算分类直线的斜率和截距。
k = (w_data[1, 0] - w_data[0, 0]) / (w_data[0, 1] - w_data[1, 1])
b = (b_data[1, 0] - b_data[0, 0]) / (w_data[0, 1] - w_data[1, 1])

# 分类面的端点。
x_range = np.array([np.min(c1_x), np.max(c2_x)])

# 绘制图像。
plt.title('Paradox implement Linear SVM')
plt.plot(c1_x, c1_y, 'ro', label='Category 1')
plt.plot(c2_x, c2_y, 'bo', label='Category 2')
plt.plot(x_range, k * x_range + b, 'y', label='SVM')
plt.legend()
plt.show()
