import paradox as pd

A = pd.Constant([[1, 2], [1, 3]], name='A')
x = pd.Variable([0, 0], name='x')
b = pd.Constant([3, 4], name='b')

y = pd.reduce_sum((A @ x - b) ** 2) / 2

print('value =\n{}\n'.format(y.get_value())) # 完善自动求值
print('x gradient =\n{}\n'.format(y.get_gradient(x))) # 完善自动求导
