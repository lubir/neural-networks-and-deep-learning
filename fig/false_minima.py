"""
false_minimum
~~~~~~~~~~~~~

绘制一个含有多个假极小值的二维函数。
"""

#### 依赖库
# 第三方库
from matplotlib.ticker import LinearLocator
# 注意：axes3d 在代码中未直接使用，但用于正确注册 3D 绘图类型
from mpl_toolkits.mplot3d import axes3d 
import matplotlib.pyplot as plt
import numpy

fig = plt.figure()
ax = fig.gca(projection='3d')
X = numpy.arange(-5, 5, 0.1)
Y = numpy.arange(-5, 5, 0.1)
X, Y = numpy.meshgrid(X, Y)
Z = numpy.sin(X)*numpy.sin(Y)+0.2*X

colortuple = ('w', 'b')
colors = numpy.empty(X.shape, dtype=str)
for x in range(len(X)):
    for y in range(len(Y)):
        colors[x, y] = colortuple[(x + y) % 2]

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0)

ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5, 5)
ax.set_zlim3d(-2, 2)
ax.w_xaxis.set_major_locator(LinearLocator(3))
ax.w_yaxis.set_major_locator(LinearLocator(3))
ax.w_zaxis.set_major_locator(LinearLocator(3))

plt.show()

