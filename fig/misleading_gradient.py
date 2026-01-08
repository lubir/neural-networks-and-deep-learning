"""
misleading_gradient
~~~~~~~~~~~~~~~~~~~

绘制一个会“误导”梯度下降的函数。
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
X = numpy.arange(-1, 1, 0.025)
Y = numpy.arange(-1, 1, 0.025)
X, Y = numpy.meshgrid(X, Y)
Z = X**2 + 10*Y**2

colortuple = ('w', 'b')
colors = numpy.empty(X.shape, dtype=str)
for x in range(len(X)):
    for y in range(len(Y)):
        colors[x, y] = colortuple[(x + y) % 2]

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0)

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 12)
ax.w_xaxis.set_major_locator(LinearLocator(3))
ax.w_yaxis.set_major_locator(LinearLocator(3))
ax.w_zaxis.set_major_locator(LinearLocator(3))
ax.text(0.05, -1.8, 0, "$w_1$", fontsize=20)
ax.text(1.5, -0.25, 0, "$w_2$", fontsize=20)
ax.text(1.79, 0, 9.62, "$C$", fontsize=20)

plt.show()

