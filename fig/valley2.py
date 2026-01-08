"""valley2.py
~~~~~~~~~~~~~

绘制一个待最小化的二维函数（典型的“山谷”形函数）。

这是 valley.py 的重复版本，但去掉了坐标轴标签。
虽然重复代码并不理想，但我在 matplotlib 上为添加/移除标签
遇到了较大麻烦，因此用这种折中方式解决。
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
X = numpy.arange(-1, 1, 0.1)
Y = numpy.arange(-1, 1, 0.1)
X, Y = numpy.meshgrid(X, Y)
Z = X**2 + Y**2

colortuple = ('w', 'b')
colors = numpy.empty(X.shape, dtype=str)
for x in range(len(X)):
    for y in range(len(Y)):
        colors[x, y] = colortuple[(x + y) % 2]

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0)

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 2)
ax.w_xaxis.set_major_locator(LinearLocator(3))
ax.w_yaxis.set_major_locator(LinearLocator(3))
ax.w_zaxis.set_major_locator(LinearLocator(3))
ax.text(1.79, 0, 1.62, "$C$", fontsize=20)

plt.show()
