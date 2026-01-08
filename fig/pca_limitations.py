"""
pca_limitations
~~~~~~~~~~~~~~~

绘制图表说明 PCA 的局限性。
"""

# 第三方库
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# 仅绘制数据点
fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(-2, 2, 20)
theta = np.linspace(-4 * np.pi, 4 * np.pi, 20)
x = np.sin(theta)+0.03*np.random.randn(20)
y = np.cos(theta)+0.03*np.random.randn(20)
ax.plot(x, y, z, 'ro')
plt.show()

# 同时绘制数据与螺旋线
fig = plt.figure()
ax = fig.gca(projection='3d')
z_helix = np.linspace(-2, 2, 100)
theta_helix = np.linspace(-4 * np.pi, 4 * np.pi, 100)
x_helix = np.sin(theta_helix)
y_helix = np.cos(theta_helix)
ax.plot(x, y, z, 'ro')
ax.plot(x_helix, y_helix, z_helix, '')
plt.show()
