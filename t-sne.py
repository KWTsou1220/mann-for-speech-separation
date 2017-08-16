import numpy as np
from sklearn.manifold import TSNE

'''
z is the output of encoder
z = model.encoding(sess, data)
'''

tsne = TSNE(n_components = 2, random_state = 0)
t_z = tsne.fit_transform(z)


'''
plot the t_z, color is determined by label
'''
colors = cm.rainbow(np.linspace(0, 1, 10))
scatter = []
index = xrange(10)
for i in range(10):
    tmp = np.where(indexes == i)
    scatter.append(plt.scatter(t_z[tmp, 0], t_z[tmp, 1], c = colors[i] ,s = 5))

plt.legend(scatter, index)
plt.show()