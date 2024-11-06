# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

from matplotlib import pyplot as plt
import numpy as np

point_color = '#F950D3'
line_color = 'white'
number_color = 'white'
point_size = 1000

lr = 0.1

X, Y = np.linspace(-20, 20, 1000), np.linspace(-20, 20, 1000)
X, Y = np.meshgrid(X, Y)
Z = 6 * X ** 2 + 4 * Y ** 2 - 4 * X * Y
levels = [3000, 2000, 1000, 300, 100, 10, 1, 0.1][::-1]
cp = plt.contour(X, Y, Z, levels, linestyles='dashed', colors='black', alpha=0.9)
plt.contourf(X, Y, Z, levels, cmap='gist_earth', alpha=0.9, extend='both')
plt.clabel(cp, inline=1, fontsize=10, colors=f'{number_color}')
plt.gca().text(0.45, 0.8, f'$LR={lr}$', c='white', size=15, transform=plt.gca().transAxes)  # "transform" allows to place on plot as percentage

epochs = 5

# w = np.array([-15., 0])
# points = [w.copy()]
# for i in range(epochs):
#     plt.scatter(w[0], w[1], point_size/(2*i+1), c=point_color)
#     # plt.gca().text(w[0], w[1], f'${str(i)}$', color='white')
#     g = np.array([12 * w[0] - 4 * w[1], 8 * w[1] - 4 * w[0]])
#     w -= lr * g
#     points.append(w.copy())
#
# for i in range(len(points) - 1):
#     plt.annotate('', xy=points[i + 1], xytext=points[i],
#                  arrowprops={'arrowstyle': '-|>', 'color': f'{line_color}', 'lw': 1}, va='center', ha='center')
# f = 6 * w[0] ** 2 + 4 * w[1] ** 2 - 4 * w[0] * w[1]
# print(f)
# plt.gca().text(w[0], w[1], '$*$', size=20, color='red')
# plt.scatter(w[0], w[1], point_size/(2*epochs+2), c=f'{point_color}')


def dist(g1, g2):
    e = 1e-05
    d1, d2 = np.linalg.norm(g1), np.linalg.norm(g2)
    dists = np.array([d2, d1])
    return dists/(dists.sum()+e)


w = np.array([-15., 0])
LR = np.array([lr, lr])
points = [w.copy()]
s = 1
for i in range(epochs):
    plt.scatter(w[0], w[1], point_size/(2*i+1), c=f'{point_color}')
    # plt.gca().text(w[0], w[1], str(i))
    g = np.array([12 * w[0] - 4 * w[1], 8 * w[1] - 4 * w[0]])
    s = abs(g) + s*0.999
    oracle = w - lr * g
    g_orc = np.array([12 * oracle[0] - 4 * oracle[1], 8 * oracle[1] - 4 * oracle[0]])
    if g @ g_orc < 0:  # (784)
        d1, d2 = dist(g, g_orc)
        if d1 > 0.7:
            lr /= s
        w = (w*d1 + oracle*d2)
    else:
        w = oracle - g_orc * lr
    points.append(w.copy())

for i in range(len(points) - 1):
    plt.annotate('', xy=points[i + 1], xytext=points[i],
                 arrowprops={'arrowstyle': '->', 'color': f'{line_color}', 'lw': 1},
                 va='center', ha='center')
f = 6 * w[0] ** 2 + 4 * w[1] ** 2 - 4 * w[0] * w[1]
print(f)
plt.gca().text(w[0], w[1], '$*$', size=20, color='red')
plt.scatter(w[0], w[1], point_size/(2*epochs+2), c=f'{point_color}')

plt.xlabel('$w_1$', size=15)
plt.ylabel('$w_2$', size=15)
plt.tight_layout()
plt.show()
