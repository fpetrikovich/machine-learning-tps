import matplotlib.pyplot as plt
import numpy as np
import exercise2
# Running with: python test_ex2_3d.py

def get_color_range(color, std):
    rs, gs, bs = [], [], []
    for r in range(max(0, int(color[0] - std[0])), min(255, int(color[0] + std[0])), 7):
        for g in range(max(0, int(color[1] - std[1])), min(255, int(color[1] + std[1])), 7):
            for b in range(max(0, int(color[2] - std[2])), min(255, int(color[2] + std[2])), 7):
                rs.append(r)
                gs.append(g)
                bs.append(b)
    return np.array(rs), np.array(gs), np.array(bs)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

train_cielo, train_pasto, train_vaca = exercise2.load_train_images('./pics')
dataset, labels = exercise2.build_image_dataset([train_cielo, train_pasto, train_vaca], [0, 1, 2])
dataset, labels = exercise2.shuffle_dataset(dataset, labels)
train_dataset, train_labels, test_dataset, test_labels = exercise2.split_dataset(dataset, labels, percentage=.1)

cielo_r, cielo_g, cielo_b, pasto_r, pasto_g, pasto_b, vaca_r, vaca_g, vaca_b = [], [], [], [], [], [], [], [], []
for i in range(test_dataset.shape[0]):
    if test_labels[i] == 0:
        cielo_r.append(test_dataset[i][0])
        cielo_g.append(test_dataset[i][1])
        cielo_b.append(test_dataset[i][2])
    elif test_labels[i] == 1:
        pasto_r.append(test_dataset[i][0])
        pasto_g.append(test_dataset[i][1])
        pasto_b.append(test_dataset[i][2])
    elif test_labels[i] == 2:
        vaca_r.append(test_dataset[i][0])
        vaca_g.append(test_dataset[i][1])
        vaca_b.append(test_dataset[i][2])

ax.scatter(cielo_r, cielo_g, cielo_b, marker='o', c="b")
ax.scatter(pasto_r, pasto_g, pasto_b, marker='s', c="g")
ax.scatter(vaca_r, vaca_g, vaca_b, marker='P', c="black")

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()