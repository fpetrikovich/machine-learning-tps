import matplotlib.pyplot as plt
import numpy as np

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

cielo_r, cielo_g, cielo_b = get_color_range([190.52270613, 202.58514447, 209.04143763], [13.32043258, 7.17914836, 4.41458626])
pasto_r, pasto_g, pasto_b = get_color_range([92.5472028, 118.29623155, 90.21595765], [32.48662022, 29.77305296, 20.58161407])
vaca_r, vaca_g, vaca_b = get_color_range([164.08314986, 141.66010298, 129.20699574], [77.1473218, 73.54467836, 65.99261118])

ax.scatter(cielo_r, cielo_g, cielo_b, marker='o', c="b")
ax.scatter(pasto_r, pasto_g, pasto_b, marker='s', c="g")
ax.scatter(vaca_r, vaca_g, vaca_b, marker='P', c="black")

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()