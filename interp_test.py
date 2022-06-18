# from scipy.interpolate import interpn
# import numpy as np
# import matplotlib.pyplot as plt

# def value_func_3d(x, y, z):

#     return x*y*z

# x = np.linspace(0, 4, 5)
# y = np.linspace(0, 5, 5)
# z = np.linspace(0, 6, 5)

# points = (x, y, z)

# M = np.meshgrid(*points, indexing='ij')
# values = value_func_3d(*M)

# point = np.array([2.21, 3.12, 1.15])


# xi = np.linspace(1, 2, 5)
# yi = np.linspace(1, 3, 5)
# zi = np.linspace(1, 4, 5)

# Mi = np.meshgrid(xi,yi,zi)
# i_vals = np.array(Mi)

# print(i_vals.shape)
# print(i_vals.T.shape)

# # i_pts = np.rollaxis(i_vals, 0, 3).reshape(5,5,5,3)
# i_pts = np.rollaxis(i_vals, 0, 3).reshape(3,5,5,5)

# # i_3d = interpn(points, values, i_pts) 
# i_3d = interpn(points, values, i_vals.T) 
# print(i_3d)

# print(interpn(points, values, point))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # surf = ax.scatter(xs = M[0], ys = M[1], zs = M[2], c=values, alpha=0.25)
# # surf = ax.scatter(xs = Mi[0], ys = Mi[1], zs = Mi[2], c=i_3d, alpha=0.5)

# zs = np.concatenate([values, i_3d], axis=0)
# min_, max_ = zs.min(), zs.max()

# # ax.scatter(xs = M[0], ys = M[1], zs = M[2], c=values, alpha=0.25,cmap='viridis_r',marker='s')
# # ax.clim(min_, max_)
# # ax.scatter(xs = Mi[0], ys = Mi[1], zs = Mi[2], c=i_3d, alpha=0.5,cmap='viridis_r',marker='o')
# # ax.clim(min_, max_)

# ax.scatter(xs = M[0], ys = M[1], zs = M[2], c=values, alpha=0.25,cmap='viridis_r',marker='s',vmin=min_,vmax=max_)
# surf = ax.scatter(xs = Mi[0], ys = Mi[1], zs = Mi[2], c=i_3d, alpha=0.5,cmap='viridis_r',marker='o',vmin=min_,vmax=max_)

# fig.colorbar(surf).set_label('gl',rotation=270)

# # fig.colorbar(surf, shrink=0.5, aspect=5)



# # Plot the result
# # xx, yy = np.meshgrid(self.interp_vals[0], self.interp_vals[1])
# # ax.scatter(xs = xx, ys = yy, zs = zz, c= interp_arr[0,:], s=20)
# # ax.scatter(xs = xx, ys = yy, c= self.interp_arr[4,:], s=20)
# # ax.scatter(i_pts[1][0] * np.ones(i_pts[2].shape), i_pts[2], c=i_3d, s=20)
        
# ax.set_xlabel('v1')
# ax.set_ylabel('v2')
# ax.set_zlabel('w1')
# plt.show()

import numpy as np
from itertools import compress


l = [[1],[1,2],[1,2],[1],[1],[1],[1],[1]]
i = [len(l)>1]


mask = np.zeros((len(l)))
for idx, dim in enumerate(l):
    if len(dim) != 1:
        mask[idx] = 1

mask = list(map(bool,mask))
o = list(compress(l, mask))

print(i)
print(len(l)>1)
print(mask)
print(o)