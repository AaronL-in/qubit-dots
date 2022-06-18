import numpy as np
import matplotlib.pyplot as plt

mesh = np.array(np.meshgrid(np.linspace(0,1,5),np.linspace(1,1,5),np.linspace(-1,0,5),np.linspace(100,500,5)))[0,:,:,:,:]

data = mesh[:,0,:,:]

data = [[1,1,1],[2,2,2],[3,3,3]]
arr = np.array([[1,1,1],[2,2,2],[3,3,3]])

ind = (0,0,0)


# print(mesh)
# print(data)
l = [[1],[2]]
# l = (0,0,0)
new = arr[np.ix_(*l)]
print(new)
# print(data[l])














