import numpy as np
import matplotlib.pyplot as plt
import time

x = np.array([0.247, 0.321, 0.393, 0.463, 0.564, 0.633, 0.722, 0.841, 
          1.008, 1.198, 1.478, 1.931, 2.508, 3.579, 5.008, 8.341, 
          12.51, 16.68, 25.01, 33.34, 50.01, 100.01])
F = np.array([9.56e-3, 9.264e-3, 8.970e-3, 8.689e-3, 8.293e-3, 8.037e-3, 7.713e-3, 
        7.286e-3, 6.756e-3, 6.199e-3, 5.520e-3, 4.630e-3, 3.793e-3, 2.760e-3, 1.934e-3,
        1.008e-3, 5.487e-4, 3.551e-4, 1.792e-4, 1.084e-4, 6.139e-5, 2.339e-5])

point = 1000000
eta_array = np.random.random(point)
Fn=F/F[0]
hp = np.array([])
hit = 0
start_time = time.time()
dtime = point/10
for j, eta in enumerate(eta_array):
    if j % dtime == 0 : print("{:d}, elapsed: {:.1f}".format(j, time.time() - start_time))
    for i in range(1, len(Fn)):
        if Fn[i-1]>eta>=Fn[i]:
            hit+=1
            #print (Fn[i-1], eta, Fn[i])
            ci = (1/(x[i]-x[i-1]))
            hp=np.append(hp, x[i-1] + (eta - F[i-1])/ci)
print("point {:d}, hit {:d}, miss: {:d} ".format(point, hit, point-hit))