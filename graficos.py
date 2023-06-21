from matplotlib import pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 30})

file_CPU = np.loadtxt("CPU/tiempos.txt", skiprows = 1)
file_GPU = np.loadtxt("GPU/tiempos.txt", skiprows = 1)

Q_CPU = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
Q_GPU = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000]


data_CPU = {
    10 : np.mean(file_CPU[:, 0]),
    30 : np.mean(file_CPU[:, 1]),
    100 : np.mean(file_CPU[:, 2]),
    300 : np.mean(file_CPU[:, 3]),
    1000 : np.mean(file_CPU[:, 4]),
    3000 : np.mean(file_CPU[:, 5]),
    10000 : np.mean(file_CPU[:, 6]),
    30000 : np.mean(file_CPU[:, 7]),
    100000 : np.mean(file_CPU[:, 8])
}

data_GPU = {
    10 : np.mean(file_GPU[:, 0]),
    30 : np.mean(file_GPU[:, 1]),
    100 : np.mean(file_GPU[:, 2]),
    300 : np.mean(file_GPU[:, 3]),
    1000 : np.mean(file_GPU[:, 4]),
    3000 : np.mean(file_GPU[:, 5]),
    10000 : np.mean(file_GPU[:, 6]),
    30000 : np.mean(file_GPU[:, 7]),
    100000 : np.mean(file_GPU[:, 8]),
    300000 : np.mean(file_GPU[:, 9]),
}



plt.plot(Q_CPU, [data_CPU[v] for v in Q_CPU], 'ko-' , label = "CPU") 
plt.plot(Q_GPU, [data_GPU[v] for v in Q_GPU], 'ro-' , label = "GPU")

plt.xlabel(r"$Q$")
plt.ylabel(r"$t[ms]$")
plt.legend(loc = 'best')

plt.yscale('log')
plt.xscale('log')

plt.show()

