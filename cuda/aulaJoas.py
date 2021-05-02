from numba import cuda
import numpy
from numpy import *


@cuda.jit
def my_kernel(matriz, matriz2):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x

    # Compute flattened index inside the array
    pos = tx + ty * bw

    i = 0
    count = 0

    for i in range(len(matriz)):

        numeros = matriz[i]

        for n in matriz:
            if (numeros == n):
                count = count + 1

        matriz2[numeros - 1] = count

        count = 0

        i += 1

    # vetor com números


data = random.randint(1, 11, size=(100))
print(data)
# print(f"dataMatriz {data}")
o_array = numpy.empty([10])
# números de threads por bloco
threads_per_block = 32

# número de blocos por grid
blocks_per_grid = (len(data) + (threads_per_block - 1))

# iniciando o kernel
my_kernel[blocks_per_grid, threads_per_block](data, o_array)

# mostra o resultado
for i in range(len(o_array)):
    print(f"O número {i + 1} apareceu {o_array[i]} vezes!")



