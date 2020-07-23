import numpy


arr1 = numpy.random.randint(5, size=(5,5))
print(arr1)

print()

d3 = numpy.random.rand(arr1.shape[0], arr1.shape[1])
print(d3)
print()
for i in range(arr1.shape[0]):
    for j in range(arr1.shape[1]):
        if (d3[i][j] > 0.8):
            d3[i][j] = 0

print(d3)
print()

arr1 = numpy.multiply(arr1, d3)
arr1 = arr1/d3
print(arr1)