import numpy as np

array = np.array([[1, 2, 3], [4, 5, 6]])
array.ndim
array.shape
array.size

a = np.array([1, 2, 3], dtpye = np.float32)
b = np.zeros(3, 4)
c = np.ones(3, 4)
d = np.arange(12).reshape((3, 4))
e = np.random.random((2, 4))
f = np.sum(a)
g = np.max(a)
h = np.min(a)

A = np.arange(1, 13).reshape((3, 4))
argmin = np.argmin(A)
argmax = np.argmax(A)
np.mean(A)
np.average(A)
A.mean()
A.averange(A)
A.median()
np.cunsum(A)
np.diff(A)

B = arange(13, 1, -1).reshape((3, 4))
np.sorf(B)
np.transpose(B)
B.T
np.clip(B, 5, 9)

C = np.arange(3, 15)
C[3]
D = np.arange(3, 15).reshape(3, 4)
D[3]
D[1][1]
D[1, 1]
D[1, 1:3]
for row in D:
	print(row)
for row in D.T:
	print(row)
D.flatten()
for item in A.flat:
	print(item)

E = np.array([1, 1, 1])
F = np.array([2, 2, 2])
np.vstack(E, F)
np.hstack(E, F)
F[:;np.newaxis]
F[np.newaxis;:]
G = np.concatenate((E,F), axis=0)
G = np.concatenate((E,F), axis=1)
np.split(F,2,axis=1)
np.split(F,2,axis=0)
