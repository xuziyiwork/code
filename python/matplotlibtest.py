import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x + 1
y2 = x**2


plt.figure(figsize=(8,5))
plt.plot(x, y1, label='y1')
plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', lable='y2')
plt.title('y-x')
plt.xlim(())
plt.ylim(())
plt.yticks([],[r'$ $'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.scatter(x, y)
plt.subplot(x, y, n)
