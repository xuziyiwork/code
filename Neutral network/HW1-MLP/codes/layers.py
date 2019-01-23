import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        self.result = input
        self.result[input < 0] = 0
        return self.result

    def backward(self, grad_output):
        '''Your codes here'''
        relu_grad = self.result
        relu_grad[self.result > 0] = 1
        return grad_output * relu_grad


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        self.result = 1 / (1 + np.exp(-input))

    def backward(self, grad_output):
        '''Your codes here'''
        return grad_output * self.result * (1 - self.result)
    

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        self.in_data = input
        self.out_data = np.dot(input, self.W) + self.b
        return self.out_data

    def backward(self, grad_output):
        '''Your codes here'''
        data_grad = np.dot(grad_output, self.W.T)
        self.grad_W = np.dot(np.ones((100, self.in_num)).T, grad_output)
        self.grad_b = np.sum(grad_output,axis=0)/100
        return data_grad

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
