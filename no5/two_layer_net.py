import sys, os

from no5.layer import SoftmaxWithLoss
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from layer import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初期化を行う

        Parameters
        ----------
        input_size: 
            入力層のニューロンの数
        hidden_size:
            隠れ層のニューロンの数
        output_size:
            出力層のニューロンの数
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 1層目の重み
        self.params['b1'] = np.zeros(hidden_size) # 1層目のバイアス
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 2層目の重み
        self.params['b2'] = np.zeros(output_size) # 2層目のバイアス        

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        認識(推論)を行う
        Parameters
        ----------
        x:
            画像データ  
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        損失関数の値を求める
        Parameters
        ----------
        x:
            画像データ  
        t:
            正解ラベル
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        認識精度を求める
        Parameters
        ----------
        x:
            画像データ  
        t:
            正解ラベル
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        重みパラメーターに対する勾配を求める
        Parameters
        ----------
        x:
            画像データ  
        t:
            正解ラベル
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads