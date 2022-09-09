import sys, os
sys.path.append(os.pardir)
import numpy as np

from common.functions import *
from common.gradient import numerical_gradient

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
        self.params = {} # ニューラルネットワークのパラメーター
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 1層目の重み
        self.params['b1'] = np.zeros(hidden_size) # 1層目のバイアス
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 2層目の重み
        self.params['b2'] = np.zeros(output_size) # 2層目のバイアス

    def predict(self, x):
        """
        認識(推論)を行う
        Parameters
        ----------
        x:
            画像データ  
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

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
        return cross_entropy_error(y, t)

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
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        
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

        grads = {} # 勾配
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 1層目の重みの勾配
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) # 1層目のバイアスの勾配
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 2層目の重みの勾配
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) # 2層目のバイアスの勾配

        return grads