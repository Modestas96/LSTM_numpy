import numpy as np
import math

class LSTMCell:
    data = open('maironis.txt', 'r').read()
    chars = list(set(data))
    seq_length = 25
    learning_rate = 1e-1
    data_size, vocab_size = len(data), len(chars)
    print('data has %d chars, %d unique' % (data_size, vocab_size))

    output_size = vocab_size
    num_hidden_units = 100

    ranomvar = 40

    Ws = np.random.randn(vocab_size, num_hidden_units)
    Us = np.random.rand(vocab_size, num_hidden_units)

    Wf = np.random.randn(vocab_size, num_hidden_units)
    Uf = np.random.rand(vocab_size, num_hidden_units)

    Wo = np.random.randn(vocab_size, num_hidden_units)
    Uo = np.random.rand(vocab_size, num_hidden_units)

    Wi = np.random.randn(vocab_size, num_hidden_units)
    Ui = np.random.rand(vocab_size, num_hidden_units)

    Gs = {}
    Gf = {}
    Go = {}
    Gi = {}

    Cc = np.zeros(num_hidden_units)
    C = np.zeros(num_hidden_units)
    activition = np.zeros(num_hidden_units)

    bs = np.zeros((num_hidden_units, 1))
    bf = np.zeros((num_hidden_units, 1))
    bo = np.zeros((num_hidden_units, 1))
    bi = np.zeros((num_hidden_units, 1))

    h = {}

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def loss_function(self, x, hp, sp, y):
        self.h[-1] = hp
        self.Gs[-1] = sp
        Loss = 0
        for t in range(len(y)):
            self.Gf[t] = self.sigmoid(self.bf + np.dot(self.Uf, x[t]) + np.dot(self.Wf, self.h[t-1]))
            self.Gi[t] = self.sigmoid(self.bi + np.dot(self.Ui, x[t]) + np.dot(self.Wi, self.h[t-1]))
            self.Gs[t] = self.Gf[t] * self.Gs[t-1] + self.Gi[t] * self.sigmoid(self.bs + np.dot(self.Us, x[t]) + np.dot(self.Ws, self.h[t-1]))
            self.Go[t] = self.sigmoid(self.bo + np.dot(self.Uo, x[t]) + np.dot(self.Wo, self.h[t-1]))
            self.h[t] = np.tanh(self.Gs[t]) * self.Go
            Loss += -np.log(self.h[t][y[t], 0])

        dWo, dUo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.Uo), np.zeros_like(self.bo)
        dWs, dUs, dbs = np.zeros_like(self.Ws), np.zeros_like(self.Us), np.zeros_like(self.bs)
        dWf, dUf, dbf = np.zeros_like(self.Wf), np.zeros_like(self.Uf), np.zeros_like(self.bf)
        dWi, dUi, dbi = np.zeros_like(self.Wi), np.zeros_like(self.Ui), np.zeros_like(self.bi)

        for t in reversed(range(len(y))):
            dhst = (1-math.pow(np.tanh(self.Gs[t]), 2))
            L = (self.h[t] - y[t])
            doos = self.Go[t] * (1-self.Go[t])
            diis = self.Gi[t] * (1-self.Gi[t])
            dho = np.tanh(self.Gs[t])
            dsi = self.sigmoid(self.bs + np.dot(self.Us, x[t]) + np.dot(self.Ws, self.h[t-1]))
            dffs = self.Gf[t] * (1-self.Gf[t])
            sss = self.sigmoid(self.bs + np.dot(self.Us, x[t]) + np.dot(self.Ws, self.h[t-1]))
            dsss = sss * (1-sss)

            dWo += L * dho * doos * self.h[t-1]
            dUo += L * dho * doos * x
            dbo += L * dho * doos

            dWi += L * self.Go * dhst * dsi * diis * self.h[t-1]
            dUi += L * self.Go * dhst * dsi * diis * x
            dbi += L * self.Go * dhst * dsi * diis

            dWf += L * self.Go * dhst * self.Gs[t-1] * dffs * self.h[t-1]
            dUf += L * self.Go * dhst * self.Gs[t-1] * dffs * x
            dbf += L * self.Go * dhst * self.Gs[t-1] * dffs

            dWs += L * self.Go * dhst * self.Gi[t] * dsss * self.h[t-1]
            dUs += L * self.Go * dhst * self.Gi[t] * dsss * x
            dbs += L * self.Go * dhst * self.Gi[t] * dsss

        return dWo, dUo, dbo, dWi, dUi, dbi, dWf, dUf, dbf, dWs, dUs, dbs











