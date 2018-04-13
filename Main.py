
import numpy as np

data = open('kafka.txt', 'r').read()
chars = list(set(data))
seq_length = 25
learning_rate = 0.1
data_size, vocab_size = len(data), len(chars)
print('data has %d chars, %d unique' % (data_size, vocab_size))


output_size = vocab_size
num_hidden_units = 100


char_to_ix = { ch:i for i, ch in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}
print(char_to_ix)
print(ix_to_char)


ranomvar = 40

Uf = np.random.randn(output_size, vocab_size) * 0.01 # np.array([[0.2, -0.3], [0.5, 0.2]])
Wf = np.random.rand(output_size, output_size) * 0.01 # np.array([[0.1, -0.2], [0, 0.2]])

Uo = np.random.randn(output_size, vocab_size) * 0.01 #np.array([[0.1, 0.5], [0.2, -0.2]])
Wo = np.random.rand(output_size, output_size) * 0.01 #np.array([[0.6, 0.2], [-0.5, -0.1]])

Ua = np.random.randn(output_size, vocab_size) * 0.01 #np.array([[0.4, 0.2], [0.4, -0.2]])
Wa = np.random.rand(output_size, output_size) * 0.01 #np.array([[-0.1, 0.2], [0.5, -0.3]])

Ui = np.random.randn(output_size, vocab_size) * 0.01 #np.array([[0.3, -0.1], [0, 0.2]])
Wi = np.random.rand(output_size, output_size) * 0.01 #np.array([[0.2, 0.1], [0, 0.2]])

bf = np.zeros((output_size))
bo = np.zeros((output_size))
ba = np.zeros((output_size))
bi = np.zeros((output_size))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update(dWo, dUo, dbo, dWi, dUi, dbi, dWf, dUf, dbf, dWs, dUs, dbs):
    global Wf, Uf, Wo, Uo, Wa, Ua, Wi, Ui, bf, bo, ba, bi
    Wf -= dWf * learning_rate
    Uf -= dUf * learning_rate
    Wo -= dWo * learning_rate
    Uo -= dUo * learning_rate
    Wi -= dWi * learning_rate
    Ui -= dUi * learning_rate
    Wa -= dWs * learning_rate
    Ua -= dUs * learning_rate

    bo -= dbo * learning_rate
    bi -= dbi * learning_rate
    ba -= dbs * learning_rate
    bf -= dbf * learning_rate

def err_handler(type, flag):
    print("Floating point error (%s), with flag %s" % (type, flag))


saved_handler = np.seterrcall(err_handler)
save_err = np.seterr(all='call')


def loss_function(input, hp, sp, y):
    Gs = {}
    Gf = {}
    Go = {}
    Gi = {}
    Ga = {}
    x = {}
    h = {}

    h[-1] = hp
    Gs[-1] = sp
    Loss = 0
    y = np.asarray(y)
    for t in range(len(y)):
        x[t] = np.zeros((vocab_size))
        x[t][input[t]] += 1

        Gf[t] = sigmoid(bf + np.dot(Uf, x[t]) + np.dot(Wf, h[t - 1]))
        Gi[t] = sigmoid(bi + np.dot(Ui, x[t]) + np.dot(Wi, h[t - 1]))
        Ga[t] = np.tanh(ba + np.dot(Ua, x[t]) + np.dot(Wa, h[t - 1]))
        Go[t] = sigmoid(bo + np.dot(Uo, x[t]) + np.dot(Wo, h[t - 1]))
        Gs[t] = Gf[t] * Gs[t - 1] + Gi[t] * Ga[t]

        h[t] = np.tanh(Gs[t]) * Go[t]
        ca = np.exp(h[t]) / np.sum(np.exp(h[t]))
        Loss += -np.log(ca[y[t]])

    dWo, dUo, dbo = np.zeros_like(Wo), np.zeros_like(Uo), np.zeros_like(bo)
    dWa, dUa, dba = np.zeros_like(Wa), np.zeros_like(Ua), np.zeros_like(ba)
    dWf, dUf, dbf = np.zeros_like(Wf), np.zeros_like(Uf), np.zeros_like(bf)
    dWi, dUi, dbi = np.zeros_like(Wi), np.zeros_like(Ui), np.zeros_like(bi)


    deltaOut = np.zeros(output_size)
    deltaSFuture = np.zeros(output_size)

    for t in reversed(range(len(y))):
        L = np.copy(h[t])
        L[y[t]] -= 1

        dout = L + deltaOut
        dstate = dout * Go[t] * (1 - np.power(np.tanh(Gs[t]), 2)) + deltaSFuture
        df = dstate * Gs[t-1] * Gf[t] * (1 - Gf[t])
        di = dstate * Ga[t] * Gi[t] * (1 - Gi[t])
        da = dstate * Gi[t] * (1 - Ga[t]**2)
        do = dout * np.tanh(Gs[t]) * Go[t] * (1 - Go[t])

        deltaOut = np.dot(Wi.T, di) + np.dot(Wo.T, do) + np.dot(Wf.T, df) + np.dot(Wa.T, da)
        deltaSFuture = dstate * Gf[t]

        dUo += np.outer(do, x[t])
        dUi += np.outer(di, x[t])
        dUf += np.outer(df, x[t])
        dUa += np.outer(da, x[t])

        dWo += np.outer(do, h[t - 1])
        dWi += np.outer(di, h[t - 1])
        dWf += np.outer(df, h[t - 1])
        dWa += np.outer(da, h[t - 1])

        dbo += do
        dba += da
        dbf += df
        dbi += di

    return Loss, dWo, dUo, dbo, dWi, dUi, dbi, dWf, dUf, dbf, dWa, dUa, dba, h[len(y) - 1], Gs[len(y) - 1]



#prediction, one full forward passnp.atleast_2d(h[t]).T
def sample(inputs, hp, sp, n):
  Gs = {}
  Gf = {}
  Go = {}
  Gi = {}
  Ga = {}
  h = {}

  h[-1] = hp
  Gs[-1] = sp
  ixes = []
  x = np.zeros((vocab_size))
  x[inputs] = 1
  for t in range(n):
      Gf[t] = sigmoid(bf + np.dot(Uf, x) + np.dot(Wf, h[t - 1]))
      Gi[t] = sigmoid(bi + np.dot(Ui, x) + np.dot(Wi, h[t - 1]))
      Ga[t] = np.tanh(ba + np.dot(Ua, x) + np.dot(Wa, h[t - 1]))
      Gs[t] = Gf[t] * Gs[t - 1] + Gi[t] * Ga[t]
      Go[t] = sigmoid(bo + np.dot(Uo, x) + np.dot(Wo, h[t - 1]))
      h[t] = np.tanh(Gs[t]) * Go[t]
      p = np.exp(h[t]) / np.sum(np.exp(h[t]))
      ix = np.random.choice(range(vocab_size), p=p.ravel())
      x = np.zeros((vocab_size))
      x[ix] = 1
      ixes.append(ix)

  txt = ''.join(ix_to_char[ix] for ix in ixes)
  print('----\n %s \n----' % (txt, ))




def train():
    np.random.seed(0)
    n, p = 0, 0
    hprev = np.zeros((output_size, 1))

    smooth_loss = -np.log(
        1.0 / vocab_size) * seq_length  # loss at iteration 0
    while n <= 1000 * 100 * 3:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        # check "How to feed the loss function to see how this part works
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((output_size))  # reset RNN memory
            sprev = np.zeros((output_size))
            p = 0  # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

        # forward seq_length characters through the net and fetch gradient
        loss, dWo, dUo, dbo, dWi, dUi, dbi, dWf, dUf, dbf, dWs, dUs, dbs, hprev, sprev = loss_function(inputs, hprev, sprev, targets)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # sample from the model now and then
        if n % 1000 == 0:
            print('iter %d, loss: %f' % (n, loss))
            sample(inputs[0], hprev, sprev, 30)

        update(dWo, dUo, dbo, dWi, dUi, dbi, dWf, dUf, dbf, dWs, dUs, dbs)
        p += seq_length  # move data pointer
        n += 1  # iteration counter

train()