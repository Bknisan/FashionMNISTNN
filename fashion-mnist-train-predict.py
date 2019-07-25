import numpy as np


def predict(test, parameters):
    MEM = forward_prop(test.T, parameters)
    predictions = np.argmax(MEM["probabilities"], axis=0)
    return predictions


def normalize(x):
    x_normed = x / x.max(axis=0)
    return x_normed



def update_parameters(parameters, gradients, eta):
    parameters["W1"] = ((1 - (0.001 * eta)) * parameters["W1"]) - eta * gradients["dW1"]
    parameters["B1"] = ((1 - (0.001 * eta)) * parameters["B1"]) - eta * gradients["db1"]
    parameters["W2"] = ((1 - (0.001 * eta)) * parameters["W2"]) - eta * gradients["dW2"]
    parameters["B2"] = ((1 - (0.001 * eta)) * parameters["B2"]) - eta * gradients["db2"]
    return parameters


def log_min(vec, labels):
    length = labels.shape[1]
    logarithmic_probs = np.multiply(np.log(vec), labels)
    loss = - np.sum(logarithmic_probs) / length
    loss = np.squeeze(loss)
    return loss


def forward_prop(x_train, parameters):
    A1 = np.dot(parameters["W1"], x_train) + parameters["B1"]
    Activation = normalize(re_lu(A1))
    A2 = np.dot(parameters["W2"], Activation) + parameters["B2"]
    Activation2 = re_lu(A2)
    probabilities_vector = soft_max(Activation2.T)
    MEM = {"A1": A1, "Activation": Activation, "A2": A2, "Activation2": Activation2,
           "probabilities": probabilities_vector}
    return MEM


def back_prop(parameters, mem, x_train, y_train):
    length = len(y_train)
    derivative2 = mem["probabilities"] - np.array(y_train).T
    derivativeW2 = (1 / length) * np.dot(deravative2, (mem["Activation"]).T)
    derivativeB2 = (1 / length) * np.sum(deravative2, axis=1, keepdims=True)
    derivative1 = np.multiply(np.dot((parameters["W2"]).T, deravative2), 1 - np.square(mem["Activation"]))
    derivativeW1 = (1 / length) * np.dot(deravative1, x_train)
    derivativeB1 = (1 / length) * np.sum(deravative1, axis=1, keepdims=True)
    gradients = {"dW1": deravativeW1,
                 "db1": deravativeB1,
                 "dW2": deravativeW2,
                 "db2": deravativeB2}
    return gradients


def re_lu(x):
    return x * (x > 0)


def soft_max(x):
    x -= np.max(x)
    sm = (np.exp(x).T / np.sum(np.exp(x), axis=1))
    return sm


def translate(translator, data):
    translated = []
    for x in data:
        translated.append(translator[x])
    return translated


def main():
    dictionary = {
        0: 'T - shirt / top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankleboot'
    }
    translator = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    # random parameters.
    np.random.seed(2)
    Weights1 = np.random.uniform(-1, 1, (300, 784))
    Bias1 = np.random.uniform(-1, 1, (300, 1))
    Weights2 = np.random.uniform(-1, 1, (10, 300))
    Bias2 = np.random.uniform(-1, 1, (10, 1))
    Parameters_Dictionary = {"W1": Weights1, "W2": Weights2, "B1": Bias1, "B2": Bias2}

    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    train_y_vector = translate(translator, train_y)
    # normalize brightness.
    train_x /= 255
    test_x /= 255
    eta = 0.5
    # try for 1200 iterations.
    for iter in range(1200):
        print(iter)
        if iter % 300 == 0:
            eta *= 0.95
        mem = forward_prop(train_x.T, Parameters_Dictionary)
        gradients = back_prop(Parameters_Dictionary, mem, train_x, train_y_vector)
        Parameters_Dictionary = update_parameters(Parameters_Dictionary, gradients, eta)
    predictions = predict(test_x, Parameters_Dictionary)
    i = 0
    fs = open("test_y", 'w+')
    # write just the first 5k predictions.
    for prediction in predictions:
        if i == 5000:
            break
        fs.write(str(prediction))
        fs.write('\n')
        i += 1
    fs.close()


# goto main()
if __name__ == "__main__":
    main()
