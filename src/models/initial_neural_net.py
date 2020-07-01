from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

class InitialNN:
    @staticmethod
    def build(lb):
        model = Sequential()
        model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
        model.add(Dense(512, activation="sigmoid"))
        model.add(Dense(len(lb.classes_), activation="softmax"))

        return model