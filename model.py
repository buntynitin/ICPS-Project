import pickle
from keras.models import load_model


def NeuralNetClassifier(arr):
    classifier = load_model('NN4_model.h5')
    prediction = classifier.predict(arr)
    if prediction[0][0] == 0:
        return 'healthy'
    else:
        return 'diabetic'

def DecisionTreeClassifier(arr):
    pickled_model = pickle.load(open('dtc.sav', 'rb'))
    prediction = pickled_model.predict(arr)
    return prediction[0]

def KNNClassifier(arr):
    pickled_model = pickle.load(open('knn.sav', 'rb'))
    prediction = pickled_model.predict(arr)
    return prediction[0]
