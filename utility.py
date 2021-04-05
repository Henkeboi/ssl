from keras.models import model_from_json


def store_model(model, name):
    model_json = model.to_json()
    with open('data/' + name + '.json', 'w') as data_file:
        data_file.write(model_json)
    model.save_weights('data/' + name + '.h5')

def load_model(name):
    json_file = open('data/' + name + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights('data/' + name + '.h5')
    return model

