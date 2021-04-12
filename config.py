import re

class Config:
    def __init__(self):
        self.data = {
            'dataset' : None,
            'la_autoencoder' : None,
            'la_autoencoder_classifier' : None,
            'la_simple_classifier' : None,
            'loss_function_autoencoder' : None,
            'loss_function_classifier' : None,
            'optimizer_autoencoder' : None,
            'optimizer_classifier' : None,
            'mnist_latent_size' : None,
            'fashion_mnist_latent_size' : None,
            'cifar10_latent_size' : None,
            'digits_latent_size' : None,
            'mnist_autoencoder_epochs' : None,
            'fashion_mnist_autoencoder_epochs' : None,
            'cifar10_autoencoder_epochs' : None,
            'digits_autoencoder_epochs' : None,
            'mnist_classifier_epochs' : None,
            'fashion_mnist_classifier_epochs' : None,
            'cifar10_classifier_epochs' : None,
            'digits_classifier_epochs' : None,
            'freeze' : None
        }
    
    def get_config(self):
        self.read_config()
        return self.data

    def read_config(self):
        with open("./config_file.txt", "r") as f:
            config_data = f.readlines()

        for line in config_data:
            line = line.strip("\n")
            try:
                variable, value = line.split(":")
                if variable in self.data:
                    self.parse_data(variable, value)
            except:
                print(variable + " is missing a value.")
    
    def parse_data(self, variable, data):
        if variable == 'dataset':
            self.data[variable] = data
        elif variable == 'la_autoencoder':
            self.data[variable] = float(data)
        elif variable == 'la_autoencoder_classifier':
            self.data[variable] = float(data)
        elif variable == 'la_simple_classifier':
            self.data[variable] = float(data)
        elif variable == 'loss_function_autoencoder':
            self.data[variable] = data
        elif variable == 'loss_function_classifier':
            self.data[variable] = data
        elif variable == 'optimizer_autoencoder':
            self.data[variable] = data
        elif variable == 'optimizer_classifier':
            self.data[variable] = data
        elif variable == 'mnist_latent_size':
            self.data[variable] = int(data)
        elif variable == 'fashion_mnist_latent_size':
            self.data[variable] = int(data)
        elif variable == 'cifar10_latent_size':
            self.data[variable] = int(data)
        elif variable == 'digits_latent_size':
            self.data[variable] = int(data)
        elif variable == 'mnist_autoencoder_epochs':
            self.data[variable] = int(data)
        elif variable == 'fashion_mnist_autoencoder_epochs':
            self.data[variable] = int(data)
        elif variable == 'cifar10_autoencoder_epochs':
            self.data[variable] = int(data)
        elif variable == 'digits_autoencoder_epochs':
            self.data[variable] = int(data)
        elif variable == 'mnist_classifier_epochs':
            self.data[variable] = int(data)
        elif variable == 'fashion_mnist_classifier_epochs':
            self.data[variable] = int(data)
        elif variable == 'cifar10_classifier_epochs':
            self.data[variable] = int(data)
        elif variable == 'digits_classifier_epochs':
            self.data[variable] = int(data)
        elif variable == 'freeze':
            self.data[variable] = int(data)





