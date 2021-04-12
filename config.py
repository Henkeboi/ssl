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
            'optimizer_autoencoder': None,
            'optimizer_classifier': None
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



