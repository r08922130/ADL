

class Logger:
    def __init__(self, model_name):
        self.file_name = model_name + "_log.txt"
    def write(self, s):
        with open(self.file_name, 'a') as f:
            f.write(s)