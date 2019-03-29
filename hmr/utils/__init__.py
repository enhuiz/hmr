import os
import json


class Config(object):
    def __init__(self, source):
        self.source = source
        self.basename = os.path.basename(source)

        with open(source, 'r') as f:
            self.data = json.load(f)

        self.__dict__.update(self.data)

    def save(self, dir_):
        path = os.path.join(dir_, self.basename)
        with open(path, 'w') as f:
            json.dump(f, self.data)
