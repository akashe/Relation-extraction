import os

def create_if_not_there_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)