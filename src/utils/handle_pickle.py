import pickle


def dump(data: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load(path: str) -> object:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
