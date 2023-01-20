from read import read_game


class Game:
    def __init__(self, path):
        self.data = read_game(path)
        for key, value in self.data.items():
            if isinstance(value, dict):
                if key == 'meta':
                    self.meta = Meta(value)

                if key == 'info':
                    self.info = Info(value)
            elif key == 'innings':
                self.innings = Innings(value)
            else:
                self.__dict__[key] = value


class Meta:
    def __init__(self, meta_data):
        for key, value in meta_data.items():
            self.__dict__[key] = value


class Info:
    def __init__(self, info_data):
        for key, value in info_data.items():
            self.__dict__[key] = value


class Innings:
    def __init__(self, innings):
        self.innings = innings
        self.length = int(len(innings))

