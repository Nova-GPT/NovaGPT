

class TrainingData():
    _cache = {}

    @classmethod
    def _load_file(cls, filename, encoding='utf-8') -> str:
        if filename not in cls._cache:
            with open(filename, 'r', encoding=encoding) as f:
                cls._cache[filename] = f.read()
        return cls._cache[filename]

    @classmethod
    def shakesphere(cls, encoding='utf-8') -> str:
        return cls._load_file('data/shakesphere.txt', encoding)

    @classmethod
    def stories(cls, encoding='utf-8') -> str:
        return cls._load_file('data/stories.txt', encoding)

    @classmethod
    def TinyStories(cls, encoding='utf-8') -> str:
        return cls._load_file('data/TinyStories_part1.txt', encoding)
