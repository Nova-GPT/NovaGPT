class TrainingData():
    def __init__(self, file_path):
        self.file_path = file_path

    def shakesphere(encoding='utf-8'):
        with open('data/shakesphere.txt', 'r', encoding=encoding) as f:
            data = f.read()
        return data
    
    def stories(encoding='utf-8'):
        with open('data/stories.txt', 'r', encoding=encoding) as f:
            data = f.read()
        return data
