from abc import ABC, abstractmethod
import random

class SelectMethod(ABC):
    @abstractmethod
    def __init__(self, data):
        pass

    @abstractmethod
    def select(self, context):
        pass

class RandomSelect(SelectMethod):
    def __init__(self, data, k=None):
        self.data = data
        self.k = k

    def select(self, context):
        data_without_context = [item for item in self.data if item != context]
        if self.k is None:
            return data_without_context
        else:
            return random.sample(data_without_context, self.k)


