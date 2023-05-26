import random
class RandomSelect:
    def __init__(self, data, k=None):
        self.data = data
        self.k = k
        # You can also add code to learn parameters here.

    def select(self, context):
        data_without_context = [item for item in self.data if item != context]
        if self.k is None:
            return data_without_context
        else:
            return random.sample(data_without_context, self.k)
