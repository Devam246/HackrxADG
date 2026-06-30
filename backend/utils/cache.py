from collections import OrderedDict


class DocumentCache(OrderedDict):
    def __init__(self, capacity: int = 10) -> None:
        super().__init__()
        self.capacity = capacity

    def get(self, key: str, default=None):
        if key in self:
            self.move_to_end(key)
            return super().get(key)
        return default

    def set(self, key: str, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)


document_cache = DocumentCache(capacity=10)
