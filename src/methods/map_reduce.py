class MapReduce:
    name = "Map-Reduce"
    def __init__(self, model, chunk_size: int = 800, top_k: int = 3):
        self.model = model
        self.chunk_size = chunk_size
        self.top_k = top_k

    def _chunks(self, text):
        for i in range(0, len(text), self.chunk_size):
            yield text[i:i+self.chunk_size]

                return candidate
        # else return top voted
        return votes.most_common(1)[0][0]
