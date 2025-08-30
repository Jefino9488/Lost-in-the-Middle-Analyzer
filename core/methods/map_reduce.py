class MapReduce:
    name = "Map-Reduce"
    def __init__(self, model, chunk_size: int = 800, top_k: int = 3):
        self.model = model
        self.chunk_size = chunk_size
        self.top_k = top_k

    def _chunks(self, text):
        for i in range(0, len(text), self.chunk_size):
            yield text[i:i+self.chunk_size]

    def answer(self, question: str, document: str) -> str:
        # map: ask each chunk for an answer
        partials = []
        for ch in self._chunks(document):
            a = self.model.ask(question, ch)
            if a:
                partials.append(a.strip())

        # reduce: choose most common result that looks like code
        from collections import Counter
        votes = Counter([p for p in partials if p])
        if not votes:
            return partials[0] if partials else ""
        # prefer candidates containing 'ANSWER-'
        for candidate, _ in votes.most_common():
            if "ANSWER-" in candidate:
                return candidate
        # else return top voted
        return votes.most_common(1)[0][0]
