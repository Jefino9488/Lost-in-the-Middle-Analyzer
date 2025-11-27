class SlidingWindow:
    name = "SlidingWindow"
    def __init__(self, model, window_size: int = 800, num_windows: int = 3):
        self.model = model
        self.window_size = window_size
        self.num_windows = num_windows

    def answer(self, question: str, document: str) -> dict:
        doc = document
        L = len(doc)
        # create overlapping windows evenly spaced
        if self.num_windows <= 0:
            return {"answer": "", "retrieved_chunks": []}
        step = max(1, (L - self.window_size) // max(1, self.num_windows - 1)) if L > self.window_size else self.window_size
        candidates = []
        retrieved = []
        for i in range(0, L, step):
            chunk = doc[i:i + self.window_size]
            if not chunk:
                continue
            retrieved.append(chunk)
            candidates.append(self.model.ask(question, chunk))
            if len(candidates) >= self.num_windows:
                break

        # first try to return any candidate that looks valid
        ans = ""
        for c in candidates:
            if c and "couldn't find" not in c.lower():
                ans = c
                break
        if not ans and candidates:
            ans = candidates[0]
            
        return {"answer": ans, "retrieved_chunks": retrieved}
