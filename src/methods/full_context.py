class FullContext:
    name = "FullContext"
    def __init__(self, model):
        self.model = model
    def answer(self, question: str, document: str) -> dict:
        ans = self.model.ask(question, document)
        return {"answer": ans, "retrieved_chunks": [document]}
