class FullContext:
    name = "FullContext"
    def __init__(self, model):
        self.model = model
    def answer(self, question: str, document: str) -> str:
        # pass entire doc as context
        return self.model.ask(question, document)
