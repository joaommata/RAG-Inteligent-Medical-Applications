import json
import atexit
import string
import random

FOLDER = r"C:\Users\CCIG\joao_mata\classifications" 

def generate_random_id(id_length: int, chars: str = string.ascii_uppercase + string.digits) -> str:
    return ''.join(random.SystemRandom().choices(chars, k=id_length))

class ClassificationWriter:
    def __init__(self, conversation_id=None):
        self.conversation_id = conversation_id
        if self.conversation_id is None:
            self.conversation_id = generate_random_id(16)
        self.path = f"{FOLDER}/{self.conversation_id}.json"
        
        self.chat_history = []
        self.classifications = []
        self.contexts = []

        
        atexit.register(self.write) # aqui estamos a definir o que é executado quando o programa falha ou cai

    def update_chat_history(self, id: str, content: str):
        # chamar isto com cada mensagem, seja ela humana ou da AI
        self.chat_history.append(
            {"id": id, "content": content, "idx": len(self.chat_history)}
        )

    def update_classifications(self, classification: str):
        self.classifications.append(
            {"classification": classification, "idx": len(self.chat_history) - 1}
        )

    def update_context(self, context: str):
        self.contexts.append(
            { "content": context, "idx": len(self.chat_history)}
        )
        

    def write(self):
        with open(self.path, "w") as o:
            data = {
                "chat_history": self.chat_history,
                "classifications": self.classifications,
                "conversation_id": self.conversation_id,
                "contexts": self.contexts 
            }
            json.dump(data, o)

 