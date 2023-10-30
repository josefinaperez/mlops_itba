import mlflow
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from model_src.preprocess import preprocess_text

class NER_HFModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

    def load_context(self, context):        
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint)
        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    def predict(self, context, model_input):
        preprocess_text(model_input)
        predictions = self.nlp(model_input['text'].to_list())
        return predictions