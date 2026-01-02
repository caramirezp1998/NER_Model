from fastapi import FastAPI
import spacy

app = FastAPI()

nlp = spacy.load("en_core_web_sm")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ner")
def ner(text: str):
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
    ]
