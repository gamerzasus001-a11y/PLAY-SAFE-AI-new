from fastapi import FastAPI

app = FastAPI(title="PlaySafe AI")

@app.get("/")
def home():
    return {
        "message": "PlaySafe AI backend is running!"
    }