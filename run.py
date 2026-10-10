"""Sobe o Escriba em http://localhost:8000"""
import uvicorn

if __name__ == "__main__":
    # Para acessar de outros dispositivos da rede: host="0.0.0.0"
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
