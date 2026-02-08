from fastapi import FastAPI
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Cabby is running"}

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=1604,
        reload=True,
        log_level="info",
    )