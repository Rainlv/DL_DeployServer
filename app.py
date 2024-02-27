import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import config
from database.db import create_db_and_tables
from extensions.log import log_init

sys.path.append(str(Path(__file__).parent.absolute() / 'src'))
from apis import root_router

app = FastAPI()
app.include_router(root_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.middleware("http")
# async def db_session_middleware(request: Request, call_next):
#     request.state.config = config
#     response = await call_next(request)
#     return response

@app.on_event("startup")
async def startup():
    log_init()
    await create_db_and_tables()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app='app:app', host="0.0.0.0", port=config.PORT, reload=config.DEBUG)
