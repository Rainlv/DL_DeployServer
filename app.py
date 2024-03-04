import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).parent.absolute() / 'src'))

from config import config
from database.db import create_db_and_tables
from extensions.exceptions import BaseWebException, ExternalServiceException
from extensions.log import log_init
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

app.add_exception_handler(BaseWebException, BaseWebException.handler)
app.add_exception_handler(ExternalServiceException, ExternalServiceException.handler)


@app.on_event("startup")
async def startup():
    log_init()
    # await create_db_and_tables()
    create_db_and_tables()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app='app:app', host="0.0.0.0", port=config.PORT, reload=config.DEBUG)
