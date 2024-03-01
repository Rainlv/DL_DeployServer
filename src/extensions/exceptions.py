from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger


class BaseWebException(HTTPException):
    def __init__(self, code: int = 500, message: str = "服务器内部错误"):
        self.message = message
        self.code = code
        super().__init__(status_code=code, detail=message)

    @staticmethod
    def handler(request, exc):
        logger.error(f"Request error: {exc}, request {request}")
        return JSONResponse({
            "code": exc.code,
            "message": exc.message
        })


class ExternalServiceException(BaseWebException):
    def __init__(self, code: int = 500, message: str = "外部服务错误"):
        super().__init__(code, message)
