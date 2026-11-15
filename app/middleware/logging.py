import time
import uuid

from fastapi import Request


async def request_logging_middleware(
    request: Request,
    call_next
):
    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = request_id

    start = time.perf_counter()

    try:
        response = await call_next(
            request
        )

        return response

    finally:
        elapsed = (
            time.perf_counter() - start
        ) * 1000

        print(
            f"[{request_id}] "
            f"{request.method} "
            f"{request.url.path} "
            f"{elapsed:.2f}ms"
        )