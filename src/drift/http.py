"""Bound requests and require authentication when the pilot is deployed."""

import hmac
import json
import os


def configuration_error():
    if os.getenv("APP_ENV", "local") == "production":
        for name in (
            "DRIFT_STATE_PATH",
            "FRAUD_REFERENCE_PATH",
            "FRAUD_REFERENCE_SHA256",
        ):
            if not os.getenv(name):
                return f"Production requires {name}."
        if len(os.getenv("DRIFT_API_KEY", "")) < 32:
            return "Production requires DRIFT_API_KEY with at least 32 characters."
    return None


class MonitoringBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def reject(status, detail):
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps({"detail": detail}).encode(),
                }
            )

        if scope["path"] not in {"/health", "/live", "/ready"}:
            error = configuration_error()
            if error:
                return await reject(503, error)
            expected = os.getenv("DRIFT_API_KEY", "")
            if expected and not hmac.compare_digest(
                dict(scope.get("headers", [])).get(b"x-api-key", b""), expected.encode()
            ):
                return await reject(401, "Valid X-API-Key required.")
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > 4 * 1024 * 1024:
                return await reject(413, "Request body exceeds 4 MiB.")
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, bounded_receive, send)
