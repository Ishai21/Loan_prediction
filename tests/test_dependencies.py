"""Guards the starlette pin.

Streamlit 1.60.0 overrides Starlette's gzip middleware `send_with_compression`
and relies on the base class NOT forwarding the `http.response.start` message.
Starlette 1.8.0 changed that contract, so any starlette >= 1.8.0 makes uvicorn
raise `RuntimeError: Expected ASGI message 'http.response.start'` on responses
that skip compression (the /media/ route: st.image, st.download_button, ...).

Starlette 1.7.0 is the last compatible release. If this test fails, the pin in
requirements.txt was relaxed -- do not "fix" the test.
"""

from importlib.metadata import version


def _parse(v):
    return tuple(int(p) for p in v.split(".")[:2] if p.isdigit())


def test_starlette_is_compatible_with_streamlit_gzip_middleware():
    installed = version("starlette")
    assert _parse(installed) < (1, 8), (
        f"starlette {installed} breaks Streamlit's gzip middleware. "
        "Pin starlette==1.7.0 in requirements.txt."
    )


def test_streamlit_gzip_middleware_forwards_response_start():
    """Behavioural check: the full middleware chain must emit start before body."""
    import asyncio

    from streamlit.web.server.starlette.starlette_gzip_middleware import (
        MediaAwareGZipMiddleware,
    )

    sent = []

    async def app(scope, receive, send):
        # image/png is an "excluded" content type -> exercises the broken path
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"image/png")],
            }
        )
        await send({"type": "http.response.body", "body": b"x" * 2000})

    async def send(message):
        sent.append(message["type"])

    async def receive():
        return {"type": "http.request"}

    async def main():
        await MediaAwareGZipMiddleware(app)(
            {
                "type": "http",
                "path": "/media/chart.png",
                "headers": [(b"accept-encoding", b"gzip")],
            },
            receive,
            send,
        )

    asyncio.run(main())

    assert sent[:1] == ["http.response.start"], (
        f"middleware emitted {sent} -- uvicorn will raise RuntimeError. "
        "starlette is incompatible with this Streamlit version."
    )
