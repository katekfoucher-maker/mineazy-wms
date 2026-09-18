"""FastAPI application: web UI (server-rendered) + JSON API + Swagger."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from wms.config import get_settings
from wms.db import init_db
from wms.errors import WMSError
from wms.api.routes import backorders, analytics, reports
from wms.web.deps import Redirect
from wms.web.routes import router as web_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(SessionMiddleware, secret_key=settings.secret_key,
                   max_age=settings.session_max_age, same_site="lax")

# generated charts / files, referenced by the reports page as /output/<name>
app.mount("/output", StaticFiles(directory=str(settings.out)), name="output")


@app.exception_handler(WMSError)
async def _wms_error(_: Request, exc: WMSError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


@app.exception_handler(Redirect)
async def _redirect(_: Request, exc: Redirect):
    return RedirectResponse(exc.url, status_code=303)


# JSON API
for r in (backorders.router, analytics.router, reports.router):
    app.include_router(r)

# Browser UI (server-rendered, session auth) - mounted last, owns "/"
app.include_router(web_router)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name}
