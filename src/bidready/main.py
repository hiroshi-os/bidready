"""FastAPI application and the small server-rendered UI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from bidready import __version__
from bidready.config import Settings
from bidready.db import make_session_factory
from bidready.models import Case, Notification
from bidready.pipeline import analyse, latest_run
from bidready.synthetic import get_profile, list_profiles
from bidready.tenderlens import TenderlensAdapter, TenderlensError, TenderlensNotConfigured

logger = logging.getLogger(__name__)
_PACKAGE = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE / "web" / "templates"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    session_factory = make_session_factory(settings)
    app = FastAPI(title="bidready", version=__version__)
    app.state.settings = settings
    app.state.session_factory = session_factory
    static_dir = _PACKAGE / "web" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
            "reranker": settings.reranker,
            "tenderlens_configured": bool(settings.tenderlens_base_url),
        }

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        with _session(session_factory) as session:
            cases = list(session.scalars(select(Case).order_by(Case.created_at.desc()).limit(20)))
            payload = [_case_brief(case) for case in cases]
        return _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {"profiles": list_profiles(), "cases": payload, "settings": settings},
        )

    @app.post("/cases")
    async def create_case(
        tender_files: list[UploadFile] = File(...),
        company_files: list[UploadFile] | None = File(None),
        profile_id: str = Form(""),
        title: str = Form(""),
    ):
        tenders = await _read_uploads(tender_files)
        companies = await _read_uploads(company_files or [])
        chosen = profile_id.strip() or None
        if chosen and get_profile(chosen) is None:
            raise HTTPException(status_code=400, detail="unknown synthetic profile")
        with _session(session_factory) as session:
            case_id = analyse(
                settings,
                session,
                tender_files=tenders,
                company_files=companies,
                profile_id=chosen,
                title=title.strip() or None,
            )
        return RedirectResponse(f"/cases/{case_id}", status_code=303)

    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    def case_page(request: Request, case_id: str):
        with _session(session_factory) as session:
            view = _case_view(session, case_id)
        if view is None:
            raise HTTPException(status_code=404, detail="case not found")
        return _TEMPLATES.TemplateResponse(request, "case.html", view)

    @app.get("/api/cases/{case_id}")
    def case_json(case_id: str):
        with _session(session_factory) as session:
            view = _case_view(session, case_id)
        if view is None:
            raise HTTPException(status_code=404, detail="case not found")
        return JSONResponse(
            {
                "case": view["case"],
                "report": view["report"],
                "notifications": view["notifications"],
            }
        )

    @app.post("/api/cases")
    async def create_case_json(
        tender_files: list[UploadFile] = File(...),
        company_files: list[UploadFile] | None = File(None),
        profile_id: str = Form(""),
        title: str = Form(""),
    ):
        tenders = await _read_uploads(tender_files)
        companies = await _read_uploads(company_files or [])
        chosen = profile_id.strip() or None
        with _session(session_factory) as session:
            case_id = analyse(
                settings,
                session,
                tender_files=tenders,
                company_files=companies,
                profile_id=chosen,
                title=title.strip() or None,
            )
            view = _case_view(session, case_id)
        return JSONResponse({"case": view["case"], "report": view["report"]})

    @app.post("/imports/tenderlens/{tender_id}")
    def import_tenderlens(tender_id: str, profile_id: str = Form("")):
        adapter = TenderlensAdapter(settings.tenderlens_base_url)
        try:
            pack = adapter.fetch(tender_id)
        except TenderlensNotConfigured as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TenderlensError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        chosen = profile_id.strip() or None
        files = [(document.filename, document.content) for document in pack.documents]
        with _session(session_factory) as session:
            case_id = analyse(
                settings,
                session,
                tender_files=files,
                profile_id=chosen,
                title=pack.title,
            )
        return RedirectResponse(f"/cases/{case_id}", status_code=303)

    return app


def _session(session_factory):
    return _SessionScope(session_factory)


class _SessionScope:
    def __init__(self, session_factory) -> None:
        self._factory = session_factory
        self.session: Session | None = None

    def __enter__(self) -> Session:
        self.session = self._factory()
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.session is not None
        if exc_type is None:
            self.session.commit()
        else:
            self.session.rollback()
        self.session.close()


async def _read_uploads(uploads: list[UploadFile]) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for upload in uploads:
        if not upload.filename:
            continue
        data = await upload.read()
        if data:
            files.append((upload.filename, data))
    return files


def _case_brief(case: Case) -> dict:
    return {
        "id": case.id,
        "title": case.title,
        "status": case.status,
        "go_no_go": case.go_no_go,
        "created_at": case.created_at.isoformat() if case.created_at else "",
    }


def _case_view(session: Session, case_id: str) -> dict | None:
    case = session.get(Case, case_id)
    if case is None:
        return None
    run = latest_run(session, case_id)
    report = run.report_json if run and run.report_json else None
    notifications = []
    if run:
        rows = session.scalars(select(Notification).where(Notification.run_id == run.id)).all()
        notifications = [
            {"channel": row.channel, "status": row.status, "payload": row.payload_json} for row in rows
        ]
    return {
        "case": {
            **_case_brief(case),
            "error": case.error,
            "profile_id": case.profile_id,
            "latency_ms": run.latency_ms if run else None,
            "provider": run.provider if run else None,
            "model": run.model if run else None,
        },
        "report": report,
        "notifications": notifications,
    }


def cli() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("bidready.main:app", host="0.0.0.0", port=8000, reload=False)


app = create_app()
