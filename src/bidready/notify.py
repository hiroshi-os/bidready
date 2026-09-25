"""Webhook and email notifications. Both are stubs unless the environment points them at a server."""

from __future__ import annotations

import smtplib
import uuid
from email.message import EmailMessage

import httpx
from sqlalchemy.orm import Session

from bidready.config import Settings
from bidready.models import Notification, Run


def notify_run(session: Session, settings: Settings, run: Run, *, title: str, go_no_go: str) -> None:
    summary = {
        "run_id": run.id,
        "case_id": run.case_id,
        "title": title,
        "go_no_go": go_no_go,
        "status": run.status,
    }
    _webhook(session, settings, run, summary)
    _email(session, settings, run, summary)


def _webhook(session: Session, settings: Settings, run: Run, summary: dict) -> None:
    if not settings.webhook_url:
        session.add(
            Notification(
                id=uuid.uuid4().hex,
                run_id=run.id,
                channel="webhook",
                status="stubbed",
                payload_json={"reason": "WEBHOOK_URL is unset", "summary": summary},
            )
        )
        return
    try:
        response = httpx.post(settings.webhook_url, json=summary, timeout=5)
        session.add(
            Notification(
                id=uuid.uuid4().hex,
                run_id=run.id,
                channel="webhook",
                status="sent" if response.status_code < 400 else "error",
                payload_json={"status_code": response.status_code, "summary": summary},
            )
        )
    except Exception as exc:
        session.add(
            Notification(
                id=uuid.uuid4().hex,
                run_id=run.id,
                channel="webhook",
                status="error",
                payload_json={"error": str(exc), "summary": summary},
            )
        )


def _email(session: Session, settings: Settings, run: Run, summary: dict) -> None:
    body = (
        f"bidready finished {summary['title']}.\n"
        f"Decision: {summary['go_no_go']}\n"
        f"Case: {summary['case_id']}\n"
        "This message is a notification stub, not legal advice.\n"
    )
    if not settings.smtp_host or not settings.smtp_to:
        session.add(
            Notification(
                id=uuid.uuid4().hex,
                run_id=run.id,
                channel="email",
                status="stubbed",
                payload_json={"reason": "SMTP_HOST or SMTP_TO is unset", "body": body, "summary": summary},
            )
        )
        return
    message = EmailMessage()
    message["Subject"] = f"bidready {summary['go_no_go']}: {summary['title']}"
    message["From"] = settings.smtp_from or "bidready@localhost"
    message["To"] = settings.smtp_to
    message.set_content(body)
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as client:
            client.send_message(message)
        status = "sent"
        payload: dict = {"summary": summary}
    except Exception as exc:
        status = "error"
        payload = {"error": str(exc), "body": body, "summary": summary}
    session.add(
        Notification(
            id=uuid.uuid4().hex,
            run_id=run.id,
            channel="email",
            status=status,
            payload_json=payload,
        )
    )
