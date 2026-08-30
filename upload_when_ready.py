"""
Model upload watcher for the Inventory CV Pipeline.

Monitors for the final exported TFLite model (`models/inventory_model.tflite`)
and delivers it to a destination as soon as training + export finish. Designed
to run as an independent background process so it keeps working even after the
training terminal is closed.

Built-in destinations, selected via the `UPLOAD_METHOD` environment variable:

    * `email`  (default) - Emails the model as an attachment over SMTP so it
                           can be forwarded to a team.
    * `local`             - Copies the model into a local, cloud-synced folder
                            (e.g. OneDrive / Google Drive / Dropbox).
    * `sftp`              - Uploads to a remote SFTP server (system `sftp`).

Example (email):

    $env:UPLOAD_METHOD="email"
    $env:SMTP_HOST="smtp.gmail.com"
    $env:SMTP_PORT="587"
    $env:SMTP_USER="you@gmail.com"        # the sending account
    $env:SMTP_APP_PASSWORD="your_app_password"
    $env:SMTP_TO="teammate@company.com, another@company.com"
    python upload_when_ready.py
"""

from __future__ import annotations

import logging
import os
import shutil
import smtplib
import ssl
import subprocess
import sys
import time
from email import encoders
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import config.config as config

# Configure module-level logging.
logger = logging.getLogger("upload_when_ready")

# Polling interval (seconds) while waiting for the model to appear.
POLL_INTERVAL_S: int = int(os.getenv("UPLOAD_POLL_INTERVAL", "30"))

# How long to keep watching overall (0 = watch forever).
MAX_WAIT_S: int = int(os.getenv("UPLOAD_MAX_WAIT", "0"))  # 0 = indefinite


# -----------------------------------------------------------------------------
# Email destination
# -----------------------------------------------------------------------------
def _send_email(model_path: Path) -> str:
    """
    Email the exported model as an attachment via SMTP.

    Configuration is read from environment variables:
        SMTP_HOST          - e.g. smtp.gmail.com / smtp.office365.com / smtp.mail.yahoo.com
        SMTP_PORT          - e.g. 587 (STARTTLS) or 465 (SSL)
        SMTP_USER          - the sending account e-mail address
        SMTP_APP_PASSWORD  - app-specific password (Gmail: 16-char App Password)
        SMTP_TO            - comma-separated recipients
        SMTP_USE_TLS       - "true" to use STARTTLS on port 587 (default true)
        SMTP_FROM          - optional display name

    Args:
        model_path: Absolute path to the exported TFLite model.

    Returns:
        str: A summary of the recipients the email was sent to.

    Raises:
        ValueError: If required SMTP configuration is missing.
        RuntimeError: If the SMTP negotiation or send fails.
    """
    host: str = os.getenv("SMTP_HOST", "").strip()
    port: str = os.getenv("SMTP_PORT", "587").strip()
    user: str = os.getenv("SMTP_USER", "").strip()
    app_password: str = os.getenv("SMTP_APP_PASSWORD", "").strip()
    to_raw: str = os.getenv("SMTP_TO", "").strip()

    if not (host and user and app_password and to_raw):
        raise ValueError(
            "UPLOAD_METHOD='email' requires SMTP_HOST, SMTP_PORT, SMTP_USER, "
            "SMTP_APP_PASSWORD, and SMTP_TO to be set in the environment (.env)."
        )

    recipients: list[str] = [
        r.strip() for r in to_raw.split(",") if r.strip()
    ]
    use_tls: bool = os.getenv("SMTP_USE_TLS", "true").strip().lower() == "true"
    port_int: int = int(port)

    # Build the multipart message.
    msg = MIMEMultipart()
    msg["From"] = os.getenv("SMTP_FROM", user).strip() or user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = (
        f"YOLO Inventory CV Model - {time.strftime('%Y-%m-%d %H:%M')}"
    )

    body = (
        "Hi team,\n\n"
        "The inventory object-detection model has finished training and was "
        "exported to TensorFlow Lite. Please find the deployment-ready model "
        "attached.\n\n"
        f"Model file: {model_path.name}\n"
        f"Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB\n"
        f"Classes: 15\n"
        "\nBest regards,\nInventory CV Pipeline\n"
    )
    msg.attach(MIMEText(body, "plain"))

    # Attach the model.
    with open(model_path, "rb") as f:
        part = MIMEApplication(f.read(), _subtype="octet-stream")
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={model_path.name}",
    )
    encoders.encode_base64(part)
    msg.attach(part)

    # Send over SMTP.
    try:
        if use_tls:
            # STARTTLS on a plain connection.
            with smtplib.SMTP(host, port_int, timeout=120) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(user, app_password)
                server.sendmail(user, recipients, msg.as_string())
        else:
            # Explicit SSL (SMTPS) - typically port 465.
            with smtplib.SMTP_SSL(host, port_int, timeout=120) as server:
                server.login(user, app_password)
                server.sendmail(user, recipients, msg.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(
            "SMTP authentication failed. For Gmail, use a 16-character App "
            f"Password, not your normal login. Details: {exc}"
        ) from exc
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"SMTP error while sending: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Network error connecting to {host}:{port}: {exc}") from exc

    logger.info("Email sent successfully to: %s", ", ".join(recipients))
    return ", ".join(recipients)


# -----------------------------------------------------------------------------
# Local & SFTP destinations
# -----------------------------------------------------------------------------
def _copy_to_local_destination(model_path: Path) -> Path:
    """Copy the model into a local, cloud-synced directory (`UPLOAD_LOCAL_DIR`)."""
    dest_dir: str = os.getenv("UPLOAD_LOCAL_DIR", "").strip()
    if not dest_dir:
        raise ValueError(
            "UPLOAD_METHOD='local' requires `UPLOAD_LOCAL_DIR` to be set."
        )
    dest_dir_path: Path = Path(dest_dir).expanduser()
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    dest_file: Path = dest_dir_path / f"inventory_model_{time.strftime('%Y%m%d_%H%M%S')}.tflite"
    shutil.copy2(model_path, dest_file)
    logger.info("Copied model to local sync folder: %s", dest_file)
    return dest_file


def _upload_via_sftp(model_path: Path) -> str:
    """Upload the model to a remote SFTP server using the system `sftp` binary."""
    host: str = os.getenv("SFTP_HOST", "").strip()
    user: str = os.getenv("SFTP_USER", "").strip()
    remote_dir: str = os.getenv("SFTP_REMOTE_DIR", ".").strip()
    port: str = os.getenv("SFTP_PORT", "22").strip()

    if not host or not user:
        raise ValueError("UPLOAD_METHOD='sftp' requires `SFTP_HOST` and `SFTP_USER`.")

    remote_name: str = f"inventory_model_{time.strftime('%Y%m%d_%H%M%S')}.tflite"
    batch_text: str = (
        f"-mkdir -p {remote_dir}\n"
        f"put {str(model_path)} {remote_dir}/{remote_name}\n"
        "bye\n"
    )
    cmd: list[str] = ["sftp", "-P", port, "-b", "-", f"{user}@{host}"]

    try:
        proc = subprocess.run(
            cmd, input=batch_text, text=True, capture_output=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("SFTP upload timed out after 300s.") from None

    if proc.returncode != 0:
        logger.error("SFTP stderr: %s", proc.stderr)
        raise RuntimeError(f"SFTP upload failed (exit code {proc.returncode}).")

    logger.info("Uploaded model to sftp://%s:%s/%s/%s", host, port, remote_dir, remote_name)
    return f"sftp://{host}:{port}/{remote_dir}/{remote_name}"


# -----------------------------------------------------------------------------
# Upload dispatch
# -----------------------------------------------------------------------------
def upload_model(model_path: Path) -> None:
    """
    Dispatch the upload to the configured destination method.

    Args:
        model_path: Absolute path to the exported TFLite model.

    Raises:
        ValueError: If the upload method is unknown or misconfigured.
        RuntimeError: If the underlying transport raises an error.
    """
    method: str = os.getenv("UPLOAD_METHOD", "email").strip().lower()

    if method == "email":
        _send_email(model_path)
    elif method == "local":
        _copy_to_local_destination(model_path)
    elif method == "sftp":
        _upload_via_sftp(model_path)
    else:
        raise ValueError(
            f"Unknown UPLOAD_METHOD '{method}'. Choose 'email', 'local', or 'sftp'."
        )


# -----------------------------------------------------------------------------
# Watcher logic
# -----------------------------------------------------------------------------
def _validate_config() -> None:
    """Pre-flight validation so the user isn't left waiting for nothing."""
    method: str = os.getenv("UPLOAD_METHOD", "email").strip().lower()

    if method == "email":
        required = ["SMTP_HOST", "SMTP_USER", "SMTP_APP_PASSWORD", "SMTP_TO"]
        missing = [k for k in required if not os.getenv(k, "").strip()]
        if missing:
            raise ValueError(
                f"Email upload missing: {', '.join(missing)}. Set them in your "
                "environment or `.env` file."
            )
    elif method == "local":
        if not os.getenv("UPLOAD_LOCAL_DIR", "").strip():
            raise ValueError(
                "UPLOAD_METHOD='local' requires `UPLOAD_LOCAL_DIR` to be set."
            )
    elif method == "sftp":
        if not os.getenv("SFTP_HOST", "").strip() or not os.getenv("SFTP_USER", "").strip():
            raise ValueError(
                "SFTP requires `SFTP_HOST` and `SFTP_USER` to be set."
            )
    else:
        raise ValueError(f"Unknown UPLOAD_METHOD '{method}'.")


def watch_and_upload() -> None:
    """
    Watch for the exported TFLite model and deliver it once it appears.

    Polls `config.EXPORTED_TFLITE_PATH` every `POLL_INTERVAL_S` seconds until
    the file exists and is stable, then performs the configured upload.
    """
    logger.info("=" * 70)
    logger.info("Model upload watcher started")
    logger.info("=" * 70)
    logger.info("Watching for: %s", config.EXPORTED_TFLITE_PATH)
    logger.info("Upload method: %s", os.getenv("UPLOAD_METHOD", "email"))
    logger.info("Poll interval: %ss | Max wait: %ss (0=forever)", POLL_INTERVAL_S, MAX_WAIT_S)

    # Validate the destination config up-front so we fail fast.
    try:
        _validate_config()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    model_path: Path = config.EXPORTED_TFLITE_PATH
    elapsed: float = 0.0
    first_seen_size: int = -1
    stable_observed: int = 0

    while True:
        if model_path.exists():
            current_size: int = model_path.stat().st_size

            # The export is complete when the file size stops changing across
            # consecutive polls (guards against catching a half-written file).
            if current_size == first_seen_size:
                stable_observed += 1
            else:
                first_seen_size = current_size
                stable_observed = 0

            if stable_observed >= 2:  # size stable across 2 polls
                logger.info("Model detected and stable (%d bytes).", current_size)
                try:
                    upload_model(model_path)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Upload failed: %s", exc)
                    sys.exit(1)
                logger.info("=" * 70)
                logger.info("Delivery complete. Watcher exiting.")
                return
        else:
            logger.info("Model not present yet (waited %s s). Still training?", int(elapsed))

        if MAX_WAIT_S > 0 and elapsed >= MAX_WAIT_S:
            logger.error("Timed out after %s s without finding the model.", MAX_WAIT_S)
            sys.exit(1)

        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S


def setup_logging() -> None:
    """Configure logging to stdout for visibility in the background console."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


if __name__ == "__main__":
    setup_logging()
    watch_and_upload()
