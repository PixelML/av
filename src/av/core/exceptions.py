"""Exception hierarchy for av."""


class AVError(Exception):
    """Base exception for all av errors."""


class FFmpegError(AVError):
    """FFmpeg/ffprobe command failed."""

    def __init__(self, message: str, cmd: str | None = None, returncode: int | None = None):
        self.cmd = cmd
        self.returncode = returncode
        super().__init__(message)


class APIError(AVError):
    """Remote API call failed."""

    def __init__(self, message: str, provider: str | None = None, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class DatabaseError(AVError):
    """Database operation failed."""


class IngestError(AVError):
    """Ingest pipeline error."""


class VideoNotFoundError(AVError):
    """Video ID not found in database."""
