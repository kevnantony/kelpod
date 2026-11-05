"""Error codes with user-facing messages (shown in the UI and audit log)."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    INGEST_UNSUPPORTED = "E_INGEST_UNSUPPORTED"
    INGEST_PARSE_FAILED = "E_INGEST_PARSE_FAILED"
    INGEST_EMPTY_CORPUS = "E_INGEST_EMPTY_CORPUS"
    SANITIZE_FAILED = "E_SANITIZE_FAILED"
    SYNTH_TOO_FEW_PAIRS = "E_SYNTH_TOO_FEW_PAIRS"
    MODEL_MISSING = "E_MODEL_MISSING"
    OOM_UNRECOVERABLE = "E_OOM_UNRECOVERABLE"
    TRAIN_FAILED = "E_TRAIN_FAILED"
    EVAL_FAILED = "E_EVAL_FAILED"
    GATES_FAILED = "E_GATES_FAILED"
    PACKAGE_FAILED = "E_PACKAGE_FAILED"
    CANCELLED = "E_CANCELLED"
    INTERNAL = "E_INTERNAL"


USER_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INGEST_UNSUPPORTED: "This file type is not supported and was skipped.",
    ErrorCode.INGEST_PARSE_FAILED: "This document could not be read. It was skipped.",
    ErrorCode.INGEST_EMPTY_CORPUS: "No readable text was found in the uploaded documents.",
    ErrorCode.SANITIZE_FAILED: "Privacy filtering failed. No data was used for training.",
    ErrorCode.SYNTH_TOO_FEW_PAIRS: "Too few reliable training examples could be created. "
    "Add more documents or check that they contain readable text.",
    ErrorCode.MODEL_MISSING: "A required model is not installed on this appliance.",
    ErrorCode.OOM_UNRECOVERABLE: "Training ran out of memory even with the smallest settings.",
    ErrorCode.TRAIN_FAILED: "Training failed. Your data is safe; you can resume the job.",
    ErrorCode.EVAL_FAILED: "Quality checks could not be completed.",
    ErrorCode.GATES_FAILED: "The new model did not pass all quality checks.",
    ErrorCode.PACKAGE_FAILED: "The model could not be packaged.",
    ErrorCode.CANCELLED: "The job was cancelled.",
    ErrorCode.INTERNAL: "An internal error occurred.",
}


class ZLLFError(Exception):
    """Base error carrying a stable code and a user-facing message."""

    def __init__(self, code: ErrorCode, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail or USER_MESSAGES[code]}")

    @property
    def user_message(self) -> str:
        return USER_MESSAGES[self.code]


class CancelledError(ZLLFError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(ErrorCode.CANCELLED, detail)
