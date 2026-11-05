"""Health and hardware."""

from __future__ import annotations

from fastapi import APIRouter

from zllf import __version__
from zllf.api.deps import CurrentUser, SettingsDep
from zllf.api.hardware import hardware_profile
from zllf.api.schemas import Health
from zllf.core.schemas import HardwareProfile

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", version=__version__)


@router.get("/hardware", response_model=HardwareProfile)
def hardware(_user: CurrentUser, settings: SettingsDep) -> HardwareProfile:
    return hardware_profile(settings.hardware, settings.root, settings.config_dir)
