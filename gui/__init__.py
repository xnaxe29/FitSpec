"""FitSpec GUI package: common application shell plus science panels."""
from gui.state import SessionState
from gui.base_gui import BaseGUI
from gui.app import FitSpecApp, build_session
from gui.mask_controller import MaskController
from gui.component_controller import ComponentController
from gui.stellar import StellarGUI
from gui.emission import EmissionGUI
from gui.absorption import AbsorptionGUI

__all__ = [
    "SessionState", "BaseGUI", "FitSpecApp", "build_session",
    "MaskController", "ComponentController",
    "StellarGUI", "EmissionGUI", "AbsorptionGUI",
]
