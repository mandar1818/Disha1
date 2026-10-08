"""
SmartVisionAI - Indoor Safety Package
=====================================
Specialized indoor safety reasoning for blind and visually impaired users:
- Door / doorway understanding (open vs closed vs blocked)
- Stairs, steps, and downward drop-offs / floor cliffs
- Floor-level micro-hazards (cables, low thresholds, small boxes)
- Glass and transparent obstacle reasoning
- Corridor structure, vanishing point, and narrow-passage clearance
- Unified IndoorSafetyManager
"""

from backend.indoor.door import DoorwayAnalyzer
from backend.indoor.stairs import StairDropoffAnalyzer
from backend.indoor.floor_hazard import FloorHazardAnalyzer
from backend.indoor.glass import GlassHazardAnalyzer
from backend.indoor.corridor import CorridorClearanceAnalyzer
from backend.indoor.safety_manager import IndoorSafetyManager

__all__ = [
    "DoorwayAnalyzer",
    "StairDropoffAnalyzer",
    "FloorHazardAnalyzer",
    "GlassHazardAnalyzer",
    "CorridorClearanceAnalyzer",
    "IndoorSafetyManager",
]

