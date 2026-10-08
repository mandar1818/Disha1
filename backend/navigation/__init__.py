"""
SmartVisionAI - Indoor Destination Navigation Package
=====================================================
Closed-loop vision-based local destination navigation:
- Destination representation and semantic landmark matching
- Turn verification via multi-frame visual change detection
- Relative destination proximity tracking (FAR -> NEAR -> VERY_CLOSE)
- Arrival detection with stability guarantees
- Navigation session state machine with strict safety precedence
"""

from backend.navigation.destination import Destination, match_destination
from backend.navigation.verifier import TurnVerifier
from backend.navigation.progress import ProgressTracker
from backend.navigation.session import NavigationSession, NavigationSessionState

__all__ = [
    "Destination",
    "match_destination",
    "TurnVerifier",
    "ProgressTracker",
    "NavigationSession",
    "NavigationSessionState",
]

