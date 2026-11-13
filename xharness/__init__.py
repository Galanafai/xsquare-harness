"""xharness: the agent harness between the X Square desktop robot and Wall-X.

Modules, in the order a request flows through them:

    protocol      wire format (public_v2), validation, HOLD chunks
    relay         WebSocket server the robot connects to; talks to Wall-X
    supervisor    stage machine: what to do next, retries, done detection
    verifier      narrow yes/no questions to a vision model
    action_filter last software gate before actions reach the robot
    recorder      everything to disk for replay and debugging
    tasks         YAML task / stage definitions
    parts         Demo 2 part dimensions and gripper feasibility
    config        YAML + environment configuration
"""

__version__ = "0.1.0"
