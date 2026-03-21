# COCO keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# COCO skeleton connections (pairs of keypoint indices)
# Eyes and ears excluded — only body joints relevant to shot mechanics
SKELETON_CONNECTIONS = [
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW), (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW), (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP), (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_LEFT_KNEE), (KP_LEFT_KNEE, KP_LEFT_ANKLE),
    (KP_RIGHT_HIP, KP_RIGHT_KNEE), (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
]

# Biomechanics thresholds
IDEAL_ELBOW_ANGLE = 90.0       # degrees
IDEAL_KNEE_MIN = 90.0          # degrees
IDEAL_KNEE_MAX = 120.0         # degrees
IDEAL_RELEASE_ARC_MIN = 45.0   # degrees above horizontal
IDEAL_RELEASE_ARC_MAX = 52.0   # degrees above horizontal

# Detection parameters
MIN_CONFIDENCE = 0.5
WRIST_TRAJECTORY_WINDOW = 15
WRIST_UPWARD_THRESHOLD = -5.0  # px/frame (negative = upward in image coords)

# Suggestion rule thresholds
ELBOW_ANGLE_TOLERANCE = 20.0   # degrees deviation from ideal before flagging
TRAJECTORY_CONSISTENCY_STD = 5.0  # degrees std dev before flagging

# Shot detector state machine
SHOT_COOLDOWN_FRAMES = 30
COCKING_KNEE_THRESHOLD = 130.0   # knee < this to be "cocked"
RELEASE_CONSECUTIVE_FRAMES = 3
FOLLOW_THROUGH_VELOCITY_DROP = 0.5  # 50% drop from peak

# Held object detection
HELD_OBJECT_MODEL = "yolo11n.pt"
HELD_OBJECT_WRIST_RADIUS = 300   # px — crop region around wrist for detection
HELD_OBJECT_NUM_NODES = 8        # number of nodes drawn around a held object
HELD_OBJECT_NODE_COLOR = (0, 128, 255)   # BGR orange-blue
HELD_OBJECT_LINE_COLOR = (0, 180, 255)

# Whitelist of COCO classes that can realistically be held in a hand
HELD_OBJECT_ALLOWED_CLASSES = {
    "sports ball", "bottle", "cell phone", "remote", "cup",
    "tennis racket", "baseball bat", "frisbee", "orange", "apple",
    "banana", "scissors", "knife", "fork", "spoon",
}

# Ball tracking
BALL_COCO_CLASS_ID = 32
BALL_CONF_THRESHOLD = 0.25
BALL_HELD_DISTANCE_PX = 120
BALL_TRAJECTORY_WINDOW = 60
BALL_MIN_FLIGHT_FRAMES = 4
BALL_ARC_SMOOTHING_WINDOW = 5
BALL_MIN_SIZE_PX = 40           # minimum bbox width AND height to count as a real ball
BALL_PERSON_PADDING_PX = 200    # expand person bbox by this much when filtering ball position

# Model and output defaults
DEFAULT_MODEL = "yolo11n-pose.pt"
DEFAULT_REPORT_FILE = "shot_report.json"
DEFAULT_FPS = 30.0
DEFAULT_IOU = 0.45
