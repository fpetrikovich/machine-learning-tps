from enum import Enum

class Headers(Enum):
    WEIGHT = 'weight'
    BICROMIAL = 'biacromial'
    PELVIC_BREADTH = 'pelvic.breadth'
    BITROCHANTERIC = 'bitrochanteric'
    CHEST_DEPTH = 'chest.depth'
    CHEST_DIAM = 'chest.diam'
    ELBOW_DIAM = 'elbow.diam'
    WRIST_DIAM = 'wrist.diam'
    KNEE_DIAM = 'knee.diam'
    ANKLE_DIAM = 'ankle.diam'
    SHOULDER_GIRTH = 'shoulder.girth'
    CHEST_GIRTH = 'chest.girth'
    WAIST_GIRTH = 'waist.girth'
    NAVEL_GIRTH = 'navel.girth'
    HIP_GIRTH = 'hip.girth'
    THIGH_GIRTH = 'thigh.girth'
    BICEP_GIRTH = 'bicep.girth'
    FOREARM_GIRTH = 'forearm.girth'
    KNEE_GIRTH = 'knee.girth'
    CALF_GIRTH = 'calf.girth'
    ANKLE_GIRTH = 'ankle.girth'
    WRIST_GIRTH = 'wrist.girth'
    AGE = 'age'
    HEIGHT = 'height'

x_headers_list = [ Headers.BICROMIAL.value, Headers.PELVIC_BREADTH.value, Headers.BITROCHANTERIC.value, Headers.CHEST_DEPTH.value,
    Headers.CHEST_DIAM.value, Headers.ELBOW_DIAM.value, Headers.WRIST_DIAM.value, Headers.KNEE_DIAM.value, Headers.ANKLE_DIAM.value,
    Headers.SHOULDER_GIRTH.value, Headers.CHEST_GIRTH.value, Headers.WAIST_GIRTH.value, Headers.NAVEL_GIRTH.value, Headers.HIP_GIRTH.value,
    Headers.THIGH_GIRTH.value, Headers.BICEP_GIRTH.value, Headers.FOREARM_GIRTH.value, Headers.KNEE_GIRTH.value, Headers.CALF_GIRTH.value,
    Headers.ANKLE_GIRTH.value, Headers.WRIST_GIRTH.value, Headers.AGE.value, Headers.HEIGHT.value ]

best_headers_list = [ Headers.WAIST_GIRTH.value, Headers.HEIGHT.value, Headers.HIP_GIRTH.value,
                Headers.THIGH_GIRTH.value, Headers.CHEST_GIRTH.value, Headers.KNEE_GIRTH.value, Headers.CALF_GIRTH.value,
                Headers.AGE.value, Headers.FOREARM_GIRTH.value, Headers.KNEE_DIAM.value, Headers.CHEST_DEPTH.value,
                Headers.PELVIC_BREADTH.value, Headers.SHOULDER_GIRTH.value, Headers.CHEST_DIAM.value, Headers.BICEP_GIRTH.value ]
