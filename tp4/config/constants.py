from enum import Enum

class Headers(Enum):
    SEX = 'sex'
    AGE = 'age'
    CAD_DUR = 'cad.dur'
    CHOLESTEROL = 'choleste'
    SIGDZ = 'sigdz'
    TVDLM = 'tvdlm'
    EXTRA_ID_HEADER = 'original id'

class Similarity_Methods(Enum):
    MAX = 'max'
    MIN = 'min'
    AVG = 'avg'
    CENTROID = 'centroid'
