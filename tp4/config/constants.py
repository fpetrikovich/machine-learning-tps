from enum import Enum

class Headers(Enum):
    SEX = 'sex'
    AGE = 'age'
    CAD_DUR = 'cad.dur'
    CAD_DUR_GOOD = 'caddur'
    CHOLESTEROL = 'choleste'
    SIGDZ = 'sigdz'
    TVDLM = 'tvdlm'
    EXTRA_ID_HEADER = 'original id'

ALL_HEADERS = [Headers.SEX.value,Headers.AGE.value,Headers.CAD_DUR.value,Headers.CHOLESTEROL.value,Headers.SIGDZ.value,Headers.TVDLM.value]

class Similarity_Methods(Enum):
    MAX = 'max'
    MIN = 'min'
    AVG = 'avg'
    CENTROID = 'centroid'
