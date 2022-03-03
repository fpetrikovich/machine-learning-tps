from enum import Enum

class Headers(Enum):
    GRASAS = 'Grasas_sat'
    ALCOHOL = 'Alcohol'
    CALORIAS = 'Calorías'
    SEXO = 'Sexo'

class Modes(Enum):
    REMOVE = 'REMOVE'
    MEDIAN = 'MEDIAN'
    MEAN = 'MEAN'