from enum import Enum

class Ex1_Headers(Enum):
    SCONES = 'scones'
    CERVERZA = 'cerveza'
    WHISKEY = 'wiskey'
    AVENA = 'avena'
    FUTBOL = 'futbol'
    NACIONALIDAD = 'Nacionalidad'

class Ex1_Nacionalidad(Enum):
    ENGLISH = 'I'
    SCOTTISH = 'E'

class Ex2_Headers(Enum):
    FECHA = 'fecha'
    TITULAR = 'titular'
    FUENTE = 'fuente'
    CATEGORIA = 'categoria'

class Ex2_Categoria(Enum):
    INTERNACIONAL = "Internacional"
    NACIONAL = "Nacional"
    DESTACADAS = "Destacadas"
    DEPORTES = "Deportes"
    SALUD = "Salud"
    CIENCIA_TECNOLOGIA = "Ciencia y Tecnologia"
    ENTRETENIMIENTO = "Entretenimiento"
    ECONOMIA = "Economia"
    NOTICIAS_DESTACADAS = "Noticias destacadas"

class Memory_Keys(Enum):
    FREQUENCIES = 1
    PROBABILITY = 2
    CLASS_PROBABILITY = 3
    KEY_FREQUENCIES = 4

class Ex2_Mode(Enum):
    SOLVE = "solve"
    ANALYZE = "analyze"

Ex2_Blacklist = {
    'para': True,
    'como': True,
    'Como': True,
    'cómo': True,
    'Cómo': True,
    'luego': True,
    'sobre': True,
    'tras': True,
    'luego': True,
    'pero': True,
    'ante': True,
    'entre': True,
    'durante': True,
    'saber': True,
    '2018:': True,
    '2018': True,
    '2019:': True,
    '2019': True,
    'desde': True,
    'donde': True,
    'hasta': True,
    'hizo': True,
    'este': True,
    'esta': True,
    'está': True,
    'tiene': True,
    'podría': True,
    'podrá': True,
    'dijo': True,
    # Eliminar números
    'tres': True,
    'cuatro': True,
    'cinco': True,
    'seis': True,
    'siete': True,
    'ocho': True,
    'nueve': True,
    'diez': True,
    # Nafta también está
    # 'naftas': True,
    # River y Boca tienen mejor frecuencia, al pedo tener esto
    # 'River-Boca:': True,
    # 'River-Boca': True,
    # Piel ya está, y después se va a ver con un includes
    # 'piel:': True,
    'dijo': True,
    'porque': True,
    'recién': True,
    'puede': True,
    'todo': True,
    'tenía': True,
    'tiene': True,
    'tipo': True,
    'volvió': True,
    'Tras': True,
    '5,8%': True,
    'podrían': True,
    'tenés': True,
    'llegaron': True,
    'cada': True,
    'tuvo': True,
    'dejó': True,
    'dejo': True,
    'entró': True,
    'entra': True,
    'paso': True,
    'antes': True,
    'quedó': True,
    'queda': True,
    'será': True,
    'llegó': True,
    'después': True,
    'días': True,
    'Desde': True,
    'cuando': True,
    'ella': True,
    'también': True
}

Ex2_Must_Have = ['tuberculosis', 'investigación']

class Ex3_Headers(Enum):
    ADMIT = 'admit'
    GRE = 'gre'
    GPA = 'gpa'
    RANK = 'rank'

class Ex3_Negated_Headers(Enum):
    ADMIT = 'no_admit'
    GRE = 'low_gre'
    GPA = 'low_gpa'

class Ex3_Ranks(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
