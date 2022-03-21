from enum import Enum

class Ex1_Headers(Enum):
    SCONES = 'scones'
    CERVERZA = 'cerveza'
    WHISKEY = 'wiskey'
    AVENA = 'avena'
    FUTBOL = 'futbol'
    NACIONALIDAD = 'Nacionalidad'

class Ex1_Nacionalidad(Enum):
    ENGLISH = 'E'
    IRISH = 'I'

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
    'naftas': True,
    # River y Boca tienen mejor frecuencia, al pedo tener esto
    'River-Boca:': True,
    'River-Boca': True,
    # Piel ya está, y después se va a ver con un includes
    'piel::': True,
    'dijo': True
}