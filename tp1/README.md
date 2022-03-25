# TP1

## Instalar

Crear el entorno virtual con `virtualenv`:
```
virtualenv .env
```

Activar el entorno:
```
source .env/bin/activate
```

Instalar dependencias:
```
pip install -r requirements.txt
```

### Actualizar Dependencias

Para actualizar el `requirements.txt`:
```
pip freeze > requirements.txt
```

## Ejecutar

Para ejecutar se usa la siguiente llamada:
```
python main.py -p [1|2|3] -f PATH_AL_ARCHIVO_INPUT -m [analyze|solve] [-v] [-vv] [-k NUMERO] [-wc NUMERO]
```

- `-p` indica que ejercicio correr, el `1`, `2` o `3`
- `-m` sirve para el ejercicio 2
- `-v` corre en modo verbose, imprime cada resultado
- `-vv` corre en modo very verbose, imprime la matrix además de cada resultado
- `-k` corre en cross validation, con `k` cantidad de separaciones
- `-wc` define la cantidad de palabras significativas por categoria que se usan

### Ejemplos de Llamados

Ejemplos de los llamados:
```
python main.py -p 1 -f input/PreferenciasBritanicos.xlsx
python main.py -p 2 -f input/Noticias_argentinas.xlsx -m analyze
python main.py -p 2 -f input/Noticias_argentinas.xlsx -m solve
python main.py -p 2 -f input/Noticias_argentinas.xlsx -m solve -v
python main.py -p 2 -f input/Noticias_argentinas.xlsx -m solve -v -vv
```
