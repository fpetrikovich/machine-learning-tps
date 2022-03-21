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
python main.py -p [1|2|3] -f PATH_AL_ARCHIVO_INPUT -m [analyze|solve]
```

- `-p` indica que ejercicio correr, el `1`, `2` o `3`
- `-m` sirve para el ejercicio 2

### Ejemplos de Llamados

Ejemplos de los llamados:
```
python main.py -p 1 -f input/PreferenciasBritanicos.xlsx
python main.py -p 2 -f input/Noticias_argentinas.xlsx -m analyze
```
