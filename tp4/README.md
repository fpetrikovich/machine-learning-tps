# TP4

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
python main.py -f path_al_archivo -p [b|cd|e]
```

Parámetros:
- `-p` --> Ejercicio que se corre, el b, cd o e
- `-f` --> Path al archivo con los datos

Ejemplos de los llamados:
```
# Preprocessing del dataset
python preprocessing.py -f input/acath.csv -o input/acath_good.csv -analyze
# Running the B item
python main.py -f input/acath_good.csv -p b
# Running the CD item
python main.py -f input/acath_good.csv -p cd
```
