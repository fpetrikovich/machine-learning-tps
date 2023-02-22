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

## Datasets

Hay varios datasets in `/input`:
- `movies_metadata.csv` dataset raw.
- `movies_metadata_good.csv` es el dataset limpio, con reemplazos.

## Ejecutar

Para ejecutar se usa la siguiente llamada:
```
python main.py -f path_al_archivo -p [b|cd|e]
```

Parámetros:
- `-p` --> Ejercicio que se corre, `means`, `hier`, `koho`.
- `-f` --> Path al archivo con los datos

Ejemplos de los llamados:
```
# Preprocessing del dataset
python preprocessing.py -f input/movies_metadata.csv -o input/movies_metadata_good.csv -analyze
# Mangling the dataset
python mangling.py -f input/movies_metadata_good.csv -o input/movies_metadata_mangle.csv -r 0.05 -m 0.1
# Kohonen
python main.py -f input/movies_metadata_good.csv -p koho -n 6 -it 100
# K-Medias
python main.py -f input/movies_metadata_good.csv -p means
# Agrupamiento Jerarquico
python main.py -f input/movies_metadata_good.csv -p hier
# Plotting
python main.py -f input/movies_metadata_good.csv -p plot
```
