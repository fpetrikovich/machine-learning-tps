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
- `acath_good.csv` tiene reemplazados con KNN los registros con malos datos
- `acath_good--train.csv` y `acath_good--test.csv` son el mejor conjunto de train/test que se obtuvo para el ejercicio B.
- `acath_good_sex--train.csv` y `acath_good_sex--test.csv` son el mejor conjunto de train/test que se obtuvo para el ejercicio CD.

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
# Corriendo el B
python main.py -f input/acath_good.csv -p b
# Corriendo el B with train/test
python main.py -f input/acath_good--train.csv -ftest input/acath_good--test.csv -p b
# Corriendo el CD
python main.py -f input/acath_good.csv -p cd
# Corriendo el CD with train/test
python main.py -f input/acath_good_sex--train.csv -ftest input/acath_good_sex--test.csv -p cd
```
