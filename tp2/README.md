# TP2

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
python main.py -f PATH_TO_FILE -p [1|2] [-k NEIGHBORS] [-v] [-m [weighted|simple]] [-crossk BIN_NUMBER

Ejemplos de los llamados:
```
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -v -m weighted
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -v -m simple
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -crossk 5 -m weighted
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -crossk 5 -m simple
```
