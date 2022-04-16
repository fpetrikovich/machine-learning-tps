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
python main.py -f PATH_TO_FILE -p [1|2] [-k NEIGHBORS] [-v] [-m [weighted|simple]] [-crossk BIN_NUMBER] [-sm [solve|analyze]]
```

Parámetros:
- `-f` --> Path al archivo de datos que se va a levantar
- `-p` --> Ejercicio que se corre, el 1 o 2
- `-k` --> Cantidad de neighbors base que se usan
- `-v` --> Modo verboso, imprime más información y gráficos
- `-m` --> Modo de KNN, simple o con los pesos
- `-crossk` --> Cantidad de bins para validación cruzada
- `-sm` --> Solve mode, `solve` para resolver y `analyze` para mostrar analysis y tratar de reemplazar los NaN
- `-eu` --> Example umbral, default 0
- `-gu` --> Gain umbral, default 0
- `-hl` --> Height limit, default 20
- `-sal` --> Split attribute limit, default None (how many attributes to randomly consider when finding the children)
- `-ss` --> Sample size in percentage of the training set for RF (default 0.9)
- `-ns` --> Number samples, how many trees to create in RF (default 10)

Ejemplos de los llamados:
```
python main.py -f input/german_credit.csv -p 3 -m solve -eu 3 -gu 0.01 -hl 4 -sal 6 -ss 0.5 -ns 20
python main.py -f input/german_credit.csv -p 1 -m solve
python main.py -f input/german_credit.csv -p 1 -crossk 5 -m solve -v
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -v -sm analyze
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -v -m weighted -sm solve
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -v -m simple -sm solve
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -crossk 5 -m weighted -sm solve
python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -crossk 5 -m simple -sm solve
```

El árbol del Ejercicio 1 puede generarse ejecutando:
```
dot -Tpdf tree.dot -o tree.pdf
```
