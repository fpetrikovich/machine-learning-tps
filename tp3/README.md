# TP3

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
python main.py -p [1|2]
```

Parámetros:
- `-p` --> Ejercicio que se corre, el 1 o 2

Ejemplos de los llamados:
```
python main.py -p 1 -n 25 -s 5 -i 100 -m 4 -c 100
python main.py -p 2 -f pics/ -v -vv
# Validación cruzada en ejercicio 2 para división train/test
python main.py -p 2 -f pics/ -ker linear -c 1 -mode dataset -k 10
# Correr pruebas para la imagen y para otras imagenes
python main.py -p 2 -f pics/ -ker linear -c 0.01 -mode solve -v -vv
```
