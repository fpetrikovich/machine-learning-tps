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
python main.py -p [1|2]
```

Parámetros:
- `-p` --> Ejercicio que se corre, el 1 o 2

Ejemplos de los llamados:
```
python main.py -p 1
```
