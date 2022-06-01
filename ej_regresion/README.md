# TP0

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
python main.py -f PATH_AL_ARCHIVO_INPUT 
```

### Ejemplos de Llamados

Ejemplos de los llamados:
```
python main.py -f input/data.csv 
python main.py -f input/data.csv 
python main.py -f input/data.csv 
```
