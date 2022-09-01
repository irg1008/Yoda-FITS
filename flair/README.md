# Order of execution

Ejecuta run.py para ejecutar todo el proceso de preparación, entrenamiento y test.

o

Ejecuta los archivos individuales para cada paso.

## Problema

El problema como siempre es el dataset, ya que a poco que metemos un texto grande o hay más de una instancia de una clase, no se clasifica correctamente.


## Importante instalar las últimas versiones de optimum y transformers para incluir los modelos de mt5

Optimum: 
python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]

Transformers:
pip install git+https://github.com/huggingface/transformers 


## Importante

Los modelos de mt5 no funcionan con optimum porque no están añadidos, los modelos de Bart si funcionan perfectamente.

Vamos a ir tirando de estos de momento.