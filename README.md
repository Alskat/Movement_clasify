# Movement_clasify

Dentro de el siguiente repo se encuentran 5 diferentes archivos, 2 librerías creadas para este proyecto, 1 librería con el modelo EEGnet y 2 jupyters de procesamiento.

Dentro de todo este procedimiento, se usó el dataser otorgado por Physionet https://www.physionet.org/content/eegmmidb/1.0.0/ 

Dentro de todo el repo, se encuentran: 

scripts/backend_dl.py: Donde se encuentran todas las funciones necesarias para el procesamiento de EEGnet, deep learning y funciones varias 

scripts/gui_backend: Tiene funciones en desuso, así como funciones para el procesamiento de datos directos del dataset para preprocesarlos, limpiarlos y ajustarlos para el modelo 

models/EEGModels.py: Modelo compacto EEGnet proporcionado por Larhern, V. https://iopscience.iop.org/article/10.1088/1741-2552/aace8c 

notebooks/procesamiento_DL.ipynb: Todo el preprocesamientro que se le aplica a las señales crudas antes de entrenar el modelo 

notebooks/entrenamiento_DL.ipynb: El procedimiento paso a paso para entrenar distintos modelos
