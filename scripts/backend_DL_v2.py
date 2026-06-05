#Generador de clases para procesamiento

#Primero, importamos todas las librerías que ya teníamos antes 
#Clásicas
from pdb import run
from turtle import mode

import numpy as np
import pandas as pd

#Para directorios 
import sys
import os
import warnings
import json
from pathlib import Path
from datetime import datetime




#Librerías de sklearn y tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import clone_model

project_path = "G:\\Mi unidad\\Tesis v1\\Movement_clasify"

#Agregamos los modelos EEGnet a nuestro path
models_path = os.path.join(project_path, 'models')

if os.path.exists(models_path):
    if models_path not in sys.path:
        sys.path.append(models_path)
    
else:
    print("No se encontró directorio de modelos en", models_path)

#Personales
from EEGModels import EEGNet  

#Para evaluar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    
)

#0 Clase para almacenar los datos EEG de manera más fácil
class EEGSubset:
    def __init__(self, data_dict):
        self.X = data_dict["X"]
        self.y = data_dict["y"]
        self.sub = data_dict["sub"]
        self.trial = data_dict["trial"]
        self.run = data_dict["run"]
        self.mode = data_dict["mode"]

#1 Vamos a crear la clase donde procesaremos todos los datos con el pick, el dataset
class Selection:

    """La función de este nuevo objeto va a ser obtener los datos, pickear las clases que queremos, undersamplear
    la clase 0 (rest) si es necesario, y luego fusionar las clases motoras e imaginarias, para finalmente retornar
    los datos ya procesados, listos para el entrenamiento."""
    def __init__(self, pick=None, fusionar=True, random_state=42, Debug=False):
    
        self.fusionar = fusionar #Fusionar las clases mecánicas con las imaginarias
        self.random_state = random_state
        self.Debug = Debug

        config = self._determine_state(pick)

        self.pick = config["pick"]
        self.fusionar = config["fusionar"]
        self.binary = config["binary"]
        self.allowed_modes = config["allowed_modes"]

        #Ahora las los atributos que queremos que retorne 

        #Salidas principales 

        self.X = None 
        self.y = None 
        self.sub = None #Los sujetos totales elegidos

        #Salidas de parámetros 

        self.clases = None #(Nos determina cuales son las clases de acá mismo)
        self.n_classes = None #Número total de clases después del pick y la fusión
        self.class_counts = None #Conteo de cada clase después del pick y la fusión
        self.subjects = None #
        self.n_subjects = None
        self.input_shape = None
        self.label_map = None

        self.fine_subject = False

        #Ahora ejercemos nuestro pipeline, el cual era: pick, fusionar, luego después se elegirán los sujetos de entrenamiento

    def _determine_state(self, mode):

        if isinstance(mode, list):
            return {
                "pick": mode,
                "fusionar": self.fusionar,
                "binary": False,
                "allowed_modes": None
            }

        config_map = {

            # 🔹 FULL
            0: dict(pick=[0,1,2,3,4,5,6,7,8], fusionar=True, binary=False,
                    allowed_modes = ["LR_I", "HF_I", "LR_M", "HF_M"]),

            # 🔹 SOLO MOTOR
            1: dict(pick=[0,5,6,7,8], fusionar=True, binary=False,
                    allowed_modes = ["LR_M", "HF_M"]),

            # 🔹 SOLO IMAGINADO
            2: dict(pick=[0,1,2,3,4], fusionar=True, binary=False,
                    allowed_modes = ["LR_I", "HF_I"]),

            # 🔹 SIN FUSIÓN (9 clases)
            3: dict(pick=[0,1,2,3,4,5,6,7,8], fusionar=False, binary=False,
                    allowed_modes = ["LR_I", "HF_I", "LR_M", "HF_M"]),

            # 🔹 REST-IZQ-DER fusionado
            4: dict(pick=[0,1,2,5,6], fusionar=True, binary=False,
                    allowed_modes = ["LR_I", "LR_M"]),

            # 🔹 REST-MANOS-PIES fusionado
            5: dict(pick=[0,3,4,7,8], fusionar=True, binary=False,
                    allowed_modes = ["HF_I", "HF_M"]),

            # 🔹 REST-IZQ-DER SOLO MOTOR
            6: dict(pick=[0,5,6], fusionar=False, binary=False,
                    allowed_modes = ["LR_M"]),

            # 🔹 REST-MANOS-PIES SOLO MOTOR
            7: dict(pick=[0,7,8], fusionar=False, binary=False,
                    allowed_modes = ["HF_M"]),

            # 🔹 BINARIO
            8: dict(pick=[0,1,2,3,4,5,6,7,8], fusionar=True, binary=True,
                    allowed_modes = ["LR_I", "HF_I", "LR_M", "HF_M"]),

            # 🔹 BINARIO SOLO MOTOR
            9: dict(pick=[0,5,6,7,8], fusionar=False, binary=True,
                    allowed_modes = ["LR_M", "HF_M"]),

            # 🔹 MANOS-PIES FUSIONADO 
            10: dict(pick=[3,4,7,8], fusionar=True, binary=False,
                     allowed_modes = ["HF_I", "HF_M"]),

            # 🔹 MANOS-PIES MOTORES
            11: dict(pick=[7,8], fusionar=False, binary=False,
                     allowed_modes = ["HF_M"]),

            # 🔹 IZQ-DER FUSIONADOS 
            12: dict(pick=[1,2,5,6], fusionar=True, binary=False,
                     allowed_modes = ["LR_I", "LR_M"]),

            # 🔹 IZQ-DER SOLO MOTOR
            13: dict(pick=[5,6], fusionar=False, binary=False,
                     allowed_modes = ["LR_M"]),

            # 🔹 IZQ-DER-MANO-PIES FUSIONADAS 

            14: dict(pick=[1,2,3,4,5,6,7,8], fusionar=True, binary=False,
                     allowed_modes = ["LR_I", "LR_M", "HF_I", "HF_M"]),

            # 🔹 IZQ-DER-MANO-PIES MOTORAS 

            15: dict(pick=[5,6,7,8], fusionar=False, binary=False,
                     allowed_modes = ["LR_M", "HF_M"])

        }

        if mode not in config_map:
            raise ValueError(f"Modo {mode} no definido")

        return config_map[mode]
        
    def load(self, path):   

        #Creamos nuestro self.data acá

        data = np.load(path, allow_pickle=True)

        self.X = data["X"]
        self.y = data["y"]
        self.sub = data["sub"]
        self.run = data["run"]
        self.trial = data["trial"]
        self.mode = data["mode"]

        self.class_names_og = list(data["class_names"])
        self.channels_names = list(data["channel_names"])
        self.sfreq = float(data["sfreq"][0])

        self.windows = float(data["window_size"][0])
        self.window_step = float(data["window_step"][0])
        self.tmin = float(data["tmin"][0])
        self.tmax = float(data["tmax"][0])

        # reconstrucción automática
        self.n_channels = self.X.shape[1]
        self.n_samples = self.X.shape[0]

        print(f"✔️ Dataset cargado desde: {path}")

        self._pick()

        return self
    
    def _pick(self):
        X = self.X
        y = self.y
        sub = self.sub
        trial = self.trial
        run = self.run
        mode = self.mode

        assert len(X) == len(y) == len(sub) == len(trial) == len(run) == len(mode), "Inconsistencia en la longitud de los arrays"

        mask_class = np.isin(y, self.pick) #Elegimos una máscara con las clases que queremos

        if self.allowed_modes is not None:
            mask_mode = np.isin(mode, self.allowed_modes) #Elegimos una máscara con los modos que queremos
   
            mask = mask_class & mask_mode #Aplicamos ambas máscaras

            if self.Debug:
                print(f"✔ Aplicando filtro de modos: {self.allowed_modes}")
                print("Modos presentes antes del filtro:", np.unique(mode))
                print("Modos presentes después del filtro:", np.unique(mode[mask_mode]))
        else:
            mask = mask_class

        #Aplicamos la máscara (Evaluar después en el notebook)
        self.X = X[mask]
        self.y = y[mask]
        self.sub = sub[mask]
        self.trial = trial[mask]
        self.run = run[mask]
        self.mode = mode[mask]

        if self.X.shape[0] == 0:
            raise ValueError(
                f"No hay muestras para pick={self.pick} "
                f"con allowed_modes={self.allowed_modes}"
            )

        if self.Debug:
            print("✔ Pick aplicado en load()")
            print("Clases:", np.unique(self.y))
            print("Modes:", np.unique(self.mode))
            print("Runs:", np.unique(self.run))
            print("X:", self.X.shape)
            print("Canales disponibles:" , self.channels_names)

        assert len(self.X) == len(self.y) == len(self.sub) == len(self.trial) == len(self.run) == len(self.mode), "Inconsistencia en la longitud de los arrays finales!"
    
    def pick_fine(self, subject_id, test_run=None, run_split=False):

        mask_model = self.sub != subject_id
        mask_fine  = self.sub == subject_id


        data_model = {
            "X": self.X[mask_model],
            "y": self.y[mask_model],
            "sub": self.sub[mask_model],
            "trial": self.trial[mask_model],
            "run": self.run[mask_model],
            "mode": self.mode[mask_model]
        }

        data_fine = {
            "X": self.X[mask_fine],
            "y": self.y[mask_fine],
            "sub": self.sub[mask_fine],
            "trial": self.trial[mask_fine],
            "run": self.run[mask_fine],
            "mode": self.mode[mask_fine]
        }

        self.data_model = EEGSubset(data_model)
        self.data_fine = EEGSubset(data_fine)

        self.fine_subject = subject_id

        if not run_split:

            if self.Debug:
                print(f"✔ Sujetos entrenamiento: {np.unique(data_model['sub'])}")
                print(f"✔ Sujeto fine-tune: {subject_id}")
                print(f"Shape modelo: {data_model['X'].shape}")
                print(f"Shape fine: {data_fine['X'].shape}")

            return self.data_model, self.data_fine, None

        else: 
            #Vamos a escoger un único run para el test
            unique_runs = np.unique(data_fine["run"])
            if len(unique_runs) < 2:
                raise ValueError(
                    f"El sujeto {subject_id} tiene menos de 2 runs. "
                    "No se puede separar fine-tuning y test por run."
                )
            if test_run is None:
                test_run = unique_runs[0]  #Elegimos el primer run como test por defecto

            if test_run not in unique_runs:
                raise ValueError(
                    f"test_run={test_run} no existe para el sujeto {subject_id}. "
                    f"Runs disponibles: {unique_runs}"
                )
            mask_test = data_fine["run"] == test_run
            mask_fine_train = ~mask_test
            
            data_fine_train = {
                "X": data_fine["X"][mask_fine_train],
                "y": data_fine["y"][mask_fine_train],
                "sub": data_fine["sub"][mask_fine_train],
                "trial": data_fine["trial"][mask_fine_train],
                "run": data_fine["run"][mask_fine_train],
                "mode": data_fine["mode"][mask_fine_train]
}
            data_test = {
                "X": data_fine["X"][mask_test],
                "y": data_fine["y"][mask_test],
                "sub": data_fine["sub"][mask_test],
                "trial": data_fine["trial"][mask_test],
                "run": data_fine["run"][mask_test],
                "mode": data_fine["mode"][mask_test]}
            
            

            self.data_model = EEGSubset(data_model)
            self.data_fine = EEGSubset(data_fine_train)
            self.data_test = EEGSubset(data_test)

            self.fine_subject = subject_id
            self.test_run = test_run

            if self.Debug:
                print(f"✔ Sujetos entrenamiento general: {np.unique(self.data_model.sub)}")
                print(f"✔ Sujeto fine-tune: {subject_id}")
                print(f"✔ Run reservada para test: {test_run}")
                print(f"Runs fine-tune:", np.unique(self.data_fine.run))
                print(f"Runs test:", np.unique(self.data_test.run))
                print(f"Shape modelo: {self.data_model.X.shape}")
                print(f"Shape fine: {self.data_fine.X.shape}")
                print(f"Shape test: {self.data_test.X.shape}")

      

            return self.data_model, self.data_fine, self.data_test
    
    def resume_dataset(self, subs=False):
        print("\n" + "="*50)
        print("🧠 RESUMEN DEL DATASET EEG")
        print("="*50)

        # Estado
        if self.X is None:
            print("❌ Dataset no construido aún.")
            return
        
        #Sujetos
        print("\n👤 Sujetos únicos:")
        print(f"Total sujetos: {len(np.unique(self.sub))}")
        if subs:
            unique_sub, counts_sub = np.unique(self.sub, return_counts=True)
            for s, c in zip(unique_sub, counts_sub):
                print(f"Sujeto {s:03d}: {c} muestras")

        if self.fine_subject:
            print("\n Sujeto elegido para fine-tuning:", self.fine_subject)

        # Dimensiones
        print("\n📐 Dimensiones:")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"sub shape: {self.sub.shape}")
        print(f"trial shape: {self.trial.shape}")
        print(f"run shape: {self.run.shape}")
        print(f"Total muestras (ventanas): {self.X.shape[0]}")
        print(f"Canales: {self.X.shape[1]}")
        print(f"Muestras por ventana: {self.X.shape[2]}")
        print(f"Modos: {np.unique(self.mode)}")

        # Frecuencia
        print("\n⏱ Frecuencia de muestreo:")
        print(f"{self.sfreq} Hz")

        # Canales
        print("\n🧬 Canales:")
        print(f"Número de canales: {len(self.channels_names)}")
        print(f"Lista: {self.channels_names}")

        # Clases
        print("\n🧪 Clases:")
        print(f"Número de clases: {len(self.class_names_og)}")
        print(f"Lista: {self.class_names_og}")

        # Distribución de clases
        print("\n📊 Distribución de clases:")
        unique, counts = np.unique(self.y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"{self.class_names_og[u]:10s}: {c}")

        # Sujetos
        print("\n👤 Sujetos:")
        unique_sub, counts_sub = np.unique(self.sub, return_counts=True)
        print(f"Total sujetos: {len(unique_sub)}")
        for s, c in zip(unique_sub, counts_sub):
            print(f"Sujeto {s:03d}: {c} muestras")

        # Runs
        print("\n🎮 Runs:")
        unique_run, counts_run = np.unique(self.run, return_counts=True)
        for r, c in zip(unique_run, counts_run):
            print(f"Run {r:02d}: {c} muestras")

        # Trials
        print("\n🔁 Trials:")
        unique_trials = np.unique(self.trial)
        print(f"Total trials únicos: {len(unique_trials)}")

        print("\n" + "="*50)

    def pick_ch(self, selected_channels): 
        """Esta función selecciona los canales que queremos usar, 
        basándose en la lista de nombres de canales que tenemos en self.channels_names"""

        if self.channels_names is None:
            raise ValueError("No hay channel_names cargados")

        channel_to_idx = {ch: i for i, ch in enumerate(self.channels_names)}

        missing = [ch for ch in selected_channels if ch not in channel_to_idx]
        if missing:
            raise ValueError(f"Canales no encontrados: {missing}")

        idx = [channel_to_idx[ch] for ch in selected_channels]

        

        self.X = self.X[:, idx, :]
        self.channels_names = [self.channels_names[i] for i in idx]

        # actualizar metadata
        self.n_channels = len(self.channels_names)

        if self.Debug:
            print(f"✔ Canales seleccionados: {self.channels_names}")
            print(f"Nuevo shape X: {self.X.shape}")
            
    def pipeline(self, n=None, undersample_rest=None, binary = False):
        #El pipeline principal consiste en aplicar un pick y luego una fusión, cosas que ya tenemos en otras clases 
        #n sólo es para limitar la cantidad de datos totales, no recuerdo porque lo quise añadir pero ahí está
        if self.binary is not None:
            binary = self.binary
        # Paso 1: Pickeo 
        X = self.X
        y = self.y
        sub = self.sub
        trial = self.trial
        run = self.run
        mode = self.mode

        # Paso 2: Elegir un máximo del conjunto
        total = len(y)
        if n is not None and n < total:
            if self.Debug:
                print(f"✔ Aplicando submuestreo adicional: limitando a los primeros {n} de {total} datos. . . ")
            X = X[:n]
            y = y[:n]
            sub = sub[:n]
            trial = trial[:n]
            run = run[:n]
            mode = mode[:n]
            print(f"Seleccionando los primeros {n} de {total} datos tras el filtrado.")

        # --- 2.5) En caso de querer hacer clases binarias

        if binary:
            if self.Debug:
                print("✔ Activando modo binario (rest vs no_rest)")

            # 0 = rest, todo lo demás = 1
            y_final = np.where(y == 0, 0, 1).astype(np.int32)

            class_names_final = ["rest", "no_rest"]

            self.clases = pd.DataFrame({
                "codigo": [0, 1],
                "nombre": class_names_final
            })

            if self.Debug:
                print(dict(zip(*np.unique(y_final, return_counts=True))))

            self._build(X, y_final, sub, trial, run, mode)

            return  

        # --- 3) fusionar clases y remapear las etiquetas--- 

        if self.fusionar:
            y_final, class_names_final = self.semantic_fusion(y)

            if self.Debug:
                print("✔ Clases fusionadas semánticamente.")
                print("Clases finales:", class_names_final)
                print(dict(zip(*np.unique(y_final, return_counts=True))))

        else:
            unique_sorted = sorted(np.unique(y))
            mapa = {old: new for new, old in enumerate(unique_sorted)}

            y_final = np.array([mapa[val] for val in y], dtype=np.int32)
            class_names_final = [self.class_names_og[old] for old in unique_sorted]

            if self.Debug:
                print("✔ Sin fusión. Solo remapeo.")
                print("Clases finales:", class_names_final)
                print(dict(zip(*np.unique(y_final, return_counts=True))))

        self.clases = pd.DataFrame({
            "codigo": list(range(len(class_names_final))),
            "nombre": class_names_final
        })
                        

        #Llevar a construir el dataset final

        self._build(X, y_final, sub, trial, run, mode) #Ahora construimos los atributos principaples

        


    
    def semantic_fusion(self, y): #Función bella toda bonita no la toquen
        """
        Fusiona clases usando las etiquetas originales.
        Si fusionar=False, conserva nombres originales.
        Si fusionar=True, fusiona pares imaginario/motor SOLO si ambos existen.
        """

        present = set(np.unique(y))

        pair_map = {
            (1, 5): "right",
            (2, 6): "left",
            (3, 7): "hands",
            (4, 8): "feet",
        }

        label_to_name = {}

        # Rest siempre igual
        if 0 in present:
            label_to_name[0] = "rest"

        used = {0}

        for pair, fused_name in pair_map.items():
            a, b = pair

            if a in present and b in present:
                label_to_name[a] = fused_name
                label_to_name[b] = fused_name
                used.update([a, b])
            else:
                if a in present:
                    label_to_name[a] = self.class_names_og[a]
                    used.add(a)
                if b in present:
                    label_to_name[b] = self.class_names_og[b]
                    used.add(b)

        # Cualquier clase no contemplada queda con su nombre original
        for cls in present:
            if cls not in used:
                label_to_name[cls] = self.class_names_og[cls]

        # Crear nombres únicos en el orden de aparición por clase original
        semantic_names = []
        name_to_new = {}

        for cls in sorted(present):
            name = label_to_name[cls]
            if name not in name_to_new:
                name_to_new[name] = len(semantic_names)
                semantic_names.append(name)

        y_semantic = np.array([name_to_new[label_to_name[val]] for val in y], dtype=np.int32)

        

        return y_semantic, semantic_names
    

    


    def _build(self, X, y, sub, trial, run, mode):
        self.X = X
        self.y = y
        self.sub = sub  
        self.trial = trial
        self.run = run
        self.mode = mode
        

        
        
        self.n_classes = len(np.unique(y)) #Número total de clases totales
        self.class_counts = pd.Series(y).value_counts().sort_index() #Conteo de cada clase después del pick y la fusión
        self.subjects = np.unique(sub) #Los sujetos totales elegidos
        self.n_subjects = len(self.subjects) #Número de sujetos totales elegidos (No creo que se utilice mucho)
        self.input_shape = X.shape[1:]   #La forma de entrada para el modelo, que es el número de canales y muestras por ventana
        self.label_map = dict(zip(self.clases["codigo"], self.clases["nombre"])) #Un diccionario que mapea el código de clase al nombre de clase, útil para interpretación y visualización
    

            
    def resume_data(self):
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"sub shape: {self.sub.shape}")
        print(f"trial shape: {self.trial.shape}")
        print(f"run shape: {self.run.shape}")
        print(f"Número de clases: {self.n_classes}")
        print(f"Número de sujetos: {self.n_subjects}")
        print("\nClases:")
        print(self.clases)
        print("\nConteo por clase:")
        print(self.class_counts)

#Bien, ahora que ya tenemos el pickeo y balanceo, vamos con todo el entrenamiento 

#0) creamos clases para almacenar nuestros datos más a futuro
class XYGroup:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test
        


class DataSplit:
    def __init__(self, X_train, X_val, X_test, 
                y_train, y_val, y_test, 
                sub_train=None, sub_val=None, sub_test=None, 
                trial_train=None, trial_val=None, trial_test=None, 
                run_train=None, run_val=None, run_test=None,
                mode_train=None, mode_val=None, mode_test=None):
        self.X = XYGroup(X_train, X_val, X_test)
        self.y = XYGroup(y_train, y_val, y_test)
        self.sub = XYGroup(sub_train, sub_val, sub_test)
        self.trial = XYGroup(trial_train, trial_val, trial_test)
        self.run = XYGroup(run_train, run_val, run_test)
        self.mode = XYGroup(mode_train, mode_val, mode_test)


#1) Splitear los datos 

def split_eeg(split,
            test_size=0.2, val_size=0.1, make_test = False,
            mode = None, debug=False): #chatgpt me puso esta función más bonita 
    X = split.X
    y = split.y
    sub = split.sub
    trial = split.trial
    run = split.run
    modes = split.mode

    assert len(X) == len(y) == len(sub) == len(trial) == len(run) == len(modes), "X, y, sub, trial y run deben tener la misma longitud!"

    unique_subj = np.unique(sub)

    
    #Decidimos el modo de split
    if mode is None and len(unique_subj) >= 10: #Modo default para datasets grandes, split por sujeto para evitar data leakage
        mode = "subject"
        group = sub
        if debug:
            print("~~Modo de split por sujeto")
    elif mode is None and len(unique_subj) < 10: #Modo default para datasets pequeños, split por trial para asegurar que haya suficientes sujetos en cada split
        mode = "trial"
        if len(np.unique(trial)) < 3:
            raise ValueError("No hay suficientes trials para hacer un split por trial. Considera usar otro modo de split o agregar más trials.")
        group = trial
        if debug: 
            print("~~Modo de split por trial")

    elif mode == "subject":
        if len(unique_subj) < 2:
            raise ValueError("No hay suficientes sujetos para hacer un split por sujeto. Considera usar otro modo de split o agregar más sujetos.")
        group = sub
    elif mode == "trial":
        if len(np.unique(trial)) < 3:
            raise ValueError("No hay suficientes trials para hacer un split por trial. Considera usar otro modo de split o agregar más trials.")
        group = trial
    elif mode == "run":
        if len(np.unique(run)) < 3:
            raise ValueError("No hay suficientes runs para hacer un split por run. Considera usar otro modo de split o agregar más runs.")
        group = run
    else: 
        raise ValueError(f"Modo de split desconocido: {mode}. Opciones válidas: 'subject', 'trial', 'run' o None (auto).")
    if debug:
            

            print(f"Modo de split: {mode}")



    #Split universal
    split = _split_by_group(
        X, y, sub, trial, run, group, modes, make_test=make_test,
        test_size=test_size,
        val_size=val_size,
        debug=debug
    )

    # 🧠 INFO
    split_info = {
        "mode": mode,
        "n_subjects": len(unique_subj),
        "n_train": len(split.y.train),
        "n_val": len(split.y.val),
        "n_test": len(split.y.test),
    }

    if debug:
        print("Modo split:", mode)
        print("X_train:", split.X.train.shape)
        print("X_val:", split.X.val.shape)
        print("X_test:", split.X.test.shape)

    clases_train = set(np.unique(split.y.train))
    clases_total = set(np.unique(y))
    clases_faltantes = clases_total - clases_train
    if clases_faltantes:
        import warnings
        warnings.warn(
            f"⚠️ Clases {clases_faltantes} no están en el train set. "
            f"Considera cambiar el random_state o el modo de split."
        )
    

    return split, split_info

def _split_by_group(X, y, sub, trial, run, group, modes, make_test = False, test_size=0.1, val_size=0.1, random_state=42, debug=False): #Para qué nos vamos a mentir, yo hice el código pero Claude lo optimizó

    # Validación
    assert len(X) == len(y) == len(group), "X, y y group deben tener mismo largo"
    if group is None:
        raise ValueError("group no puede ser None en _split_by_group (No debería salir nunca este eror igual :P)")

    # 1. Grupos únicos
    unique_groups = np.unique(group)

    n_groups = len(unique_groups)

    #Creamos el valor mínimo de reserva para test y val, para evitar quedarnos sin grupos en alguno de los splits
    if make_test:
        if n_groups < 4:
            raise ValueError("Para train/val/test por grupo necesitas al menos 4 grupos.")

        n_test = max(1, int(round(n_groups * test_size)))
        n_val  = max(1, int(round(n_groups * val_size)))

        if n_test + n_val >= n_groups:
            n_test = 1
            n_val = 1

        n_temp = n_test + n_val
    else:
        if n_groups < 2:
            raise ValueError("Para train/val por grupo necesitas al menos 2 grupos.")

        n_temp = max(1, int(round(n_groups * val_size)))
        if n_temp >= n_groups:
            n_temp = 1

    if debug:
        print("n_groups:", n_groups)
        print("n_temp:", n_temp)


    # 2. Split train vs temp
    train_groups, temp_groups = train_test_split(
        unique_groups,
        test_size=n_temp,
        random_state=random_state
    )

    #En caso de aceptar el split_val, entonces hacemos un split adicional para separar val y test, si no, dejamos todo en val
    if make_test:
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=n_test,
            random_state=random_state
        )
    else:
        val_groups = temp_groups
        test_groups = np.array([])

    # 4. Máscaras
    mask_train = np.isin(group, train_groups)
    mask_val   = np.isin(group, val_groups)
    mask_test  = np.isin(group, test_groups)
    

    # 5. Aplicar
    X_train, y_train, g_train = X[mask_train], y[mask_train], group[mask_train]
    X_val, y_val, g_val       = X[mask_val], y[mask_val], group[mask_val]
    X_test, y_test, g_test    = X[mask_test], y[mask_test], group[mask_test]

    sub_train, trial_train, run_train, modes_train = sub[mask_train], trial[mask_train], run[mask_train], modes[mask_train]
    sub_val, trial_val, run_val, modes_val       = sub[mask_val], trial[mask_val], run[mask_val], modes[mask_val]
    sub_test, trial_test, run_test, modes_test    = sub[mask_test], trial[mask_test], run[mask_test], modes[mask_test]

    if debug:
        print("Split por grupo:")
        print(f"  Train: {len(X_train)} muestras, grupos: {np.unique(g_train)}")
        print(f"  Val:   {len(X_val)} muestras, grupos: {np.unique(g_val)}")
        print(f"  Test:  {len(X_test)} muestras, grupos: {np.unique(g_test)}")

    return DataSplit(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        sub_train, sub_val, sub_test,
        trial_train, trial_val, trial_test,
        run_train, run_val, run_test,
        modes_train, modes_val, modes_test
    )


    
    
#2) Normalización

def apply_undersample_rest(split): 
        X = split.X.train
        y = split.y.train
        sub = split.sub.train
        trial = split.trial.train
        run = split.run.train
        mode = split.mode.train


        rest_label = 0

        idx_rest = np.where(y == rest_label)[0]
        idx_non_rest = np.where(y != rest_label)[0]

        unique_classes = np.unique(y)

        counts_others = []
        for cls in unique_classes:
            if cls == rest_label:
                continue
            counts_others.append(np.sum(y == cls))

        if counts_others:
            max_other = max(counts_others)

            if len(idx_rest) > max_other:
                rng = np.random.default_rng(42)
                idx_rest_sel = rng.choice(idx_rest, size=max_other, replace=False)

                idx_keep = np.concatenate([idx_rest_sel, idx_non_rest])
                rng.shuffle(idx_keep)

                X = X[idx_keep]
                y = y[idx_keep]
                sub = sub[idx_keep]
                trial = trial[idx_keep]
                run = run[idx_keep]
                mode = mode[idx_keep]

                print(f"Undersampling: clase 0 recortada de {len(idx_rest)} a {max_other} muestras.")



        return X, y, sub, trial, run, mode


def normalizar_por_canal(X, mean, std, eps=1e-6): #Normalizar por canal, es decir, restar la media y dividir por la desviación estándar de cada canal, para cada muestra
        # X: (N, 8, 160)
        # mean, std: (8,)
        return (X - mean[None, :, None]) / (std[None, :, None] + eps)


def calc_stats(X):
    #Calcular desviación y media por canal, sólo lo haremos para el dato de entrenamiento
    
    channel_mean = X.mean(axis=(0, 2))  
    channel_std  = X.std(axis=(0, 2))   


    return channel_mean, channel_std

def normalizar_split(split, eps=1e-6):
    mean, std = calc_stats(split.X.train)

    X_train = normalizar_por_canal(split.X.train, mean, std, eps)
    X_val   = normalizar_por_canal(split.X.val, mean, std, eps)
    X_test  = normalizar_por_canal(split.X.test, mean, std, eps)

    norm_split = DataSplit(
        X_train, X_val, X_test,
        split.y.train, split.y.val, split.y.test,
        split.sub.train, split.sub.val, split.sub.test,
        split.trial.train, split.trial.val, split.trial.test,
        split.run.train, split.run.val, split.run.test,
        split.mode.train, split.mode.val, split.mode.test
    )

    return norm_split, mean, std
#3 Ajustar la forma de los datos para Keras
def ajustar_keras(X):
    return X[:, :, :, None]


def prepare_keras_split(split):
    return DataSplit(
        ajustar_keras(split.X.train),
        ajustar_keras(split.X.val),
        ajustar_keras(split.X.test),
        split.y.train,split.y.val,split.y.test,
        split.sub.train,split.sub.val,split.sub.test,
        split.trial.train,split.trial.val,split.trial.test,
        split.run.train,split.run.val,split.run.test,
        split.mode.train,split.mode.val,split.mode.test
    )  

#4 Construir modelo EEGnet
def build_eegnet(classes, chans, signal_len,
                 dropout_rate=0.5, kern_length=64,
                 F1=8, D=2, norm_rate=0.25,
                 dropout_type='Dropout'):
    
    #En esta construimos nuestro modelo EEGnet
    F2 = F1 * D
    model = EEGNet(
        nb_classes=classes,
        Chans=chans,
        Samples=signal_len,
        dropoutRate=dropout_rate,
        kernLength=kern_length,
        F1=F1,
        D=D,
        F2=F2,
        norm_rate=norm_rate,
        dropoutType=dropout_type
    )
    return model

def eeg_train(data_obj, mode="subject", test_size=0.1, val_size=0.1, classes=None, epochs=20, 
              debug=False, use_class_weight=True, sfreq=160, kern_length=None, F1 = 8, make_test=False, batch = 16,
              undersample_rest=False, test_run=None, checkpoint_path = None,
              Lr =1e-3, dynamic_lr = False, Lr_grid = None, patience_level = "medium", monitor = "val_loss", dropout_rate=0.5):
    
    #0) Definir variables
    monitor_mode = "min" if monitor.endswith("loss") else "max"
    if debug:
        print("Iniciando pipeline de entrenamiento EEGNet...")  
        print(f"Split de test: {make_test}")
    if classes is None:
        classes = len(np.unique(data_obj.y))

    if dynamic_lr and Lr_grid is None: 
        
        warnings.warn("dynamic_lr=True pero no se proporcionó Lr_grid. Se usará un schedule por defecto.")


    #1 Split 
    split, info = split_eeg(data_obj, mode = mode, test_size=test_size, val_size=val_size, debug=debug, make_test=make_test)

    #2 Normalización: importante sólo se normaliza X

    norm_split, channel_mean, channel_std = normalizar_split(split)

    #3 Ajustar forma para Keras (Sólo X también)

    keras_split = prepare_keras_split(norm_split)

    if debug: 
        print("Antes de ajustar:", split.X.train.shape, "Después de ajustar para Keras:", keras_split.X.train.shape)

    #3.5 Undersampling de clase rest en caso de que se desee (Sólo para el train, no queremos tocar val ni test)

    if undersample_rest:
        X_train, y_train, _, _, _, _ = apply_undersample_rest(keras_split)
        
    else: 
        X_train = keras_split.X.train
        y_train = keras_split.y.train #El resto ni los ocupamos 

    if debug:
        print("distribución de clases después de undersample rest:", dict(zip(*np.unique(y_train, return_counts=True))))



    #4 Construir modelo EEGnet
    chans = X_train.shape[1]
    signal_len = X_train.shape[2]

    kern_length = int(sfreq // 2) if kern_length is None else kern_length
    
    modelo = build_eegnet(classes, chans, signal_len, kern_length=kern_length, F1=F1, dropout_rate=dropout_rate)

    #5) Compilar el modelo
    modelo.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=Lr),
        metrics=['accuracy']
    )

    callbacks = []

    early_cfg = get_early_stopping_config(
        level=patience_level,
        monitor=monitor
    )

    if early_cfg is not None:
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg["monitor"],
                patience=early_cfg["patience"],
                min_delta=early_cfg["min_delta"],
                restore_best_weights=early_cfg["restore_best_weights"],
                mode = monitor_mode
            )
        )
    if checkpoint_path is None:
        checkpoint_path = "EEGNet_best.keras"
    ckpt_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor=monitor,
    save_best_only=True,
    mode=monitor_mode
)
    callbacks.append(ckpt_cb)

    
    

    if dynamic_lr:
        if Lr_grid is None:
            Lr_grid = [(0, Lr), (int(epochs * 0.3), Lr * 0.1), (int(epochs * 0.6), Lr * 0.01)]

        callbacks.append(
            LearningRateScheduler(
                make_lr_scheduler(Lr_grid, verbose=debug),
                verbose=0
            )
        )

    #6 Entrenar el modelo 

    if debug:
        verbose = 1
    else:
        verbose = 0
    class_weight = None
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        classes_present = np.unique(y_train)
        weights = compute_class_weight('balanced',
                                       classes=classes_present,
                                       y=y_train)
        class_weight = dict(zip(classes_present, weights))


    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch,          
        validation_data=(keras_split.X.val, keras_split.y.val),
        callbacks=callbacks,
        class_weight=class_weight, 
        verbose=verbose
    )

    

    #7 Evaluar el modelo
    #Si existe un subconjunto de test reservado, make_test = True
    if make_test:

        test_loss, test_acc = modelo.evaluate(keras_split.X.test, keras_split.y.test, verbose=0)
        print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    #Si no, pero le especificamos un test_run, entonces evaluamos en ese test_run (que se supone que es un run completo separado del train y val)
    elif test_run is not None: 
        #Normalizamos y ajustamos el test_run para Keras 
        if debug:
            print(f"Test run reservado. Se realizará evaluación en el set de test.")
            print(f"Antes de normalizar y ajustar, test_run.X shape: {test_run.X.shape}")

        X_test = normalize_run(test_run, channel_mean, channel_std)

        if debug:
            print(f"Después de normalizar y ajustar, test_run.X shape: {X_test.shape}")
        keras_split.X.test = X_test
        keras_split.y.test = test_run.y
        keras_split.sub.test = test_run.sub
        keras_split.trial.test = test_run.trial
        keras_split.run.test = test_run.run
        keras_split.mode.test = test_run.mode

        
        test_loss, test_acc = modelo.evaluate(X_test, test_run.y, verbose=0)
        print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    
    #En caso de que no haya ruta de validación ni split, validamos con el subconjunto val
    else: 
        print("Test set no reservado, se realizará evaluación en el set de validación.")
        test_loss, test_acc = modelo.evaluate(keras_split.X.val, keras_split.y.val, verbose=0)
        print(f"Validation loss={test_loss:.4f}, acc={test_acc:.4f}")

    monitor_values = history.history[monitor]  # o early_monitor

    if monitor.endswith("loss"):
        best_epoch_idx = int(np.argmin(monitor_values))
    else:
        best_epoch_idx = int(np.argmax(monitor_values))
    best_value = monitor_values[best_epoch_idx]

    info["best_epoch"] = best_epoch_idx
    info["best_epoch"]         = best_epoch_idx
    info["best_monitor_value"] = float(monitor_values[best_epoch_idx])
    info["epochs_ran"]         = len(monitor_values)
    info["monitor"]            = monitor
    info["dynamic_lr"] = dynamic_lr
    info["Lr_grid"] = Lr_grid
    info["initial_lr"] = Lr
    info["batch_size"] = batch
    info["epochs_requested"] = epochs
    info["best_epoch_idx"] = best_epoch_idx
    info["best_monitor_value"] = best_value
    info["epochs_ran"] = len(history.history["loss"])


    

    return modelo, history, channel_mean, channel_std, keras_split, info

def make_lr_scheduler(lr_schedule, verbose=True):
    """
    Recibe una lista de tuplas indicando a partir de qué época se debe cambiar el learning rate
    """

    lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
    last_lr = {"value": None}

    def scheduler(epoch, current_lr):
        lr = lr_schedule[0][1]

        for start_epoch, scheduled_lr in lr_schedule:
            if epoch >= start_epoch:
                lr = scheduled_lr
            else:
                break

        if verbose and last_lr["value"] != lr:
            print(f"[LR Scheduler] Epoch {epoch + 1}: learning rate = {lr}")
            last_lr["value"] = lr

        return lr

    return scheduler


def get_early_stopping_config(level="medium", monitor="val_loss"):
    """
    Presets simples para EarlyStopping.

    level:
        "low"    -> paciencia baja, entrenamiento más corto
        "medium" -> balanceado
        "high"   -> paciencia alta, entrenamiento más largo
        "off"    -> sin early stopping
    """

    presets = {
        "low": {
            "patience": 8,
            "min_delta": 1e-3,
            "monitor": monitor,
            "restore_best_weights": True
        },
        "medium": {
            "patience": 15,
            "min_delta": 5e-4,
            "monitor": monitor,
            "restore_best_weights": True
        },
        "high": {
            "patience": 25,
            "min_delta": 1e-4,
            "monitor": monitor,
            "restore_best_weights": True
        }
    }

    if level == "off":
        return None

    if level not in presets:
        raise ValueError(
            f"early_stop_level='{level}' no válido. "
            "Usa 'low', 'medium', 'high' u 'off'."
        )

    return presets[level]

def eeg_fine(base, dataset_fine, mean = None, std = None, epochs = 20, debug =False, test_size=0.2, val_size=0.1,
              unfreeze_last_n=16, lr=5e-4, use_class_weight=True, batch_size=32, 
              normalize=True, undersample_rest=False, test_run=None, make_test=False,
              dynamic_lr=False, Lr_grid=None, dynamic_unfreeze=False, unfreeze_grid =None,
              patience_level="medium", monitor="val_loss",
              model_checkpoint = None, #Para un futuro gridsearch

              ):

    monitor_mode = "min" if monitor.endswith("loss") else "max"

    #Paso 1: split 

    if(len(np.unique(dataset_fine.sub)) != 1):
        warnings.warn("El dataset de fine-tuning contiene más de un sujeto. Asegúrate de que sólo haya un sujeto para evitar data leakage")
    
    if test_run is None and make_test is False: 
        raise ValueError("Si no se va a reservar un test set, entonces test_run no puede ser None. Considera reservar un test set o especificar test_run")

    split_fine, info = split_eeg(dataset_fine, mode = "trial", test_size=test_size, val_size=val_size, debug=debug, make_test=make_test )

    if debug: 
        print("Tamaño del train:", split_fine.X.train.shape, "Tamaño del val:", split_fine.X.val.shape, "Tamaño del test:", split_fine.X.test.shape)

    #Paso 2: Normalización con media y desviación dada (si no se da, se calcula con el mismo método que antes)

    

    if normalize:
        if mean is None or std is None:
            print("Advertencia: Se normaliza con los datos del usuario de ajuste")
            norm_split, mean, std = normalizar_split(split_fine)
        else:
            X_train = normalizar_por_canal(split_fine.X.train, mean, std)
            X_val = normalizar_por_canal(split_fine.X.val, mean, std)
            X_test = normalizar_por_canal(split_fine.X.test, mean, std)

            norm_split = DataSplit(
                X_train, X_val, X_test,
                split_fine.y.train, split_fine.y.val, split_fine.y.test,
                split_fine.sub.train, split_fine.sub.val, split_fine.sub.test,
                split_fine.trial.train, split_fine.trial.val, split_fine.trial.test,
                split_fine.run.train, split_fine.run.val, split_fine.run.test,
                split_fine.mode.train, split_fine.mode.val, split_fine.mode.test
            )
    else:
        norm_split, mean, std = normalizar_split(split_fine)

    

        


    #Paso 3: Ajustar forma para Keras

    keras_split_fine = prepare_keras_split(norm_split)

    #Paso 3.5: Undersampling de clase rest en caso de que se desee (Sólo para el train, no queremos tocar val ni test)

    if undersample_rest:
        X_fine_train, y_fine_train, sub_fine_train, trial_fine_train, run_fine_train, mode_fine_train = apply_undersample_rest(keras_split_fine)
    else: 
        X_fine_train = keras_split_fine.X.train
        y_fine_train = keras_split_fine.y.train #El resto ni los ocupamos we



    if debug:
        print("Antes de normalizar:", split_fine.X.train.shape, "Después de ajustar para Keras:", X_fine_train.shape)   
    
    #Paso 4: Clonar modelo para finetunning (para no afectar el modelo original)
    model_ft = clone_model(base)
    model_ft.set_weights(base.get_weights())


    callbacks = [] 
    if dynamic_unfreeze:
        if unfreeze_grid is None:
            # grid por defecto: empieza con unfreeze_last_n//4, 
            # sube a la mitad en época 0.3*epochs, y al total en 0.6*epochs
            n = unfreeze_last_n if unfreeze_last_n != None else len(model_ft.layers)
            unfreeze_grid = [
                (0,                    max(1, n // 4)),
                (int(epochs * 0.3),   max(1, n // 2)),
                (int(epochs * 0.6),   n),
            ]
        callbacks.append(
            make_unfreeze_scheduler(model_ft, unfreeze_grid, verbose=debug)
        )
    else:
        # comportamiento actual: unfreeze estático al inicio
        if  unfreeze_last_n is None or unfreeze_last_n > len(model_ft.layers): 
            for layer in model_ft.layers:
                layer.trainable = True
        else:
            for layer in model_ft.layers:
                layer.trainable = False
            for layer in model_ft.layers[-unfreeze_last_n:]:
                layer.trainable = True



    #Paso 5: recompilar con LR más bajo para fine-tuning

    model_ft.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    class_weight = None
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        classes_present = np.unique(y_fine_train)
        weights = compute_class_weight('balanced', 
                                        classes=classes_present, 
                                        y=y_fine_train)
        class_weight = dict(zip(classes_present, weights))
        if debug:
            print("Class weights:", class_weight)

    if debug:
        verbose = 1
    else:
        verbose = 0

    #Paso 6: callbacks específicos para fine-tuning
    

    if model_checkpoint is None:
        model_checkpoint = "EEGNet_finetuned_subject.keras"

    early_cfg = get_early_stopping_config(
        level=patience_level,
        monitor=monitor
    )

    

    if early_cfg is not None:
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg["monitor"],
                patience=early_cfg["patience"],
                min_delta=early_cfg["min_delta"],
                restore_best_weights=early_cfg["restore_best_weights"],
                mode = monitor_mode
            )
        )

    ckpt_cb = ModelCheckpoint(
        filepath=model_checkpoint,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode=monitor_mode
    )
    callbacks.append(ckpt_cb)
    

    if dynamic_lr:
        if Lr_grid is None:
            Lr_grid = [(0, lr), (int(epochs * 0.2), lr * 0.5), (int(epochs * 0.5), lr * 0.1)]

        callbacks.append(
            LearningRateScheduler(
                make_lr_scheduler(Lr_grid, verbose=debug),
                verbose=0
            )
        )



    #Paso 7: entrenamiento de fine-tuning
    history = model_ft.fit(
        X_fine_train, y_fine_train,
        validation_data=(keras_split_fine.X.val, keras_split_fine.y.val),
        callbacks=callbacks,
        epochs=epochs,
        batch_size=batch_size,          
        class_weight=class_weight,
        verbose=verbose
    )

    #Paso 8: evaluación en test del sujeto

    if make_test:
        print("Se realiza evaluación en el set del dataset.")
        test_loss, test_acc = model_ft.evaluate(keras_split_fine.X.test, split_fine.y.test, verbose=0)
        print(f"[Fine-tune] Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    elif test_run is not None: 
        print("Se realizará evaluación en el set reservado")

        X_test = normalize_run(test_run, mean, std) 
        if debug:
            print("Antes de normalizar:", test_run.X.shape, "Después de ajustar para Keras:", X_test.shape)

        keras_split_fine.X.test = X_test
        keras_split_fine.y.test = test_run.y
        keras_split_fine.sub.test = test_run.sub
        keras_split_fine.trial.test = test_run.trial
        keras_split_fine.run.test = test_run.run
        keras_split_fine.mode.test = test_run.mode

        

        test_loss, test_acc = model_ft.evaluate(X_test, test_run.y, verbose=0)
        print(f"[Fine-tune] Validation loss={test_loss:.4f}, acc={test_acc:.4f}")

    monitor_values = history.history[monitor]  # o early_monitor

    if monitor.endswith("loss"): #Si lo evaluamos en loss
        best_epoch_idx = int(np.argmin(monitor_values))
    else:
        best_epoch_idx = int(np.argmax(monitor_values))

    info["best_epoch"] = best_epoch_idx
    info["best_epoch"]         = best_epoch_idx
    info["best_monitor_value"] = float(monitor_values[best_epoch_idx])
    info["epochs_ran"]         = len(monitor_values)
    info["monitor"]            = monitor
    info["dynamic_lr"] = dynamic_lr
    info["Lr_grid"] = Lr_grid
    info["initial_lr"] = lr
    info["batch_size"] = batch_size
    info["epochs_requested"] = epochs
    info["unfreeze_grid"] = unfreeze_grid





    return model_ft, history, mean, std, keras_split_fine, info

def normalize_run(run, mean, std, eps=1e-6):
    X_norm = normalizar_por_canal(run.X, mean, std, eps)
    X_keras = ajustar_keras(X_norm)

    return X_keras

def normalize_test_split(split, mean, std, eps=1e-6):
    X_test_norm = normalizar_por_canal(split.X, mean, std, eps)
    X_test_keras = ajustar_keras(X_test_norm)

    split.X = X_test_keras

    print("Test set normalizado y ajustado para Keras.")
    print("X_test shape:", split.X.shape)
    print("y_test shape:", split.y.shape)
    return split



def undersample(X, y, sub, rest_label=0, random_state=42, debug=False):
    """
    Undersamplea la clase rest al de mayor cifra de las demás clases
    """

    # Seguridad básica
    assert len(X) == len(y) == len(sub), "X, y y sub deben tener la misma longitud"

    # Índices de rest y no-rest
    idx_rest = np.where(y == rest_label)[0]
    idx_non_rest = np.where(y != rest_label)[0]

    # Si no hay clase rest o no hay otras clases, no hacemos nada
    if len(idx_rest) == 0 or len(idx_non_rest) == 0:
        if debug:
            print("No se aplica undersampling: no hay rest o no hay clases distintas de rest.")
        return X, y, sub

    # Conteos por clase
    unique_cls, counts = np.unique(y, return_counts=True)
    if debug:
        print("Conteos originales por clase:")
        for c, n in zip(unique_cls, counts):
            print(f"  Clase {c}: {n}")

    # Máximo tamaño entre las clases ≠ rest_label
    counts_others = [n for c, n in zip(unique_cls, counts) if c != rest_label]
    max_other = max(counts_others)

    # Si rest ya es <= max_other, no hace falta recortar
    if len(idx_rest) <= max_other:
        if debug:
            print(f"No se aplica undersampling: rest tiene {len(idx_rest)} ≤ max_other={max_other}.")
        return X, y, sub

    # Seleccionamos aleatoriamente max_other muestras de rest
    rng = np.random.default_rng(random_state)
    idx_rest_sel = rng.choice(idx_rest, size=max_other, replace=False)

    # Unimos rest recortado + todas las demás clases
    idx_keep = np.concatenate([idx_rest_sel, idx_non_rest])
    rng.shuffle(idx_keep)

    X_new = X[idx_keep]
    y_new = y[idx_keep]
    sub_new = sub[idx_keep]

    if debug:
        unique_new, counts_new = np.unique(y_new, return_counts=True)
        print(f"Undersampling aplicado: rest de {len(idx_rest)} → {max_other}")
        print("Conteos nuevos por clase:")
        for c, n in zip(unique_new, counts_new):
            print(f"  Clase {c}: {n}")

    return X_new, y_new, sub_new


#==== Vibecoding abajo ====

from tensorflow.keras.callbacks import Callback

def make_unfreeze_scheduler(model, unfreeze_grid, verbose=False):
    class UnfreezeScheduler(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            for ep, n in sorted(unfreeze_grid, key=lambda x: x[0]):
                if epoch == ep:
                    for layer in self.model.layers:
                        layer.trainable = False
                    if n == "all":
                        for layer in self.model.layers:
                            layer.trainable = True
                    else:
                        for layer in self.model.layers[-n:]:
                            layer.trainable = True
                    current_lr = float(
                        self.model.optimizer.learning_rate
                    )
                    self.model.compile(
                        optimizer=Adam(learning_rate=current_lr),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"]
                    )
                    if verbose:
                        status = "todas" if n == "all" else f"últimas {n}"
                        print(f"\n[UnfreezeScheduler] Época {epoch}: descongelando {status} capas")
    return UnfreezeScheduler()

def evaluate(
    modelo,
    keras_split=None,
    label_map=None,
    title="EEGNet",
    all_data=False,
    test_run=None,
    plot=True,
    verbose=True
):
    """
    Evalúa un modelo EEGNet de forma más completa para experimentos tipo paper.

    Usa, en orden de prioridad:
    1) test_run si se entrega
    2) todo train+val+test si all_data=True
    3) keras_split.X.test / keras_split.y.test por defecto

    IMPORTANTE:
    - test_run debe estar ya normalizado y en formato Keras: (N, C, T, 1)
      o ser un DataSplit/EEGSubset previamente preparado.
    """

    if test_run is not None:
        X_test = test_run.X
        y_test = test_run.y

    elif all_data:
        if keras_split is None:
            raise ValueError("Debes entregar keras_split si all_data=True.")

        X_parts = []
        y_parts = []

        for X_part, y_part in [
            (keras_split.X.train, keras_split.y.train),
            (keras_split.X.val, keras_split.y.val),
            (keras_split.X.test, keras_split.y.test),
        ]:
            if X_part is not None and len(X_part) > 0:
                X_parts.append(X_part)
                y_parts.append(y_part)

        X_test = np.concatenate(X_parts, axis=0)
        y_test = np.concatenate(y_parts, axis=0)

    else:
        if keras_split is None:
            raise ValueError("Debes entregar keras_split si no usas test_run.")

        X_test = keras_split.X.test
        y_test = keras_split.y.test

    if X_test is None or len(X_test) == 0:
        raise ValueError("No hay datos de evaluación. X_test está vacío.")

    y_prob = modelo.predict(X_test, verbose=0)

    if label_map is not None:
        labels = np.array(sorted(label_map.keys()))
        class_names = [label_map[i] for i in labels]
    else:
        labels = np.arange(y_prob.shape[1])
        class_names = [str(i) for i in labels]

    results = evaluar_modelo_multiclase(
        y_true=y_test,
        y_prob=y_prob,
        labels=labels,
        class_names=class_names,
        title_prefix=title,
        plot=plot,
        verbose=verbose
    )

    return results

def evaluar_modelo_multiclase(
    y_true,
    y_prob,
    labels=None,
    class_names=None,
    title_prefix="EEGNet",
    plot=True,
    verbose=True
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
        matthews_corrcoef,
        log_loss,
        roc_auc_score
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    y_pred = np.argmax(y_prob, axis=1)

    n_classes = y_prob.shape[1]

    if labels is None:
        labels = np.arange(n_classes)

    labels = np.asarray(labels)

    if class_names is None:
        class_names = [str(i) for i in labels]

    # =====================================================
    # Métricas globales
    # =====================================================

    acc = accuracy_score(y_true, y_pred)

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    kappa = cohen_kappa_score(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        loss = log_loss(y_true, y_prob, labels=labels)
    except Exception:
        loss = np.nan

    # ROC-AUC multiclass, útil si hay probabilidades bien calibradas
    try:
        if n_classes == 2:
            auc_macro = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc_macro = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro"
            )
    except Exception:
        auc_macro = np.nan

    chance_level = 1.0 / n_classes

    if acc > chance_level:
        normalized_accuracy_over_chance = (acc - chance_level) / (1.0 - chance_level)
    else:
        normalized_accuracy_over_chance = 0.0

    # =====================================================
    # Matrices de confusión
    # =====================================================

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    # =====================================================
    # Tabla por clase
    # =====================================================

    per_class_df = pd.DataFrame({
        "class_id": labels,
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    })

    # =====================================================
    # Reporte estilo sklearn
    # =====================================================

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )

    # =====================================================
    # Print interpretativo
    # =====================================================

    if verbose:
        print("\n" + "=" * 70)
        print(f" EVALUACIÓN {title_prefix}")
        print("=" * 70)

        print("\n[Métricas globales]")
        print(f"Accuracy                  : {acc:.4f}")
        print(f"Balanced accuracy         : {bal_acc:.4f}")
        print(f"Macro precision           : {macro_precision:.4f}")
        print(f"Macro recall              : {macro_recall:.4f}")
        print(f"Macro F1                  : {macro_f1:.4f}")
        print(f"Weighted F1               : {weighted_f1:.4f}")
        print(f"Cohen's kappa             : {kappa:.4f}")
        print(f"Matthews corrcoef         : {mcc:.4f}")
        print(f"Log loss                  : {loss:.4f}")
        print(f"ROC-AUC macro             : {auc_macro:.4f}")
        print(f"Chance level              : {chance_level:.4f}")
        print(f"Accuracy over chance norm : {normalized_accuracy_over_chance:.4f}")

        print("\n[Clases]")
        for i, name in zip(labels, class_names):
            print(f"  {i}: {name}")

        print("\n[Reporte de clasificación]")
        print(report_text)

        print("\n[Interpretación rápida]")
        print("- Accuracy: rendimiento global.")
        print("- Balanced accuracy: accuracy corregida por desbalance entre clases.")
        print("- Macro F1: promedio de F1 dando el mismo peso a cada clase.")
        print("- Weighted F1: F1 ponderado por cantidad de muestras.")
        print("- Cohen's kappa: acuerdo modelo-realidad corregido por azar.")
        print("- Recall por clase: clave para ver si el modelo está ignorando una clase.")

    # =====================================================
    # Plots
    # =====================================================

    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(class_names, f1)
        plt.ylabel("F1-score")
        plt.ylim(0, 1)
        plt.title(f"F1 por clase ({title_prefix})")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.title(f"Matriz de confusión normalizada ({title_prefix})")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.title(f"Matriz de confusión en conteos ({title_prefix})")
        plt.tight_layout()
        plt.show()

    # =====================================================
    # Diccionario final compatible con CSV
    # =====================================================

    results = {
        "title": title_prefix,

        # Datos crudos de evaluación
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,

        # Métricas globales
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "mcc": mcc,
        "log_loss": loss,
        "roc_auc_macro": auc_macro,
        "chance_level": chance_level,
        "normalized_accuracy_over_chance": normalized_accuracy_over_chance,

        # Métricas por clase
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "per_class_df": per_class_df,

        # Matrices
        "cm": cm,
        "cm_norm": cm_norm,

        # Reporte
        "classification_report": report_dict,
        "classification_report_text": report_text,

        # Nombres
        "labels": labels,
        "class_names": class_names
    }

    # También agregamos llaves planas por clase para CSV
    for i, name in enumerate(class_names):
        clean_name = str(name).replace(" ", "_")
        results[f"precision_{clean_name}"] = float(precision[i])
        results[f"recall_{clean_name}"] = float(recall[i])
        results[f"f1_{clean_name}"] = float(f1[i])
        results[f"support_{clean_name}"] = int(support[i])

    return results


def plot_history(history, metrics=("accuracy",), title_prefix="EEGNet", 
                 lr_schedule=None, unfreeze_schedule=None, best_epoch=None):
    hist_dict = history.history

    def draw_vertical_lines():
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin

        # LR changes
        if lr_schedule is not None:
            for epoch, lr in sorted(lr_schedule, key=lambda x: int(x[0])):
                epoch = int(epoch)
                if epoch == 0:
                    continue
                ax.axvline(x=epoch, linestyle="--", linewidth=1, alpha=0.7, color="steelblue")
                ax.text(epoch, ymax - 0.05 * y_range, f"LR={lr:g}",
                        rotation=90, va="top", ha="right", fontsize=8, color="steelblue")

        # Unfreeze changes
        if unfreeze_schedule is not None:
            for epoch, n in sorted(unfreeze_schedule, key=lambda x: int(x[0])):
                epoch = int(epoch)
                if epoch == 0:
                    continue
                label = "todas" if n == "all" else f"unfreeze {n}"
                ax.axvline(x=epoch, linestyle=":", linewidth=1, alpha=0.7, color="darkorange")
                ax.text(epoch, ymin + 0.05 * y_range, label,
                        rotation=90, va="bottom", ha="right", fontsize=8, color="darkorange")

        # Best epoch
        if best_epoch is not None:
            ax.axvline(x=best_epoch, linestyle="-.", linewidth=1.2, alpha=0.85, color="green")
            ax.text(best_epoch, ymin + 0.50 * y_range, f"best ({best_epoch})",
                    rotation=90, va="center", ha="right", fontsize=8, color="green")

    # Loss plot
    plt.figure(figsize=(7, 4))
    plt.plot(hist_dict["loss"], label="train loss")
    if "val_loss" in hist_dict:
        plt.plot(hist_dict["val_loss"], label="val loss")
    draw_vertical_lines()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} – Curva de pérdida")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Metric plots
    for m in metrics:
        if m not in hist_dict:
            print(f"[plot_history] Métrica '{m}' no encontrada.")
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(hist_dict[m], label=f"train {m}")
        if f"val_{m}" in hist_dict:
            plt.plot(hist_dict[f"val_{m}"], label=f"val {m}")
        draw_vertical_lines()
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title(f"{title_prefix} – {m} por epoch")
        plt.legend()
        plt.tight_layout()
        plt.show()


    
def load_model(model_path = None, params_path = None):
    from tensorflow.keras.models import load_model
    if model_path is None: 
        raise IndexError("No se ha especificado el path del modelo")
        
    if params_path is None:
        raise IndexError("No se ha especificado el path de los parámetros de normalización")
    

    model = load_model(model_path)
    params = np.load(params_path, allow_pickle=True)
    mean = params["mean"]
    std = params["std"]
    return model, mean, std

def build_canonical_to_model_map(data_obj):
    """
    Construye el mapping entre las etiquetas canónicas originales del dataset
    y los índices locales de salida del modelo entrenado.

    Ejemplo para mode=4:
        rest    -> 0
        right_i -> 1
        left_i  -> 2
        right_m -> 1
        left_m  -> 2
    """

    if data_obj.label_map is None:
        raise ValueError(
            "data_obj.label_map no existe. "
            "Debes ejecutar Selection.pipeline() antes de guardar el modelo."
        )

    model_class_names = [
        data_obj.label_map[i]
        for i in sorted(data_obj.label_map.keys())
    ]

    canonical_classes_used = [
        str(data_obj.class_names_og[idx])
        for idx in data_obj.pick
    ]

    canonical_to_model = {}

    for canonical_name in canonical_classes_used:

        if data_obj.binary:
            local_name = "rest" if canonical_name == "rest" else "no_rest"

        elif canonical_name in model_class_names:
            local_name = canonical_name

        elif data_obj.fusionar:
            if canonical_name == "rest":
                local_name = "rest"
            else:
                # right_i -> right, left_m -> left, hands_i -> hands...
                local_name = canonical_name.rsplit("_", 1)[0]

        else:
            local_name = canonical_name

        if local_name not in model_class_names:
            raise ValueError(
                f"No fue posible mapear la clase canónica '{canonical_name}' "
                f"a las clases finales del modelo: {model_class_names}."
            )

        canonical_to_model[canonical_name] = model_class_names.index(local_name)

    return canonical_to_model

def save_model(
    model,
    mean,
    std,
    label_map,
    path=None,
    name="model_EEGNet"
):
    """
    Guarda:
        <name>.keras -> modelo entrenado
        <name>.npz   -> mean, std y significado de sus salidas
    """

    if path is None:
        path = os.getcwd()

    os.makedirs(path, exist_ok=True)

    if name.endswith(".keras") or name.endswith(".npz"):
        raise ValueError("name debe ir sin extensión.")
    
    

    n_outputs = model.output_shape[-1] #Entrega el número de salidas del modelo, que debería coincidir con el número de clases que tenemos después del pickeo y la fusión

    if sorted(label_map.keys()) != list(range(n_outputs)):
        raise ValueError(
            f"label_map incompatible con el modelo. "
            f"Modelo: {n_outputs} salidas | label_map: {label_map}"
        )

    class_names = np.array( #Entrega un array con los nombres de las salidas
        [label_map[i] for i in range(n_outputs)],
        dtype=str
    )

    simp_class_names = []

    for name_in in class_names:
        if name_in.endswith("_m") or name_in.endswith("_i"):
            name_in = name_in[:-2]

        simp_class_names.append(name_in)


    model_path = os.path.join(path, f"{name}.keras")
    params_path = os.path.join(path, f"{name}.npz")

    model.save(model_path)

    np.savez_compressed(
        params_path,
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        class_names=simp_class_names
    )

    print(f"Modelo guardado: {model_path}")
    print(f"Parámetros guardados: {params_path}")
    print(f"Clases del modelo: {dict(enumerate(class_names))}")

 
def compare_models(
    modelo,
    modelo_ft,
    keras_split,
    keras_split_ft,
    label_map=None,
    verbose=True
):


    # =====================================================
    # MODELO GENERAL
    # =====================================================

    X_test_gen = keras_split.X.test
    y_test_gen = keras_split.y.test

    y_prob_gen = modelo.predict(X_test_gen, verbose=0)
    y_pred_gen = np.argmax(y_prob_gen, axis=1)

    acc_gen = accuracy_score(y_test_gen, y_pred_gen)

    f1_gen = f1_score(
        y_test_gen,
        y_pred_gen,
        average='macro'
    )

    # =====================================================
    # MODELO FINE-TUNED
    # =====================================================

    X_test_ft = keras_split_ft.X.test
    y_test_ft = keras_split_ft.y.test

    y_prob_ft = modelo_ft.predict(X_test_ft, verbose=0)
    y_pred_ft = np.argmax(y_prob_ft, axis=1)

    acc_ft = accuracy_score(y_test_ft, y_pred_ft)

    f1_ft = f1_score(
        y_test_ft,
        y_pred_ft,
        average='macro'
    )

    # =====================================================
    # MEJORAS
    # =====================================================

    acc_improvement = (
        (acc_ft - acc_gen)
        / acc_gen
    ) * 100

    f1_improvement = (
        (f1_ft - f1_gen)
        / f1_gen
    ) * 100

    abs_acc = (acc_ft - acc_gen) * 100
    abs_f1 = (f1_ft - f1_gen) * 100

    # =====================================================
    # NOMBRES DE CLASES
    # =====================================================

    if label_map is not None:
        class_names = [
            label_map[k]
            for k in sorted(label_map.keys())
        ]
    else:
        n_classes = len(np.unique(y_test_gen))
        class_names = [str(i) for i in range(n_classes)]

    # =====================================================
    # PRINT BONITO
    # =====================================================

    if verbose:

        print("\n" + "="*60)
        print(" COMPARACIÓN MODELO GENERAL vs FINE-TUNING ")
        print("="*60)

        print("\n[ MODELO GENERAL ]")
        print(f"Accuracy : {acc_gen:.4f}")
        print(f"Macro F1 : {f1_gen:.4f}")

        print("\n[ FINE-TUNING ]")
        print(f"Accuracy : {acc_ft:.4f}")
        print(f"Macro F1 : {f1_ft:.4f}")

        print("\n[ MEJORAS ]")
        print(f"Δ Accuracy absoluta : +{abs_acc:.2f} puntos")
        print(f"Δ Accuracy relativa : +{acc_improvement:.2f}%")

        print(f"Δ Macro F1 absoluta : +{abs_f1:.2f} puntos")
        print(f"Δ Macro F1 relativa : +{f1_improvement:.2f}%")

        print("\n" + "="*60)

        print("\n=== Classification Report (General) ===\n")

        print(
            classification_report(
                y_test_gen,
                y_pred_gen,
                target_names=class_names,
                zero_division=0
            )
        )

        print("\n=== Classification Report (Fine-Tuning) ===\n")

        print(
            classification_report(
                y_test_ft,
                y_pred_ft,
                target_names=class_names,
                zero_division=0
            )
        )

    # =====================================================
    # RETURN
    # =====================================================

    return {

        # GENERAL
        "acc_general": acc_gen,
        "f1_general": f1_gen,

        # FINE
        "acc_fine": acc_ft,
        "f1_fine": f1_ft,

        # MEJORAS
        "acc_improvement_percent": acc_improvement,
        "f1_improvement_percent": f1_improvement,

        "acc_absolute_gain": abs_acc,
        "f1_absolute_gain": abs_f1,

        # PREDICCIONES
        "y_true_general": y_test_gen,
        "y_pred_general": y_pred_gen,

        "y_true_fine": y_test_ft,
        "y_pred_fine": y_pred_ft,
    }
    
def save_metrics_csv(
    info,
    metrics,
    csv_path,
    experiment_name="experiment",
    model_type="general",
    label_map=None,
    extra_info=None,
    append=True
):
    """
    Guarda en CSV una fila con métricas del experimento.

    Parámetros
    ----------
    info : dict
        Diccionario con información del entrenamiento:
        mode, n_train, n_val, n_test, Lr_grid, batch_size, epochs, etc.

    metrics : dict
        Diccionario devuelto por evaluate/evaluar_modelo_multiclase.
        Idealmente debe contener:
        - y_true
        - y_pred
        - precision
        - recall
        - f1
        - cm
        - cm_norm

    csv_path : str
        Ruta donde guardar el CSV.

    experiment_name : str
        Nombre del experimento.

    model_type : str
        "general", "fine_tuned", "fine_tuned_own_stats", etc.

    label_map : dict
        Ejemplo:
        {0: "rest", 1: "left", 2: "right"}

    extra_info : dict
        Información adicional:
        sujeto, run, lr, unfreeze_last_n, mean_source, etc.

    append : bool
        Si True, agrega la fila al CSV si ya existe.
    """



    row = {}

    # =====================================================
    # Identificación del experimento
    # =====================================================
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["experiment_name"] = experiment_name
    row["model_type"] = model_type

    # =====================================================
    # Info del entrenamiento
    # =====================================================
    if info is not None:
        for key, value in info.items():
            if isinstance(value, (list, tuple, dict, np.ndarray)):
                row[key] = json.dumps(
                    value if not isinstance(value, np.ndarray) else value.tolist()
                )
            else:
                row[key] = value

    # =====================================================
    # Métricas globales
    # =====================================================
    y_true = metrics.get("y_true", None)
    y_pred = metrics.get("y_pred", None)

    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        row["accuracy"] = accuracy_score(y_true, y_pred)
        row["macro_f1"] = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
        row["weighted_f1"] = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
        row["n_test_samples"] = len(y_true)
    else:
        row["accuracy"] = None
        row["macro_f1"] = None
        row["weighted_f1"] = None
        row["n_test_samples"] = None

    # =====================================================
    # Métricas por clase
    # =====================================================
    precision = metrics.get("precision", None)
    recall = metrics.get("recall", None)
    f1 = metrics.get("f1", None)

    if label_map is not None:
        class_names = [label_map[i] for i in sorted(label_map.keys())]
    else:
        if f1 is not None:
            class_names = [f"class_{i}" for i in range(len(f1))]
        else:
            class_names = []

    if precision is not None:
        for i, name in enumerate(class_names):
            row[f"precision_{name}"] = float(precision[i])

    if recall is not None:
        for i, name in enumerate(class_names):
            row[f"recall_{name}"] = float(recall[i])

    if f1 is not None:
        for i, name in enumerate(class_names):
            row[f"f1_{name}"] = float(f1[i])

    # =====================================================
    # Matrices de confusión
    # =====================================================
    if "cm" in metrics:
        cm = np.asarray(metrics["cm"])
        row["confusion_matrix"] = json.dumps(cm.tolist())

    if "cm_norm" in metrics:
        cm_norm = np.asarray(metrics["cm_norm"])
        row["confusion_matrix_norm"] = json.dumps(cm_norm.tolist())

    # =====================================================
    # Información extra del experimento
    # =====================================================
    if extra_info is not None:
        for key, value in extra_info.items():
            if isinstance(value, (list, tuple, dict, np.ndarray)):
                row[key] = json.dumps(
                    value if not isinstance(value, np.ndarray) else value.tolist()
                )
            else:
                row[key] = value

    # =====================================================
    # Guardar CSV
    # =====================================================
    df_row = pd.DataFrame([row])

    if append and os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_out = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_out = df_row

    df_out.to_csv(csv_path, index=False)

    print(f"✔ Métricas guardadas en: {csv_path}")
    return row

