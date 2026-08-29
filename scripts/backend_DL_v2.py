#Generador de clases para procesamiento

#Primero, importamos todas las librerías que ya teníamos antes 
#Clásicas

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

project_path = r'C:\Users\Josjuan\Desktop\Tesis\Movement_clasify' 

#Agregamos los modelos EEGnet a nuestro path
models_path = os.path.join(project_path, 'models')

if os.path.exists(models_path):
    if models_path not in sys.path:
        sys.path.append(models_path)
    
else:
    print("No se encontró directorio de modelos en", models_path)

#Personales
from EEGModels import EEGNet
from models import ATCNet_ as ATCNet, EEGNeX_8_32  

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
        # Propagamos channels_names si viene en el dict (opcional)
        self.channels_names = data_dict.get("channels_names", None)

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
                     allowed_modes = ["LR_M", "HF_M"]),

            # 🔹 REST-IZQ-DER IMAGINARIAS

            16: dict(pick=[0,1,2], fusionar=False, binary=False,
                     allowed_modes = ["LR_I"]), 

            # 🔹 REST-MANOS-PIES IMAGINARIAS

            17: dict(pick=[0,3,4], fusionar=False, binary=False,
                     allowed_modes = ["HF_I"]),

            # 🔹 BINARIO-IMAGINARIO

            18: dict(pick=[0,1,2,3,4], fusionar=True, binary=True,
                    allowed_modes = ["LR_I", "HF_I"]),

            # 🔹 IZQ-DER IMAGINARIO       
        
            19: dict(pick=[1,2], fusionar=False, binary=False,
                    allowed_modes = ["LR_I"]),

            # 🔹 REST-RIGHT-FEET (moabb)

            20: dict(pick = [0, 1, 4], fusionar = False, binary = False,
                    allowed_modes = ["LR_I", "HF_I"]),

            # 🔹 REST-IZQ-DER HGD (solo motor, sin fusión)

            21: dict(pick=[0, 5, 6], fusionar=False, binary=False,
                    allowed_modes = ["HGD_LRRF_M"]),

            # 🔹 IZQ-DER HGD (solo motor, sin fusión)

            22: dict(pick=[5, 6], fusionar=False, binary=False,
                    allowed_modes = ["HGD_LRRF_M"]),

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
    
    def pick_fine(self, subject_id, test_run=None, run_split=False, run_fold = False):

        mask_model = self.sub != subject_id
        mask_fine  = self.sub == subject_id

        #1) Parte: Splits de entrenamiento general y fine-tune
        data_model = {
            "X": self.X[mask_model],
            "y": self.y[mask_model],
            "sub": self.sub[mask_model],
            "trial": self.trial[mask_model],
            "run": self.run[mask_model],
            "mode": self.mode[mask_model],
            "channels_names": self.channels_names,
        }

        data_fine_full = {
            "X": self.X[mask_fine],
            "y": self.y[mask_fine],
            "sub": self.sub[mask_fine],
            "trial": self.trial[mask_fine],
            "run": self.run[mask_fine],
            "mode": self.mode[mask_fine],
            "channels_names": self.channels_names,
        }

        self.data_model = EEGSubset(data_model)
        self.data_fine = EEGSubset(data_fine_full)

        self.fine_subject = subject_id


        #=================================================
        #Caso 1: no separar por run: Retornamos todo nomás
        #=====================================================
        if not run_split and not run_fold:

            if self.Debug:
                print(f"✔ Sujetos entrenamiento general: {np.unique(data_model['sub'])}")
                print(f"✔ Sujeto fine-tune: {subject_id}")
                print(f"Shape modelo: {data_model['X'].shape}")
                print(f"Shape fine: {data_fine_full['X'].shape}")
                print(f"Runs fine disponibles: {np.unique(data_fine_full['run'])}")

            return self.data_model, self.data_fine, None

        #=====================================================
        #Validaar que hayan suficientes runs
        #=================================================
        unique_runs = np.unique(data_fine_full["run"])

        if len(unique_runs) < 2:
            raise ValueError(
                f"El sujeto {subject_id} tiene menos de 2 runs. "
                "No se puede separar fine-tuning y test por run."
            )

        #=====================================================
        #Caso 2: leave-one-run-out / run-fold
        #=====================================================
        if run_fold: #Creamos una lista con las diferentes probabilidades de folds

            data_fine_folds = []
            data_test_folds = []
            fold_info = []

            for current_test_run in unique_runs:

                mask_test = data_fine_full["run"] == current_test_run
                mask_fine_train = ~mask_test

                data_fine_train = {
                    "X": data_fine_full["X"][mask_fine_train],
                    "y": data_fine_full["y"][mask_fine_train],
                    "sub": data_fine_full["sub"][mask_fine_train],
                    "trial": data_fine_full["trial"][mask_fine_train],
                    "run": data_fine_full["run"][mask_fine_train],
                    "mode": data_fine_full["mode"][mask_fine_train],
                    "channels_names": self.channels_names,
                }

                data_test = {
                    "X": data_fine_full["X"][mask_test],
                    "y": data_fine_full["y"][mask_test],
                    "sub": data_fine_full["sub"][mask_test],
                    "trial": data_fine_full["trial"][mask_test],
                    "run": data_fine_full["run"][mask_test],
                    "mode": data_fine_full["mode"][mask_test],
                    "channels_names": self.channels_names,
                }

                fine_subset = EEGSubset(data_fine_train)
                test_subset = EEGSubset(data_test)

                # Seguridad anti-leakage
                overlap = np.intersect1d(fine_subset.run, test_subset.run)
                if len(overlap) > 0:
                    raise RuntimeError(
                        f"Leakage detectado en fold con test_run={current_test_run}. "
                        f"Runs compartidas: {overlap}"
                    )
                

                #Guardamos los datos en la lista
                data_fine_folds.append(fine_subset)
                data_test_folds.append(test_subset)

                fold_info.append({
                    "test_run": current_test_run,
                    "fine_runs": np.unique(fine_subset.run),
                    "test_runs": np.unique(test_subset.run),
                    "n_fine": len(fine_subset.y),
                    "n_test": len(test_subset.y)
                })

            self.data_fine_folds = data_fine_folds
            self.data_test_folds = data_test_folds
            self.fold_info = fold_info
            print(f"cantidad total de folds: {len(data_fine_folds)}")

            if self.Debug:
                print(f"✔ Run-fold activado para sujeto {subject_id}")
                print(f"Runs disponibles: {unique_runs}")
                print(f"Número de folds: {len(data_fine_folds)}")

                for i, info in enumerate(fold_info):
                    print("\n" + "-" * 40)
                    print(f"Fold {i}")
                    print(f"Run test: {info['test_run']}")
                    print(f"Runs fine: {info['fine_runs']}")
                    print(f"Runs test: {info['test_runs']}")
                    print(f"Shape fine: {data_fine_folds[i].X.shape}")
                    print(f"Shape test: {data_test_folds[i].X.shape}")

            return self.data_model, data_fine_folds, data_test_folds

        # =====================================================
        # Caso 3: run_split clásico con una sola run de test
        # =====================================================
        if test_run is None:
            test_run = unique_runs[0]

        if test_run not in unique_runs:
            raise ValueError(
                f"test_run={test_run} no existe para el sujeto {subject_id}. "
                f"Runs disponibles: {unique_runs}"
            )

        mask_test = data_fine_full["run"] == test_run
        mask_fine_train = ~mask_test

        data_fine_train = {
            "X": data_fine_full["X"][mask_fine_train],
            "y": data_fine_full["y"][mask_fine_train],
            "sub": data_fine_full["sub"][mask_fine_train],
            "trial": data_fine_full["trial"][mask_fine_train],
            "run": data_fine_full["run"][mask_fine_train],
            "mode": data_fine_full["mode"][mask_fine_train]
        }

        data_test = {
            "X": data_fine_full["X"][mask_test],
            "y": data_fine_full["y"][mask_test],
            "sub": data_fine_full["sub"][mask_test],
            "trial": data_fine_full["trial"][mask_test],
            "run": data_fine_full["run"][mask_test],
            "mode": data_fine_full["mode"][mask_test]
        }

        self.data_fine = EEGSubset(data_fine_train)
        self.data_test = EEGSubset(data_test)
        self.test_run = test_run

        # Seguridad anti-leakage
        overlap = np.intersect1d(self.data_fine.run, self.data_test.run)
        if len(overlap) > 0:
            raise RuntimeError(
                f"Leakage detectado. Runs compartidas entre fine y test: {overlap}"
            )

        if self.Debug:
            print(f"✔ Sujetos entrenamiento general: {np.unique(self.data_model.sub)}")
            print(f"✔ Sujeto fine-tune: {subject_id}")
            print(f"✔ Run reservada para test: {test_run}")
            print(f"Runs fine-tune: {np.unique(self.data_fine.run)}")
            print(f"Runs test: {np.unique(self.data_test.run)}")
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
            
    def pipeline(self, n=None, binary = False, filter_modes=None):
        #El pipeline principal consiste en aplicar un pick y luego una fusión, cosas que ya tenemos en otras clases
        #n sólo es para limitar la cantidad de datos totales, no recuerdo porque lo quise añadir pero ahí está
        #filter_modes: lista de modos a conservar tras el pipeline, e.g. ["LR_I"] para quedarse sólo con MI
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

            if filter_modes is not None:
                self.filter_modes(filter_modes)
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

        if filter_modes is not None:
            self.filter_modes(filter_modes)

    def filter_modes(self, keep_modes):
        """
        Filtra los datos para conservar sólo los modos indicados.
        Útil para entrenar con picks fusionados (MI+MR) pero hacer
        fine-tuning y test sólo con MI.

        Uso típico:
            sel = Selection(pick=4)          # pick fusionado MI+MR
            sel.load(path); sel.pipeline()
            sel.filter_modes(["LR_I"])       # quedarse sólo con MI

        Parámetros
        ----------
        keep_modes : list[str]
            Modos a conservar, e.g. ["LR_I"], ["HF_I"], ["LR_I","HF_I"].
        """
        mask = np.isin(self.mode, keep_modes)

        if mask.sum() == 0:
            raise ValueError(
                f"filter_modes: no quedaron muestras con modos {keep_modes}. "
                f"Modos presentes: {list(np.unique(self.mode))}"
            )

        self.X = self.X[mask]
        self.y = self.y[mask]
        self.sub = self.sub[mask]
        self.trial = self.trial[mask]
        self.run = self.run[mask]
        self.mode = self.mode[mask]
        self.n_samples = self.X.shape[0]

        return self

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

#4 Construir modelo
def build_eegnet(classes, chans, signal_len,
                 dropout_rate=0.5, kern_length=64,
                 F1=8, D=2, norm_rate=0.25,
                 dropout_type='Dropout',
                 model_type='eegnet'):

    model_type = model_type.lower()

    if model_type == 'eegnet':
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

    elif model_type == 'eegnex':
        from tensorflow.keras.layers import Input, Permute
        from tensorflow.keras.models import Model as KerasModel
        inner = EEGNeX_8_32(
            n_timesteps=signal_len,
            n_features=chans,
            n_outputs=classes,
        )
        inp = Input(shape=(chans, signal_len, 1))
        x = Permute((3, 1, 2))(inp)
        out = inner(x)
        model = KerasModel(inp, out)

    elif model_type == 'atcnet':
        from tensorflow.keras.layers import Input, Permute
        from tensorflow.keras.models import Model as KerasModel
        inner = ATCNet(
            n_classes=classes,
            in_chans=chans,
            in_samples=signal_len,
            n_windows=5,
            eegn_F1=F1,
            eegn_D=D,
            eegn_kernelSize=kern_length,
            eegn_poolSize=4,   # adecuado para 320 muestras (160Hz × 2s)
            eegn_dropout=dropout_rate,
        )
        inp = Input(shape=(chans, signal_len, 1))
        x = Permute((3, 1, 2))(inp)
        out = inner(x)
        model = KerasModel(inp, out)

    else:
        raise ValueError(f"model_type '{model_type}' no soportado. Usa 'eegnet', 'eegnex' o 'atcnet'.")

    print(f"Modelo: {model_type} | {classes} clases, {chans} ch, {signal_len} samples, {len(model.layers)} capas")
    return model

def runfolds(
    fine_obj,
    test_obj,
    subjects=None,
    n_subjects=None,
    seed=42,
    min_runs=2,
    debug=True
):
    """
    Construye folds leave-one-run-out para sujetos holdout.

    fine_obj:
        Objeto Selection ya cargado y pasado por pipeline().
        Debe contener datos limpios/con dropbad para fine-tuning.

    test_obj:
        Objeto Selection ya cargado y pasado por pipeline().
        Debe contener datos realistas/sin dropbad para evaluación final.

    subjects:
        Lista de sujetos a evaluar. Si es None, se usan sujetos comunes
        entre fine_obj y test_obj. Si n_subjects no es None, se eligen
        aleatoriamente con seed.

    Retorna:
        data_fine_folds : lista de EEGSubset
        data_test_folds : lista de EEGSubset
        fold_table      : DataFrame con metadata de cada fold
    """



    # -------------------------------------------------
    # 0) Validaciones  (Lo pidió la IA)
    # -------------------------------------------------
    required_attrs = ["X", "y", "sub", "trial", "run", "mode"]

    for attr in required_attrs:
        if not hasattr(fine_obj, attr):
            raise ValueError(f"fine_obj no tiene atributo {attr}")
        if not hasattr(test_obj, attr):
            raise ValueError(f"test_obj no tiene atributo {attr}")

    if fine_obj.X is None or test_obj.X is None:
        raise ValueError("fine_obj y test_obj deben estar cargados y procesados con pipeline().")

    # Validación de clases
    if hasattr(fine_obj, "label_map") and hasattr(test_obj, "label_map"):
        if fine_obj.label_map != test_obj.label_map:
            raise ValueError(
                "fine_obj y test_obj tienen label_map distintos. "
                f"fine: {fine_obj.label_map} | test: {test_obj.label_map}"
            )

    # Validación de shape de canales/tiempo
    if fine_obj.X.shape[1:] != test_obj.X.shape[1:]:
        raise ValueError(
            "fine_obj y test_obj tienen distinta forma de ventana. "
            f"fine: {fine_obj.X.shape[1:]} | test: {test_obj.X.shape[1:]}"
        )

    #1) Elegimos los sujetos comunes (Deberían ser todos!)
    fine_subjects = np.unique(fine_obj.sub)
    test_subjects = np.unique(test_obj.sub)

    common_subjects = np.intersect1d(fine_subjects, test_subjects)


    

    if subjects is None:
        subjects = common_subjects

        if n_subjects is not None:
            if n_subjects > len(common_subjects):
                raise ValueError(
                    f"n_subjects={n_subjects} excede sujetos comunes disponibles: "
                    f"{len(common_subjects)}"
                )

            rng = np.random.default_rng(seed)
            subjects = rng.choice(common_subjects, size=n_subjects, replace=False)

    subjects = np.asarray(subjects)

    missing = [s for s in subjects if s not in common_subjects]
    if missing:
        raise ValueError(
            f"Estos sujetos no existen en ambos objetos fine/test: {missing}. "
            f"Sujetos comunes: {common_subjects}"
        )

 

    # -------------------------------------------------
    # Helper interno
    # -------------------------------------------------
    def make_subset(obj, mask):
        data = {
            "X":             obj.X[mask],
            "y":             obj.y[mask],
            "sub":           obj.sub[mask],
            "trial":         obj.trial[mask],
            "run":           obj.run[mask],
            "mode":          obj.mode[mask],
            "channels_names": getattr(obj, 'channels_names', None),
        }
        return EEGSubset(data)

    #2) Construimos los 2 folds
    data_fine_folds = [] #Contiene todos los runs de fine-tuning
    data_test_folds = [] #Contiene los runs del test (uno del run-folds)
    rows = []

    fold_id = 0

    for subject in subjects: #Lo vamos a hacer con cada sujeto de la lista

        fine_mask_subject = fine_obj.sub == subject
        test_mask_subject = test_obj.sub == subject

        fine_runs = np.unique(fine_obj.run[fine_mask_subject])
        test_runs = np.unique(test_obj.run[test_mask_subject])

        common_runs = np.intersect1d(fine_runs, test_runs)

        if len(common_runs) < min_runs:
            if debug:
                print(
                    f"⚠ Sujeto {subject} omitido: solo tiene {len(common_runs)} runs comunes "
                    f"entre fine y test. Runs comunes: {common_runs}"
                )
            continue

        for current_test_run in common_runs:

            # Fine-tuning: sujeto actual, todas las runs excepto current_test_run
            mask_fine = (
                (fine_obj.sub == subject) &
                (fine_obj.run != current_test_run)
            )

            # Test: sujeto actual, solo current_test_run, desde test_obj realista
            mask_test = (
                (test_obj.sub == subject) &
                (test_obj.run == current_test_run)
            )

            fine_subset = make_subset(fine_obj, mask_fine)
            test_subset = make_subset(test_obj, mask_test)

            if len(fine_subset.y) == 0:
                raise RuntimeError(
                    f"Fold inválido: sujeto {subject}, test_run {current_test_run}, "
                    "fine_subset quedó vacío."
                )

            if len(test_subset.y) == 0:
                raise RuntimeError(
                    f"Fold inválido: sujeto {subject}, test_run {current_test_run}, "
                    "test_subset quedó vacío."
                )

            # Seguridad anti-leakage por run dentro del mismo sujeto
            overlap_runs = np.intersect1d(
                np.unique(fine_subset.run),
                np.unique(test_subset.run)
            )

            if len(overlap_runs) > 0:
                raise RuntimeError(
                    f"Leakage detectado en fold {fold_id}. "
                    f"Sujeto {subject}, runs compartidas: {overlap_runs}"
                )

            # Seguridad: el sujeto debe ser el mismo en fine y test
            if len(np.unique(fine_subset.sub)) != 1 or len(np.unique(test_subset.sub)) != 1:
                raise RuntimeError(f"Fold {fold_id} contiene más de un sujeto.")

            if np.unique(fine_subset.sub)[0] != np.unique(test_subset.sub)[0]:
                raise RuntimeError(f"Fold {fold_id}: sujeto fine y sujeto test no coinciden.")

            data_fine_folds.append(fine_subset)
            data_test_folds.append(test_subset)

            rows.append({
                "fold": fold_id,
                "subject": int(subject),
                "test_run": int(current_test_run),
                "fine_runs": str(list(np.unique(fine_subset.run))),
                "test_runs": str(list(np.unique(test_subset.run))),
                "n_fine": len(fine_subset.y),
                "n_test": len(test_subset.y),
                "fine_class_counts": str(dict(zip(*np.unique(fine_subset.y, return_counts=True)))),
                "test_class_counts": str(dict(zip(*np.unique(test_subset.y, return_counts=True)))),
                "fine_modes": str(list(np.unique(fine_subset.mode))),
                "test_modes": str(list(np.unique(test_subset.mode))),
            })

            fold_id += 1

    fold_table = pd.DataFrame(rows)

    if len(data_fine_folds) == 0:
        raise ValueError("No se construyó ningún fold válido.")

    if debug:
        print("\n" + "=" * 70)
        print("RUN-FOLD HOLDOUT CONSTRUIDO")
        print("=" * 70)
        print(f"Sujetos solicitados: {subjects}")
        print(f"Folds generados: {len(data_fine_folds)}")
        print("\nResumen:")
        print(fold_table[["fold", "subject", "test_run", "fine_runs", "n_fine", "n_test"]])

    return data_fine_folds, data_test_folds, fold_table

def eeg_train(data_obj, mode="subject", test_size=0.1, val_size=0.1,
              classes=None, epochs=None,
              debug=False, use_class_weight=True, sfreq=160,
              kern_length=None, F1=None, make_test=False, batch=16,
              undersample_rest=False, test_run=None, checkpoint_path=None,
              Lr=None, dynamic_lr=False, Lr_grid=None,
              patience_level="medium", monitor="val_loss", dropout_rate=None,
              apply_ea=None,     # EA por sujeto antes de normalizar
              ea_eps=1e-3,       # regularización RELATIVA de eigenvalores
              apply_lsf=False,   # Aplicar filtro laplaciano de superficie
              model_type='eegnet',  # 'eegnet', 'eegnex' o 'atcnet'
              ):
    """
    Entrenamiento general de EEGNet.
 
    Parámetros de transformadas espaciales:
    -----------------
    apply_ea : bool (default False)
        Si True, aplica Euclidean Alignment (He & Wu 2019) a cada sujeto
        del dataset de entrenamiento ANTES de normalizar. Cada sujeto
        se alinea con su propia R^{-1/2} calculada sobre sus datos.
 
        Esto es necesario para que el fine-tuning posterior con
        apply_ea=True sea consistente: el modelo general aprende a
        clasificar señales ya alineadas, y el fine-tuning le presenta
        señales con la misma transformación.
 
        Latencia online: 0 — R_invsqrt se guarda en calibración y se
        aplica como multiplicación matricial antes de normalizar.
 
    ea_eps : float
        Umbral RELATIVO para eigenvalores de R: cualquier eigenvalor menor
        a `eigvals.max() * ea_eps` se regulariza antes de invertir. Al ser
        relativo, funciona sin importar la escala/unidades de la señal.
    apply_lsf : bool (default False)
        Si True, aplica el Small Laplacian Filter (Nunez et al., 1997) a
        cada ventana ANTES de normalizar. La matriz L se precalcula una
        sola vez a partir de data_obj.channels_names y se reutiliza en
        todos los sujetos y folds.  En runtime es L @ X — latencia < 0.1 ms.

        El pipeline con LSF es:
            X_crudo → L @ X → (EA opcional) → normalizar(mean, std) → Keras

        Los nombres de canales se leen automáticamente desde
        data_obj.channels_names (cargados con Selection.load()).
    """
 
    # ------------------------------------------------------------------
    # Defaults por model_type (sentinel None = no especificado por usuario)
    # ------------------------------------------------------------------
    if model_type == 'atcnet':
        if F1           is None: F1           = 16
        if epochs       is None: epochs       = 60
        if Lr           is None: Lr           = 1e-4
        if dropout_rate is None: dropout_rate = 0.25
        if apply_ea     is None: apply_ea     = True
    else:  # eegnet / eegnex
        if F1           is None: F1           = 8
        if epochs       is None: epochs       = 20
        if Lr           is None: Lr           = 1e-3
        if dropout_rate is None: dropout_rate = 0.5
        if apply_ea     is None: apply_ea     = False

    # ------------------------------------------------------------------
    # Paso 0: Ajustar variables
    # ------------------------------------------------------------------
    monitor_mode = "min" if monitor.endswith("loss") else "max"
 
    if debug:
        print("Iniciando pipeline de entrenamiento EEGNet...")
        print(f"Split de test: {make_test}")
 
    if classes is None:
        classes = len(np.unique(data_obj.y))
        if debug:
            print(f"Clases detectadas automáticamente: {classes}")
 
    if dynamic_lr and Lr_grid is None:
        warnings.warn(
            "dynamic_lr=True pero no se proporcionó Lr_grid. "
            "Se usará un schedule por defecto."
        )
 
    # ------------------------------------------------------------------
    # Paso 1: Surface Laplacian Filter por sujeto
    # Se aplica sobre X CRUDO, antes de EA y normalización.
    # La matriz L es estática (no depende del sujeto) y se precalcula
    # una sola vez para todos los folds.
    # ------------------------------------------------------------------
    if apply_lsf:
        if not hasattr(data_obj, 'channels_names') or data_obj.channels_names is None:
            raise ValueError(
                "apply_lsf=True requiere que data_obj tenga el atributo "
                "channels_names. Asegúrate de cargar el dataset con Selection.load()."
            )

        L_lap = compute_laplacian_matrix(data_obj.channels_names, debug=debug) #Calcula la matriz laplaciana

        if debug:
            print(f"[LSF] Matriz Laplaciana precalculada para {len(data_obj.channels_names)} canales.")
            print(f"[LSF] Aplicando LSF a {data_obj.X.shape[0]} ventanas...")

        data_obj = EEGSubset({
            "X":             apply_laplacian(data_obj.X, L_lap), #Aplica la transformada a cada ventana
            "y":             data_obj.y,
            "sub":           data_obj.sub,
            "trial":         data_obj.trial,
            "run":           data_obj.run,
            "mode":          data_obj.mode,
            "channels_names": data_obj.channels_names,
        })

        if debug:
            print(f"[LSF] Aplicación completada. Shape: {data_obj.X.shape}")

    # ------------------------------------------------------------------
    # Paso 1.2: EA por sujeto
    # Se aplica sobre data_obj.X crudo, antes de split y normalización,
    # para que channel_mean/channel_std se calculen sobre datos alineados.
    # ------------------------------------------------------------------
    if apply_ea:
        if debug:
            print("[EA] Aplicando Euclidean Alignment por sujeto...")
 
        subjects = np.unique(data_obj.sub)
        X_aligned = data_obj.X.copy()
 
        n_ok  = 0
        n_bad = 0
 
        for subj in subjects:
            mask = data_obj.sub == subj
            X_subj = data_obj.X[mask]           # (N_subj, C, T) — crudo
 
            try:
                R_invsqrt, R = compute_ea_matrix(X_subj, eps=ea_eps, debug=False)
                X_aligned[mask] = apply_ea_transform(X_subj, R_invsqrt)
                n_ok += 1
 
                if debug:
                    print(f"  Sujeto {subj:03d}: traza R={np.trace(R):.2f} "
                          f"| cond={np.linalg.cond(R):.1f}")
 
            except Exception as e:
                # Si un sujeto falla (señal degenerada), lo dejamos sin alinear
                # y avisamos — no abortamos todo el entrenamiento
                warnings.warn(
                    f"[EA] Sujeto {subj} falló ({e}). "
                    "Se usarán sus datos sin alinear."
                )
                n_bad += 1
 
        # Reemplazar X en data_obj con versión alineada
        # Usamos EEGSubset para no mutar el objeto original
        data_obj = EEGSubset({
            "X":             X_aligned,
            "y":             data_obj.y,
            "sub":           data_obj.sub,
            "trial":         data_obj.trial,
            "run":           data_obj.run,
            "mode":          data_obj.mode,
            "channels_names": data_obj.channels_names,
        })
 
        if debug:
            print(f"[EA] Alineación completada: {n_ok} sujetos OK, {n_bad} fallidos.")
 
    # ------------------------------------------------------------------
    # Paso 3: split de X para train y val (test va a ser sujetos nuevos)
    # ------------------------------------------------------------------
    split, info = split_eeg(
        data_obj, mode=mode,
        test_size=test_size, val_size=val_size,
        debug=debug, make_test=make_test
    )
 
    # ------------------------------------------------------------------
    # Paso 4: normalización — opera sobre datos ya alineados si apply_ea
    # channel_mean y channel_std serán consistentes con fine-tuning
    # ------------------------------------------------------------------
    norm_split, channel_mean, channel_std = normalizar_split(split)
 
    # ------------------------------------------------------------------
    # Paso 5: reshape de Keras para que sea compatible con eegnet
    # ------------------------------------------------------------------
    keras_split = prepare_keras_split(norm_split)
 
    if debug:
        print("Antes de ajustar:", split.X.train.shape,
              "Después de ajustar para Keras:", keras_split.X.train.shape)
 
    # ------------------------------------------------------------------
    # Paso 6: undersampling si es que lo pide
    # ------------------------------------------------------------------
    if undersample_rest:
        X_train, y_train, _, _, _, _ = apply_undersample_rest(keras_split)
    else:
        X_train = keras_split.X.train
        y_train = keras_split.y.train
 
    if debug:
        print("distribución de clases después de undersample rest:",
              dict(zip(*np.unique(y_train, return_counts=True))))
 
    # ------------------------------------------------------------------
    # Paso 7: construir modelo eegnet
    # ------------------------------------------------------------------
    chans      = X_train.shape[1]
    signal_len = X_train.shape[2]
 
    kern_length = int(sfreq // 2) if kern_length is None else kern_length
 
    modelo = build_eegnet(
        classes, chans, signal_len,
        kern_length=kern_length, F1=F1, dropout_rate=dropout_rate,
        model_type=model_type
    )
 
    # ------------------------------------------------------------------
    # Paso 8: compilar y agregar callbacks (early stopping y checkpoint)
    # ------------------------------------------------------------------
    modelo.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=Lr),
        metrics=['accuracy']
    )
 
    callbacks = []
 
    early_cfg = get_early_stopping_config(level=patience_level, monitor=monitor)
 
    if early_cfg is not None:
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg["monitor"],
                patience=early_cfg["patience"],
                min_delta=early_cfg["min_delta"],
                restore_best_weights=early_cfg["restore_best_weights"],
                mode=monitor_mode,
            )
        )
 
    if checkpoint_path is None:
        checkpoint_path = "EEGNet_best.keras"
 
    ckpt_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        mode=monitor_mode,
    )
    callbacks.append(ckpt_cb)
 
    if dynamic_lr:
        if Lr_grid is None:
            Lr_grid = [
                (0,                    Lr),
                (int(epochs * 0.3),    Lr * 0.1),
                (int(epochs * 0.6),    Lr * 0.01),
            ]
        callbacks.append(
            LearningRateScheduler(
                make_lr_scheduler(Lr_grid, verbose=debug),
                verbose=0,
            )
        )
 
    # ------------------------------------------------------------------
    # Paso 9: entrenar: aplica balance de clases de ser necesario 
    # ------------------------------------------------------------------
    verbose = 1 if debug else 0
 
    class_weight = None
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        classes_present = np.unique(y_train)
        weights = compute_class_weight(
            'balanced', classes=classes_present, y=y_train
        )
        class_weight = dict(zip(classes_present, weights))
 
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch,
        validation_data=(keras_split.X.val, keras_split.y.val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose,
    )
 
    # ------------------------------------------------------------------
    # Paso 10: evaluación:
    # Aplica LSF y EA al test_run si corresponde, consistente con el
    # pipeline de entrenamiento.
    # ------------------------------------------------------------------
    test_run_sorted = None

    if make_test:
        test_loss, test_acc = modelo.evaluate(
            keras_split.X.test, keras_split.y.test, verbose=0
        )
        print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    elif test_run is not None:
        if not isinstance(test_run, (list, tuple)):
            test_run = [test_run]

        if debug:
            print(f"\n[8] Evaluando {len(test_run)} subsets de test_run...")

        scored = []
        for i, subset in enumerate(test_run):

            if apply_lsf:
                L_lap_eval = compute_laplacian_matrix(subset.channels_names, debug=False)
                subset = EEGSubset({
                    "X":             apply_laplacian(subset.X, L_lap_eval),
                    "y":             subset.y,
                    "sub":           subset.sub,
                    "trial":         subset.trial,
                    "run":           subset.run,
                    "mode":          subset.mode,
                    "channels_names": subset.channels_names,
                })

            if apply_ea:
                R_invsqrt_i, _ = compute_ea_matrix(subset.X, eps=ea_eps, debug=False)
                X_rub_i = normalize_run_ea(subset, R_invsqrt_i, channel_mean, channel_std)
            else:
                X_rub_i = normalize_run(subset, channel_mean, channel_std)

            y_rub_i = subset.y

            loss_i, acc_i = modelo.evaluate(X_rub_i, y_rub_i, verbose=0)
 
            unique_sub_i = np.unique(subset.sub)
            unique_run_i = np.unique(subset.run)
            sub_label = int(unique_sub_i[0]) if len(unique_sub_i) == 1 else list(unique_sub_i)
            run_label = int(unique_run_i[0]) if len(unique_run_i) == 1 else list(unique_run_i)
 
            if debug:
                print(f"  subset[{i}] sub={sub_label} run={run_label} "
                      f"| loss={loss_i:.4f}, acc={acc_i:.4f}")
 
            scored.append((acc_i, subset))
 
        scored.sort(key=lambda x: x[0], reverse=True)
        test_run_sorted = [s for _, s in scored]
 
        if debug:
            print("\n  Orden final (mayor → menor acc):")
            for rank, (acc_v, s) in enumerate(scored):
                print(f"    Rank {rank+1}: sub={np.unique(s.sub)} "
                      f"run={np.unique(s.run)} | acc={acc_v:.4f}")
 
    else:
        print("Test set no reservado, se realizará evaluación en el set de validación.")
        test_loss, test_acc = modelo.evaluate(
            keras_split.X.val, keras_split.y.val, verbose=0
        )
        print(f"Validation loss={test_loss:.4f}, acc={test_acc:.4f}")
 
    # ------------------------------------------------------------------
    # Paso 11: info: guardar información relevante
    # ------------------------------------------------------------------
    monitor_values = history.history[monitor]
 
    best_epoch_idx = (int(np.argmin(monitor_values))
                      if monitor.endswith("loss")
                      else int(np.argmax(monitor_values)))
    best_value = monitor_values[best_epoch_idx]
 
    info["best_epoch"]         = best_epoch_idx
    info["best_monitor_value"] = float(best_value)
    info["epochs_ran"]         = len(monitor_values)
    info["monitor"]            = monitor
    info["dynamic_lr"]         = dynamic_lr
    info["Lr_grid"]            = Lr_grid
    info["initial_lr"]         = Lr
    info["batch_size"]         = batch
    info["epochs_requested"]   = epochs
    info["best_epoch_idx"]     = best_epoch_idx
    info["ea_applied"]         = apply_ea     # trazabilidad
    info["lsf_applied"]        = apply_lsf    # trazabilidad
 
    return modelo, history, channel_mean, channel_std, keras_split, test_run_sorted, info

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

def eeg_fine(base, dataset_fine, mean=None, std=None, epochs=20, debug=False,
             test_size=0.1, val_size=0.1,
             unfreeze_last_n=16, lr=5e-4, use_class_weight=True, batch_size=32,
             normalize=True, undersample_rest=False, test_run=None, make_test=False,
             dynamic_lr=False, Lr_grid=None, unfreeze_grid=None,
             patience_level="medium", monitor="val_loss",
             model_checkpoint=None,
             apply_ea=False,       # activa Euclidean Alignment por sujeto
             ea_eps=1e-3,          # regularización RELATIVA de eigenvalores de R
             apply_lsf=False,      # Surface Laplacian Filter antes de normalizar
             dynamic_unfreeze=False,  # descongelamiento progresivo estilo Wang
             ):
    """
    Fine-tuning de EEGNet para un sujeto específico.
 
    Parámetros nuevos
    -----------------
    apply_ea : bool (default False)
        Si True, calcula R^{-1/2} sobre dataset_fine.X (crudo, antes de
        normalizar) y lo aplica tanto al split de fine-tuning como al
        test_run. mean/std se calculan DESPUÉS de la alineación, por lo
        que son consistentes con el pipeline online.
 
        El pipeline online con EA es:
            X_ventana → R_invsqrt @ X_ventana → normalizar(mean, std) → Keras
 
    ea_eps : float
        Umbral RELATIVO para eigenvalores de R: cualquier eigenvalor menor
        a `eigvals.max() * ea_eps` se regulariza antes de invertir. Al ser
        relativo, funciona sin importar la escala/unidades de la señal.

    apply_lsf : bool (default False)
        Si True, aplica el Small Laplacian Filter (Nunez et al., 1997)
        sobre dataset_fine.X y test_run.X ANTES de EA y normalización.
        La matriz L se precalcula a partir de lsf_channels.

        El pipeline completo con ambos activos es:
            X_crudo → L @ X → R_invsqrt @ X → normalizar(mean, std) → Keras

        Los nombres de canales se leen automáticamente desde
        dataset_fine.channels_names (propagado desde Selection.pick_fine()).

    Retorna
    -------
    model_ft, history, mean, std, keras_split_fine, info, ea_matrix
 
    ea_matrix : dict con claves:
        "R_invsqrt" : np.ndarray (C, C) — transformación EA. None si apply_ea=False.
        "R"         : np.ndarray (C, C) — covarianza media original. None si apply_ea=False.
        "applied"   : bool — indica si EA fue aplicado.
 
    NOTA: si apply_ea=False, ea_matrix["R_invsqrt"] es None y el retorno
    es compatible con el código existente (solo hay un valor extra al final).
    """
 
    # ------------------------------------------------------------------
    # Paso 0: variables y validaciones 
    # ------------------------------------------------------------------
    monitor_mode = "min" if monitor.endswith("loss") else "max"
 
    if len(np.unique(dataset_fine.sub)) != 1:
        warnings.warn(
            "El dataset de fine-tuning contiene más de un sujeto. "
            "Asegúrate de que sólo haya un sujeto para evitar data leakage"
        )
 
    if test_run is None and make_test is False:
        raise ValueError(
            "Si no se va a reservar un test set, entonces test_run no puede "
            "ser None. Considera reservar un test set o especificar test_run"
        )
 
    # ------------------------------------------------------------------
    # Paso 1: Surface Laplacian Filter
    # Se aplica sobre X CRUDO, antes de EA y normalización, para que
    # mean/std se calculen sobre señales ya filtradas espacialmente.
    # La matriz L es estática: se precalcula una vez por llamada.
    # ------------------------------------------------------------------
    if apply_lsf:
        if not hasattr(dataset_fine, 'channels_names') or dataset_fine.channels_names is None:
            raise ValueError(
                "apply_lsf=True requiere que dataset_fine tenga el atributo "
                "channels_names. Propágalo al construir el EEGSubset desde pick_fine()."
            )

        L_lap = compute_laplacian_matrix(dataset_fine.channels_names, debug=debug)

        if debug:
            print(f"[LSF] Aplicando Laplacian filter a dataset_fine "
                  f"({dataset_fine.X.shape[0]} ventanas)...")

        dataset_fine = EEGSubset({
            "X":             apply_laplacian(dataset_fine.X, L_lap),
            "y":             dataset_fine.y,
            "sub":           dataset_fine.sub,
            "trial":         dataset_fine.trial,
            "run":           dataset_fine.run,
            "mode":          dataset_fine.mode,
            "channels_names": dataset_fine.channels_names,
        })

        # Aplicar también al test_run si existe
        if test_run is not None:
            test_run = EEGSubset({
                "X":             apply_laplacian(test_run.X, L_lap),
                "y":             test_run.y,
                "sub":           test_run.sub,
                "trial":         test_run.trial,
                "run":           test_run.run,
                "mode":          test_run.mode,
                "channels_names": test_run.channels_names,
            })

        if debug:
            print(f"[LSF] Aplicación completada.")

    # ------------------------------------------------------------------
    # Paso 2: Euclidean Alignment
    # Debe aplicarse sobre X CRUDO, antes de cualquier normalización,
    # para que mean/std calculados después sean consistentes con online.
    # ------------------------------------------------------------------
    R_invsqrt = None
    R_cov     = None
 
    if apply_ea:
        if debug:
            print("[EA] Calculando R^{-1/2} sobre dataset_fine.X crudo...")
 
        R_invsqrt, R_cov = compute_ea_matrix(
            dataset_fine.X, eps=ea_eps, debug=debug
        )
 
        # Alinear fine
        dataset_fine = EEGSubset({
            "X":             apply_ea_transform(dataset_fine.X, R_invsqrt),
            "y":             dataset_fine.y,
            "sub":           dataset_fine.sub,
            "trial":         dataset_fine.trial,
            "run":           dataset_fine.run,
            "mode":          dataset_fine.mode,
            "channels_names": dataset_fine.channels_names,
        })
 
        # Alinear test_run si existe
        if test_run is not None:
            test_run = EEGSubset({
                "X":             apply_ea_transform(test_run.X, R_invsqrt),
                "y":             test_run.y,
                "sub":           test_run.sub,
                "trial":         test_run.trial,
                "run":           test_run.run,
                "mode":          test_run.mode,
                "channels_names": test_run.channels_names,
            })
 
        if debug:
            print(f"[EA] Alineación aplicada. "
                  f"Traza R antes: {np.trace(R_cov):.3f} | "
                  f"Ideal (=n_canales): {dataset_fine.X.shape[1]}")
            print(f"[EA] Número de condición de R: {np.linalg.cond(R_cov):.1f}")
 
    ea_matrix = {
        "R_invsqrt": R_invsqrt,
        "R":         R_cov,
        "applied":   apply_ea,
    }
 
    # ------------------------------------------------------------------
    # Paso 3: splitear el dataset nuevamente, ahora por trials entre los runs de entrenamiento y validación
    # ------------------------------------------------------------------
    split_fine, info = split_eeg(
        dataset_fine, mode="trial",
        test_size=test_size, val_size=val_size,
        debug=debug, make_test=make_test
    )
 
    if debug:
        print("Tamaño del train:", split_fine.X.train.shape,
              "Tamaño del val:",  split_fine.X.val.shape,
              "Tamaño del test:", split_fine.X.test.shape)
 
    # ------------------------------------------------------------------
    # Paso 4: normalización 
    # ------------------------------------------------------------------
    if normalize:
        if mean is None or std is None:
            print("Advertencia: Se normaliza con los datos del usuario de ajuste")
            norm_split, mean, std = normalizar_split(split_fine)
        else:
            X_train = normalizar_por_canal(split_fine.X.train, mean, std)
            X_val   = normalizar_por_canal(split_fine.X.val,   mean, std)
            X_test  = normalizar_por_canal(split_fine.X.test,  mean, std)
 
            norm_split = DataSplit(
                X_train, X_val, X_test,
                split_fine.y.train, split_fine.y.val, split_fine.y.test,
                split_fine.sub.train,   split_fine.sub.val,   split_fine.sub.test,
                split_fine.trial.train, split_fine.trial.val, split_fine.trial.test,
                split_fine.run.train,   split_fine.run.val,   split_fine.run.test,
                split_fine.mode.train,  split_fine.mode.val,  split_fine.mode.test,
            )
    else:
        norm_split, mean, std = normalizar_split(split_fine)
 
    # ------------------------------------------------------------------
    # Paso 5: reshape Keras
    # ------------------------------------------------------------------
    keras_split_fine = prepare_keras_split(norm_split)
 
    # ------------------------------------------------------------------
    # Paso 5.5: undersampling 
    # ------------------------------------------------------------------
    if undersample_rest:
        X_fine_train, y_fine_train, _, _, _, _ = apply_undersample_rest(keras_split_fine)
    else:
        X_fine_train = keras_split_fine.X.train
        y_fine_train = keras_split_fine.y.train
 
    if debug:
        print("Antes de normalizar:", split_fine.X.train.shape,
              "Después de ajustar para Keras:", X_fine_train.shape)
 
    # ------------------------------------------------------------------
    # Paso 6: clonar modelo y ajustar capas entrenables
    # ------------------------------------------------------------------
    model_ft = clone_model(base)
    model_ft.set_weights(base.get_weights())

    callbacks = []

    # Obtener TODAS las capas (incluyendo submodelos wrapeados)
    all_layers = []
    for layer in model_ft.layers:
        if hasattr(layer, 'layers'):
            all_layers.extend(layer.layers)
        else:
            all_layers.append(layer)
    n_layers = len(all_layers)

    if not dynamic_unfreeze:
        if unfreeze_last_n is None or unfreeze_last_n > n_layers:
            for layer in all_layers:
                layer.trainable = True
        else:
            for layer in all_layers:
                layer.trainable = False
            for layer in all_layers[-unfreeze_last_n:]:
                layer.trainable = True
        trainable_count = sum(1 for l in all_layers if l.trainable)
        print(f"{trainable_count} capas desbloqueadas de {n_layers} (total, incluyendo submodelos)")
    else:
        for layer in all_layers:
            layer.trainable = True
        print(f"Dynamic unfreeze: {n_layers}→{n_layers//2}→3 capas")
 
    # ------------------------------------------------------------------
    # Paso 7: compilar 
    # ------------------------------------------------------------------
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
 
    verbose = 1 if debug else 0
 
    # ------------------------------------------------------------------
    # Paso 8: callbacks nuevamente
    # ------------------------------------------------------------------
    if model_checkpoint is None:
        model_checkpoint = "EEGNet_finetuned_subject.keras"
 
    early_cfg = get_early_stopping_config(level=patience_level, monitor=monitor)
 
    if early_cfg is not None:
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg["monitor"],
                patience=early_cfg["patience"],
                min_delta=early_cfg["min_delta"],
                restore_best_weights=early_cfg["restore_best_weights"],
                mode=monitor_mode,
            )
        )
 
    ckpt_cb = ModelCheckpoint(
        filepath=model_checkpoint,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode=monitor_mode,
    )
    callbacks.append(ckpt_cb)
 
    if dynamic_lr:
        if Lr_grid is None:
            Lr_grid = [
                (0,                    lr),
                (int(epochs * 0.2),    lr * 0.5),
                (int(epochs * 0.5),    lr * 0.1),
            ]
        callbacks.append(
            LearningRateScheduler(
                make_lr_scheduler(Lr_grid, verbose=debug),
                verbose=0,
            )
        )
 
    # ------------------------------------------------------------------
    # Paso 9: entrenamiento: SSFT
    # ------------------------------------------------------------------
    if dynamic_unfreeze and epochs >= 6:
        phase_epochs = [epochs // 3, epochs // 3, epochs - 2 * (epochs // 3)]
        unfreeze_phases = [n_layers, max(n_layers // 2, 3), 3]

        all_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        for phase_i, (ph_epochs, ph_unfreeze) in enumerate(zip(phase_epochs, unfreeze_phases)):
            for layer in all_layers:
                layer.trainable = False
            for layer in all_layers[-ph_unfreeze:]:
                layer.trainable = True

            lr_phase = lr if phase_i == 0 else lr * (0.5 ** phase_i)
            model_ft.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=Adam(learning_rate=lr_phase),
                metrics=['accuracy']
            )

            if debug:
                print(f"  Fase {phase_i+1}: {ph_unfreeze} capas, lr={lr_phase:.1e}, {ph_epochs} epochs")

            h = model_ft.fit(
                X_fine_train, y_fine_train,
                validation_data=(keras_split_fine.X.val, keras_split_fine.y.val),
                callbacks=callbacks,
                epochs=ph_epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                verbose=verbose,
            )
            for k in all_history:
                all_history[k].extend(h.history.get(k, []))

        class _MergedHistory:
            def __init__(self, d): self.history = d
        history = _MergedHistory(all_history)
    else:
        history = model_ft.fit(
            X_fine_train, y_fine_train,
            validation_data=(keras_split_fine.X.val, keras_split_fine.y.val),
            callbacks=callbacks,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=verbose,
        )
 
    # ------------------------------------------------------------------
    # Paso 10: evaluación en test 
    # ------------------------------------------------------------------
    if make_test:
        print("Se realiza evaluación en el set del dataset.")
        test_loss, test_acc = model_ft.evaluate(
            keras_split_fine.X.test, keras_split_fine.y.test, verbose=0
        )
        print(f"[Fine-tune] Test loss={test_loss:.4f}, acc={test_acc:.4f}")
 
    elif test_run is not None:
        # normalize_run opera sobre test_run.X, que ya está alineado
        # si apply_ea=True, por lo que mean/std son consistentes
        X_test = normalize_run(test_run, mean, std)
 
        if debug:
            print("Antes de normalizar:", test_run.X.shape,
                  "Después de ajustar para Keras:", X_test.shape)
 
        keras_split_fine.X.test     = X_test
        keras_split_fine.y.test     = test_run.y
        keras_split_fine.sub.test   = test_run.sub
        keras_split_fine.trial.test = test_run.trial
        keras_split_fine.run.test   = test_run.run
        keras_split_fine.mode.test  = test_run.mode
 
        test_loss, test_acc = model_ft.evaluate(X_test, test_run.y, verbose=0)
        print(f"[Fine-tune] Validation loss={test_loss:.4f}, acc={test_acc:.4f}")
 
    # ------------------------------------------------------------------
    # Paso 11: info: guardamos la información relevante
    # ------------------------------------------------------------------
    monitor_values = history.history[monitor]
 
    best_epoch_idx = (int(np.argmin(monitor_values))
                      if monitor.endswith("loss")
                      else int(np.argmax(monitor_values)))
 
    info["best_epoch"]         = best_epoch_idx
    info["best_monitor_value"] = float(monitor_values[best_epoch_idx])
    info["epochs_ran"]         = len(monitor_values)
    info["monitor"]            = monitor
    info["dynamic_lr"]         = dynamic_lr
    info["Lr_grid"]            = Lr_grid
    info["initial_lr"]         = lr
    info["batch_size"]         = batch_size
    info["epochs_requested"]   = epochs
    info["unfreeze_grid"]      = unfreeze_grid
    info["ea_applied"]         = apply_ea    # trazabilidad en CSV
    info["lsf_applied"]        = apply_lsf   # trazabilidad en CSV
 
    return model_ft, history, mean, std, keras_split_fine, info, ea_matrix
 

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


def build_list(
    fine_obj,
    test_obj,
    subjects=None,
    n_subjects=None,
    seed=42,
    debug=True
):
    """
    Construye listas pareadas fine/test por sujeto, sin folds por run.

    Para datasets donde cada sujeto tiene una sola sesion de fine-tuning
    y una sola sesion de test (sin multiples runs), no se necesita
    leave-one-run-out. Esta funcion simplemente asocia el fine del
    sujeto i con su test correspondiente.

    Retorna:
        data_fine_list : lista de EEGSubset, uno por sujeto
        data_test_list : lista de EEGSubset, uno por sujeto
        fold_table     : DataFrame con metadata de cada par
    """

    required_attrs = ["X", "y", "sub", "trial", "run", "mode"]
    for attr in required_attrs:
        if not hasattr(fine_obj, attr) or getattr(fine_obj, attr) is None:
            raise ValueError(f"fine_obj no tiene atributo '{attr}' o es None")
        if not hasattr(test_obj, attr) or getattr(test_obj, attr) is None:
            raise ValueError(f"test_obj no tiene atributo '{attr}' o es None")

    if fine_obj.X.shape[1:] != test_obj.X.shape[1:]:
        raise ValueError(
            f"fine_obj y test_obj tienen distinta forma de ventana. "
            f"fine: {fine_obj.X.shape[1:]} | test: {test_obj.X.shape[1:]}"
        )

    if hasattr(fine_obj, "label_map") and hasattr(test_obj, "label_map"):
        if fine_obj.label_map != test_obj.label_map:
            raise ValueError(
                f"label_map distintos. fine: {fine_obj.label_map} | test: {test_obj.label_map}"
            )

    fine_subjects = np.unique(fine_obj.sub)
    test_subjects = np.unique(test_obj.sub)
    common_subjects = np.intersect1d(fine_subjects, test_subjects)

    if len(common_subjects) == 0:
        raise ValueError("No hay sujetos comunes entre fine_obj y test_obj.")

    if subjects is None:
        subjects = common_subjects
        if n_subjects is not None:
            if n_subjects > len(common_subjects):
                raise ValueError(
                    f"n_subjects={n_subjects} excede sujetos comunes: {len(common_subjects)}"
                )
            rng = np.random.default_rng(seed)
            subjects = rng.choice(common_subjects, size=n_subjects, replace=False)

    subjects = np.sort(np.asarray(subjects))

    missing = [s for s in subjects if s not in common_subjects]
    if missing:
        raise ValueError(f"Sujetos no encontrados en ambos objetos: {missing}")

    def make_subset(obj, mask):
        return EEGSubset({
            "X":     obj.X[mask],     "y":    obj.y[mask],
            "sub":   obj.sub[mask],   "trial": obj.trial[mask],
            "run":   obj.run[mask],   "mode":  obj.mode[mask],
            "channels_names": getattr(obj, 'channels_names', None),
        })

    data_fine_list = []
    data_test_list = []
    rows = []

    for subject in subjects:
        fine_subset = make_subset(fine_obj, fine_obj.sub == subject)
        test_subset = make_subset(test_obj, test_obj.sub == subject)

        if len(fine_subset.y) == 0 or len(test_subset.y) == 0:
            if debug:
                print(f"Sujeto {subject} omitido: fine={len(fine_subset.y)}, test={len(test_subset.y)}")
            continue

        data_fine_list.append(fine_subset)
        data_test_list.append(test_subset)

        rows.append({
            "idx": len(data_fine_list) - 1,
            "subject": int(subject),
            "n_fine": len(fine_subset.y),
            "n_test": len(test_subset.y),
            "fine_class_counts": str(dict(zip(*np.unique(fine_subset.y, return_counts=True)))),
            "test_class_counts": str(dict(zip(*np.unique(test_subset.y, return_counts=True)))),
        })

    if len(data_fine_list) == 0:
        raise ValueError("No se construyo ningun par fine/test valido.")

    fold_table = pd.DataFrame(rows)

    if debug:
        print("\n" + "=" * 70)
        print("BUILD_LIST: PAREJAS FINE/TEST CONSTRUIDAS")
        print("=" * 70)
        print(f"Sujetos: {len(data_fine_list)}")
        print(fold_table[["idx", "subject", "n_fine", "n_test"]])

    return data_fine_list, data_test_list, fold_table


#==== Vibecoding abajo ====

def evaluate(
    modelo,
    keras_split=None,
    label_map=None,
    title="EEGNet",
    all_data=False,
    test_run=None,
    mean=None,
    std=None,
    plot=True,
    verbose=True,
    debug=False,
):
    """
    Evalúa un modelo EEGNet de forma más completa para experimentos tipo paper.

    Usa, en orden de prioridad:
    1) test_run si se entrega — puede ser:
       a) Un EEGSubset sin normalizar → requiere mean y std para normalizar internamente.
       b) Un DataSplit ya preparado para Keras (test_run.X tiene shape (N,C,T,1)).
    2) todo train+val+test si all_data=True
    3) keras_split.X.test / keras_split.y.test por defecto

    Parámetros
    ----------
    mean, std : arrays de normalización del modelo (channel_mean, channel_std).
                Solo necesarios si test_run es un EEGSubset sin normalizar.
    debug     : imprime información del proceso de selección de datos.
    """

    if test_run is not None:
        # Detectamos si es un EEGSubset crudo (X shape 3D: N, C, T)
        # o ya está en formato Keras 4D (N, C, T, 1)
        raw_X = test_run.X
        y_test = test_run.y

        if raw_X.ndim == 3:
            # EEGSubset sin normalizar → normalizar y ajustar
            if mean is None or std is None:
                raise ValueError(
                    "test_run es un EEGSubset sin normalizar (X.ndim==3). "
                    "Debes pasar mean y std del modelo para normalizarlo."
                )
            if debug:
                print(f"[evaluate] test_run es EEGSubset crudo. Normalizando con mean/std del modelo...")
                print(f"  X shape antes : {raw_X.shape}")
            X_test = normalize_run(test_run, mean, std)
            if debug:
                print(f"  X shape después: {X_test.shape}")
        elif raw_X.ndim == 4:
            # Ya está listo para Keras
            if debug:
                print(f"[evaluate] test_run ya está en formato Keras 4D: {raw_X.shape}")
            X_test = raw_X
        else:
            raise ValueError(
                f"test_run.X tiene shape inesperado: {raw_X.shape}. "
                "Esperado 3D (EEGSubset crudo) o 4D (formato Keras)."
            )

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
    split=None,
    info=None,
    path=None,
    name="model_EEGNet",
    ea_matrix=None,    # NUEVO: dict {"R_invsqrt": ..., "R": ..., "applied": bool}
):
    """
    Guarda:
        <name>.keras  -> modelo entrenado
        <name>.npz    -> parámetros de inferencia online
 
    Parámetros nuevos
    -----------------
    ea_matrix : dict o None
        Si se entrega (y ea_matrix["applied"] es True), el .npz incluye
        R_invsqrt y R_diag, necesarios para el pipeline online con EA.
 
        Pipeline online sin EA:
            X → normalizar(mean, std) → Keras
 
        Pipeline online con EA  (cuando ea_applied=True en el .npz):
            X → R_invsqrt @ X → normalizar(mean, std) → Keras
 
    El campo "ea_applied" en el .npz actúa como flag para que el código
    de inferencia online sepa qué pipeline aplicar sin asumir nada.
    """
 
    if path is None:
        path = os.getcwd()
 
    os.makedirs(path, exist_ok=True)
 
    if name.endswith(".keras") or name.endswith(".npz"):
        raise ValueError("name debe ir sin extensión.")
 
    n_outputs = model.output_shape[-1]
 
    if sorted(label_map.keys()) != list(range(n_outputs)):
        raise ValueError(
            f"label_map incompatible con el modelo. "
            f"Modelo: {n_outputs} salidas | label_map: {label_map}"
        )
 
    class_names = np.array(
        [label_map[i] for i in range(n_outputs)],
        dtype=str
    )
 
    simp_class_names = []
    for name_in in class_names:
        if name_in.endswith("_m") or name_in.endswith("_i"):
            name_in = name_in[:-2]
        simp_class_names.append(name_in)
 
    model_path  = os.path.join(path, f"{name}.keras")
    params_path = os.path.join(path, f"{name}.npz")
 
    model.save(model_path)
 
    # ------------------------------------------------------------------
    # Construir el dict de parámetros del .npz
    # ------------------------------------------------------------------
    save_dict = dict(
        mean        = np.asarray(mean, dtype=np.float32),
        std         = np.asarray(std,  dtype=np.float32),
        class_names = simp_class_names,
        info        = info,
        split       = split,
        ea_applied  = np.array([False]),   # default: sin EA
    )
 
    # Si se entregó ea_matrix con alineación aplicada, añadir R_invsqrt
    ea_was_applied = (
        ea_matrix is not None
        and ea_matrix.get("applied", False)
        and ea_matrix.get("R_invsqrt") is not None
    )
 
    if ea_was_applied:
        save_dict["ea_applied"]  = np.array([True])
        save_dict["R_invsqrt"]   = np.asarray(
            ea_matrix["R_invsqrt"], dtype=np.float32
        )
        # R_diag solo para diagnóstico offline — no se usa en inferencia
        if ea_matrix.get("R") is not None:
            save_dict["R_diag"] = np.asarray(
                np.diag(ea_matrix["R"]), dtype=np.float32
            )
 
    np.savez_compressed(params_path, **save_dict)
 
    # ------------------------------------------------------------------
    # Print resumen
    # ------------------------------------------------------------------
    print(f"Modelo guardado    : {model_path}")
    print(f"Parámetros guardados: {params_path}")
    print(f"Clases del modelo  : {dict(enumerate(class_names))}")
    if ea_was_applied:
        print(f"EA incluida        : R_invsqrt shape={ea_matrix['R_invsqrt'].shape}")
    else:
        print(f"EA incluida        : No")

 
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


def evaluate_moabb(
    modelo,
    test_subsets,
    mean,
    std,
    label_map=None,
    fold_table=None,
    test_result="subject",
    plot=False,
    verbose=False,
    debug=False,
    apply_ea=False,
    ea_eps=1e-3,
    apply_lsf=False,
):
    """
    Evalúa un modelo sobre múltiples EEGSubset de test y genera una tabla limpia
    para informe/paper.

    test_result:
        "run"     -> una fila por run/fold.
        "subject" -> una fila por sujeto con mean ± std.
        "all"     -> una sola fila global con mean ± std.

    apply_ea : bool (default False)
        Si True, alinea cada test_subset con Euclidean Alignment ANTES de
        normalizar, usando su propia covarianza (R_invsqrt calculado sobre
        sus propios datos, sin labels). Esto es necesario para evaluar de
        forma justa un modelo entrenado con apply_ea=True en eeg_train,
        ya que ese modelo aprendió sobre señales alineadas y, sin alinear
        también el test, hay un desajuste de distribución que rompe la
        evaluación (accuracy ≈ azar).

    ea_eps : float
        Umbral relativo de regularización de eigenvalores, igual que en
        eeg_train / compute_ea_matrix.

    apply_lsf : bool (default False)
        Si True, aplica el mismo Small Laplacian Filter usado en
        eeg_train (apply_lsf=True) a cada test_subset ANTES de EA y de
        normalizar. Requiere que cada subset tenga `channels_names`.
        Pipeline resultante: LSF -> (EA) -> normalizar.

    Columnas finales:
        subject | run | fined | accuracy | macro_f1 | balanced_accuracy
    """

    import numpy as np
    import pandas as pd

    if test_subsets is None or len(test_subsets) == 0:
        raise ValueError("test_subsets está vacío o es None.")

    valid_modes = {"run", "subject", "all", None}
    if test_result not in valid_modes:
        raise ValueError(
            f"test_result='{test_result}' no válido. Usa 'run', 'subject', 'all' o None."
        )

    if test_result is None:
        test_result = "run"

    def _fmt(mean_value, std_value=None, decimals=3):
        if std_value is None:
            return f"{mean_value:.{decimals}f}"
        return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"

    def _safe_unique_one(arr):
        vals = np.unique(arr)
        if len(vals) == 1:
            return vals[0]
        return str(list(vals))

    def _get_fined_from_fold_table(i):
        if fold_table is None:
            return "unknown"

        if "fold" in fold_table.columns:
            row = fold_table[fold_table["fold"] == i]
            if len(row) == 0:
                return "unknown"
            row = row.iloc[0]
        else:
            if i >= len(fold_table):
                return "unknown"
            row = fold_table.iloc[i]

        if "fine_runs" in row:
            return row["fine_runs"]

        if "fined" in row:
            return row["fined"]

        return "unknown"

    def _eval_subset(subset, title=""):
        if apply_lsf:
            if not hasattr(subset, 'channels_names') or subset.channels_names is None:
                raise ValueError(
                    "apply_lsf=True requiere que el test_subset tenga el "
                    "atributo 'channels_names'."
                )
            L_lap = compute_laplacian_matrix(subset.channels_names, debug=debug)
            subset = EEGSubset({
                "X":             apply_laplacian(subset.X, L_lap),
                "y":             subset.y,
                "sub":           subset.sub,
                "trial":         subset.trial,
                "run":           subset.run,
                "mode":          subset.mode,
                "channels_names": subset.channels_names,
            })

        if apply_ea:
            R_invsqrt, _ = compute_ea_matrix_from_subset(subset, eps=ea_eps, debug=debug)
            X_norm = normalize_run_ea(subset, R_invsqrt, mean, std)
        else:
            X_norm = normalize_run(subset, mean, std)
        y_true = subset.y

        if debug:
            print(f"Evaluando {title}")
            print("X:", X_norm.shape)
            print("Sujeto:", np.unique(subset.sub))
            print("Run:", np.unique(subset.run))
            print("Clases:", dict(zip(*np.unique(subset.y, return_counts=True))))

        y_prob = modelo.predict(X_norm, verbose=0)

        if label_map is not None:
            labels = np.array(sorted(label_map.keys()))
            class_names = [label_map[i] for i in labels]
        else:
            labels = np.arange(y_prob.shape[1])
            class_names = [str(i) for i in labels]

        results = evaluar_modelo_multiclase(
            y_true=y_true,
            y_prob=y_prob,
            labels=labels,
            class_names=class_names,
            title_prefix=title,
            plot=plot,
            verbose=verbose
        )

        return results

    # =====================================================
    # 1) Evaluación por run/fold
    # =====================================================

    rows = []
    rows_raw = []

    for i, subset in enumerate(test_subsets):

        subject_i = _safe_unique_one(subset.sub)
        run_i = _safe_unique_one(subset.run)
        fined_i = _get_fined_from_fold_table(i)

        metrics_i = _eval_subset(
            subset,
            title=f"subject={subject_i} | run={run_i}"
        )

        row = {
            "fold": i,
            "subject": int(subject_i) if isinstance(subject_i, (int, np.integer)) else subject_i,
            "run": int(run_i) if isinstance(run_i, (int, np.integer)) else run_i,
            "fined": fined_i,
            "accuracy_raw": metrics_i["accuracy"],
            "macro_f1_raw": metrics_i["macro_f1"],
            "balanced_accuracy_raw": metrics_i["balanced_accuracy"],
        }

        rows.append(row)

        rows_raw.append({
            "fold": i,
            "subject": subject_i,
            "run": run_i,
            "fined": fined_i,
            "metrics": metrics_i,
        })

        if debug:
            print(
                f"[fold {i}] subject={subject_i}, run={run_i}, fined={fined_i} | "
                f"acc={metrics_i['accuracy']:.3f}, "
                f"macro_f1={metrics_i['macro_f1']:.3f}, "
                f"bal_acc={metrics_i['balanced_accuracy']:.3f}"
            )

    df_runs = pd.DataFrame(rows)

    # =====================================================
    # 2) Salida por run
    # =====================================================

    if test_result == "run":
        df_out = df_runs.copy()

        df_out["accuracy"] = df_out["accuracy_raw"].map(lambda x: _fmt(x))
        df_out["macro_f1"] = df_out["macro_f1_raw"].map(lambda x: _fmt(x))
        df_out["balanced_accuracy"] = df_out["balanced_accuracy_raw"].map(lambda x: _fmt(x))

        df_out = df_out[
            [
                "subject",
                "run",
                "fined",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
            ]
        ]

        return df_out, df_runs, rows_raw

    # =====================================================
    # 3) Salida agregada por sujeto
    # =====================================================

    if test_result == "subject":

        subject_rows = []

        for subject, df_s in df_runs.groupby("subject"):

            acc_mean = df_s["accuracy_raw"].mean()
            acc_std = df_s["accuracy_raw"].std(ddof=1) if len(df_s) > 1 else 0.0

            f1_mean = df_s["macro_f1_raw"].mean()
            f1_std = df_s["macro_f1_raw"].std(ddof=1) if len(df_s) > 1 else 0.0

            bal_mean = df_s["balanced_accuracy_raw"].mean()
            bal_std = df_s["balanced_accuracy_raw"].std(ddof=1) if len(df_s) > 1 else 0.0

            subject_rows.append({
                "subject": subject,
                "run": "all",
                "fined": "all folds",
                "accuracy": _fmt(acc_mean, acc_std),
                "macro_f1": _fmt(f1_mean, f1_std),
                "balanced_accuracy": _fmt(bal_mean, bal_std),

                # columnas numéricas útiles para ordenar/exportar
                "accuracy_mean": round(acc_mean, 3),
                "accuracy_std": round(acc_std, 3),
                "macro_f1_mean": round(f1_mean, 3),
                "macro_f1_std": round(f1_std, 3),
                "balanced_accuracy_mean": round(bal_mean, 3),
                "balanced_accuracy_std": round(bal_std, 3),
            })

        df_out = pd.DataFrame(subject_rows)

        df_out = df_out.sort_values(
            by=["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
            ascending=False
        ).reset_index(drop=True)

        return df_out, df_runs, rows_raw

    # =====================================================
    # 4) Salida agregada global
    # =====================================================

    if test_result == "all":

        acc_mean = df_runs["accuracy_raw"].mean()
        acc_std = df_runs["accuracy_raw"].std(ddof=1) if len(df_runs) > 1 else 0.0

        f1_mean = df_runs["macro_f1_raw"].mean()
        f1_std = df_runs["macro_f1_raw"].std(ddof=1) if len(df_runs) > 1 else 0.0

        bal_mean = df_runs["balanced_accuracy_raw"].mean()
        bal_std = df_runs["balanced_accuracy_raw"].std(ddof=1) if len(df_runs) > 1 else 0.0

        df_out = pd.DataFrame([{
            "subject": "all",
            "run": "all",
            "fined": "all folds",
            "accuracy": _fmt(acc_mean, acc_std),
            "macro_f1": _fmt(f1_mean, f1_std),
            "balanced_accuracy": _fmt(bal_mean, bal_std),

            # numéricas
            "accuracy_mean": round(acc_mean, 3),
            "accuracy_std": round(acc_std, 3),
            "macro_f1_mean": round(f1_mean, 3),
            "macro_f1_std": round(f1_std, 3),
            "balanced_accuracy_mean": round(bal_mean, 3),
            "balanced_accuracy_std": round(bal_std, 3),
        }])

        return df_out, df_runs, rows_raw

def summarize_finetuning_by_subject(df_runfold):
    def fmt(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    rows = []

    for subject, df_s in df_runfold.groupby("subject"):

        acc_mean = df_s["accuracy"].mean()
        acc_std = df_s["accuracy"].std(ddof=1) if len(df_s) > 1 else 0.0

        f1_mean = df_s["macro_f1"].mean()
        f1_std = df_s["macro_f1"].std(ddof=1) if len(df_s) > 1 else 0.0

        bal_mean = df_s["balanced_accuracy"].mean()
        bal_std = df_s["balanced_accuracy"].std(ddof=1) if len(df_s) > 1 else 0.0

        rows.append({
            "subject": subject,
            "run": "all",
            "fined": "all folds",
            "accuracy": fmt(acc_mean, acc_std),
            "macro_f1": fmt(f1_mean, f1_std),
            "balanced_accuracy": fmt(bal_mean, bal_std),
            "accuracy_mean": round(acc_mean, 3),
            "macro_f1_mean": round(f1_mean, 3),
            "balanced_accuracy_mean": round(bal_mean, 3)
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=False
    ).reset_index(drop=True)

    return out

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

# ============================================================
# Alineamiento euclidiano (Sirve para transfer learning una vez tenemos el modelo general ya hecho)
# He & Wu, 2019 — "Transfer Learning for Brain–Computer Interfaces:
# A Euclidean Space Data Alignment Approach"
#
# Dado que muchas firmas eléctricas de las señales de los sujetos de prueba pueden verse alejadas del modelo de prueba
# Se ajusta la covarianza de entrenar en el finetuning para que quede más cerca del mean entregado por el modelo 
# Dentro de estas funcioones ROBADAS, se ajiustan los datos del X parta ajustarlos más al mean del modelo general, para luego entrenar como si nada 
#
# FLUJO DE USO:
#   1. Durante fine-tuning: compute_ea_matrix(fine_subset.X)  -> R_invsqrt
#   2. apply_ea(fine_subset.X, R_invsqrt)  antes de normalizar_por_canal
#   3. Durante test online:  apply_ea(X_ventana, R_invsqrt)   antes de normalizar
# ============================================================




def compute_ea_matrix(X, eps=1e-3, debug=False):
    """
    Calcula R^{-1/2} a partir de los datos de calibración del sujeto.

    Parámetros
    ----------
    X : np.ndarray, shape (N, C, T)
        Señal cruda del sujeto (runs de fine-tuning, SIN normalizar).
        N = ventanas, C = canales, T = muestras por ventana.

    eps : float
        Umbral RELATIVO para eigenvalores: cualquier eigenvalor menor a
        `eigvals.max() * eps` se reemplaza por ese piso antes de invertir.
        Al ser relativo a la escala de R, funciona igual sin importar las
        unidades de la señal (volts crudos, microvolts, datos normalizados, etc.).
        Un valor absoluto fijo (p.ej. 1e-10) puede terminar siendo del mismo
        orden que TODOS los eigenvalores de R si la señal está en volts
        (~1e-10 a 1e-17), lo que rompe el blanqueo en vez de regularizarlo.

    Retorna
    -------
    R_invsqrt : np.ndarray, shape (C, C)
        Matriz de transformación. Guárdala junto a mean/std del modelo.

    R : np.ndarray, shape (C, C)
        Covarianza media (útil para diagnóstico).
    """

    assert X.ndim == 3, f"X debe ser (N, C, T), recibido {X.shape}"
    N, C, T = X.shape

    # Covarianza por ventana: X_i @ X_i^T / T
    # einsum es más rápido que el loop para N grande
    covs = np.einsum('nct,ndt->ncd', X, X) / T   # (N, C, C)
    R = covs.mean(axis=0)                          # (C, C)

    # Eigendecomposición simétrica (más estable numéricamente que scipy.sqrtm)
    eigvals, eigvecs = np.linalg.eigh(R)

    threshold = eigvals.max() * eps

    if debug:
        print(f"[EA] Eigenvalores de R: min={eigvals.min():.3e}, max={eigvals.max():.3e}")
        print(f"[EA] Traza de R antes de alinear: {np.trace(R):.3e}")
        n_small = np.sum(eigvals < threshold)
        if n_small > 0:
            print(f"[EA] Advertencia: {n_small} eigenvalor(es) < umbral={threshold:.3e}, se aplicará regularización.")

    eigvals_safe = np.maximum(eigvals, threshold)
    R_invsqrt = eigvecs @ np.diag(eigvals_safe ** -0.5) @ eigvecs.T

    if debug:
        # Verificación: covarianza después de alinear debe ser ≈ I
        X_check = apply_ea_transform(X[:min(N, 50)], R_invsqrt)
        covs_check = np.einsum('nct,ndt->ncd', X_check, X_check) / T
        R_check = covs_check.mean(axis=0)
        off_diag_err = np.abs(R_check - np.eye(C)).mean()
        print(f"[EA] Error off-diagonal post-alineación (debe ser ≈ 0): {off_diag_err:.4f}")

    return R_invsqrt, R


def apply_ea_transform(X, R_invsqrt):
    """
    Aplica la transformación EA a un batch de ventanas o a una sola ventana.

    Parámetros
    ----------
    X : np.ndarray
        shape (N, C, T)  — batch de ventanas
        shape (C, T)     — una sola ventana (modo online)

    R_invsqrt : np.ndarray, shape (C, C)
        Calculado con compute_ea_matrix() sobre los datos de calibración.

    Retorna
    -------
    X_aligned : np.ndarray, misma shape que X
    """
    if X.ndim == 2:
        # Modo online: una sola ventana (C, T)
        return R_invsqrt @ X

    # Modo batch: (N, C, T)
    # einsum equivale a: para cada n, R_invsqrt @ X[n]
    return np.einsum('cd,ndt->nct', R_invsqrt, X)


def compute_ea_matrix_from_subset(subset, eps=1e-10, debug=False):
    """
    Wrapper conveniente que acepta un EEGSubset directamente.

    Uso:
        R_invsqrt, R = compute_ea_matrix_from_subset(data_fine_folds[i], debug=True)
    """
    assert hasattr(subset, 'X'), "subset debe ser un EEGSubset con atributo X"
    assert subset.X.ndim == 3, f"subset.X debe ser (N, C, T), recibido {subset.X.shape}"

    return compute_ea_matrix(subset.X, eps=eps, debug=debug)


def normalize_run_ea(run, R_invsqrt, mean, std, eps=1e-6):
    """
    Pipeline completo para un EEGSubset de test: EA → normalización → reshape Keras.
    Reemplaza normalize_run() cuando se usa EA.

    Parámetros
    ----------
    run        : EEGSubset con X shape (N, C, T)
    R_invsqrt  : Calculado con compute_ea_matrix_from_subset(fine_subset)
    mean, std  : Del fine-tuning (como siempre)

    Retorna
    -------
    X_keras : np.ndarray, shape (N, C, T, 1)
    """
    from backend_DL_v2 import normalizar_por_canal, ajustar_keras

    X_aligned = apply_ea_transform(run.X, R_invsqrt)   # EA
    X_norm    = normalizar_por_canal(X_aligned, mean, std, eps)  # normalización
    X_keras   = ajustar_keras(X_norm)                   # reshape para Keras

    return X_keras


# ============================================================
# CAMBIOS EN eeg_fine() PARA INTEGRAR EA
# ============================================================
#
# En el Paso 2 de eeg_fine, ANTES de normalizar_split o normalizar_por_canal,
# añadir:
#
#   # --- EA: alinear datos del sujeto ---
#   if use_ea:
#       R_invsqrt, R_diag = compute_ea_matrix(split_fine.X.train, debug=debug)
#       split_fine_X_train_ea = apply_ea(split_fine.X.train, R_invsqrt)
#       split_fine_X_val_ea   = apply_ea(split_fine.X.val,   R_invsqrt)
#       # Reemplazar en split_fine
#       split_fine = DataSplit(
#           split_fine_X_train_ea, split_fine_X_val_ea, split_fine.X.test,
#           split_fine.y.train, split_fine.y.val, split_fine.y.test,
#           ... (resto igual)
#       )
#
# Y en el test_run dentro de eeg_fine:
#
#   X_test = normalize_run_ea(test_run, R_invsqrt, mean, std)
#
# Retornar también R_invsqrt para usarlo en evaluación posterior:
#   return model_ft, history, mean, std, keras_split_fine, info, R_invsqrt
#
# ============================================================


# ============================================================
# DIAGNÓSTICO: ver si EA ayuda a un sujeto específico
# ============================================================

def diagnose_ea(fine_subset, test_subset, debug=True):
    """
    Muestra el efecto de EA en la distribución de un sujeto.
    Útil para confirmar que el sujeto 98 / 67 tiene covarianza alejada
    de la media del grupo antes de alinear.

    Uso:
        diagnose_ea(data_fine_folds[fold_98], data_test_folds[fold_98])
    """
    R_invsqrt, R = compute_ea_matrix(fine_subset.X, debug=False)

    print("=" * 50)
    print(f"Sujeto: {np.unique(fine_subset.sub)}")
    print(f"Runs fine-tuning: {np.unique(fine_subset.run)}")
    print(f"\nCovarianza media (diagonal R):")
    print(f"  {np.diag(R).round(3)}")
    print(f"  Traza: {np.trace(R):.3f}  (ideal = n_canales = {fine_subset.X.shape[1]})")
    print(f"  Condición de R: {np.linalg.cond(R):.1f}  (>100 = mal condicionada)")

    # Tras alinear
    X_aligned = apply_ea_transform(fine_subset.X, R_invsqrt)
    covs_post = np.einsum('nct,ndt->ncd', X_aligned, X_aligned) / X_aligned.shape[2]
    R_post = covs_post.mean(axis=0)

    print(f"\nDespués de EA:")
    print(f"  Diagonal R_post: {np.diag(R_post).round(3)}")
    print(f"  Traza: {np.trace(R_post):.3f}")

    # Test subset
    X_test_aligned = apply_ea_transform(test_subset.X, R_invsqrt)
    covs_test = np.einsum('nct,ndt->ncd', X_test_aligned, X_test_aligned) / X_test_aligned.shape[2]
    R_test = covs_test.mean(axis=0)
    print(f"\nTest subset (post-EA):")
    print(f"  Traza: {np.trace(R_test):.3f}  (debe ser ≈ {fine_subset.X.shape[1]})")
    print("=" * 50)

    return R_invsqrt


# ============================================================
# SURFACE LAPLACIAN FILTER (Small Laplacian)
# Nunez et al., 1997 — "EEG coherency I: statistics, reference electrode,
# volume conduction, Laplacians, cortical imaging, and interpretation
# at multiple scales"
#
# Suprime la conducción de volumen restando a cada canal la media
# ponderada de sus vecinos inmediatos del montaje 10-10.
# En runtime es una sola multiplicación matricial L @ X:
#   latencia < 0.1 ms para cualquier número de canales razonable.
#
# FLUJO DE USO:
#   1. Una vez, al iniciar:  L = compute_laplacian_matrix(channel_names)
#   2. Offline/online:       X_lap = apply_laplacian(X, L)
#   3. En eeg_train/eeg_fine: usar apply_lsf=True + lsf_channels=channel_names
# ============================================================

# Vecindades del montaje 10-10 para canales comunes de MI-EEG.
# Clave: nombre normalizado (sin puntos).
# Valor: lista de vecinos inmediatos en el eje lateral de la franja C.

_LSF_NEIGHBORS_1010 = {
    # Franja C — eje lateral
    "C5":  ["C3"],
    "C3":  ["C5", "C1"],
    "C1":  ["C3", "Cz"],
    "Cz":  ["C1", "C2"],
    "C2":  ["Cz", "C4"],
    "C4":  ["C2", "C6"],
    "C6":  ["C4"],
    # Franja FC
    "FC5": ["FC3", "C5"],
    "FC3": ["FC5", "FC1", "C3"],
    "FC1": ["FC3", "FCz", "C1"],
    "FCz": ["FC1", "FC2", "Cz"],
    "FC2": ["FCz", "FC4", "C2"],
    "FC4": ["FC2", "FC6", "C4"],
    "FC6": ["FC4", "C6"],
    # Franja CP
    "CP5": ["CP3", "C5"],
    "CP3": ["CP5", "CP1", "C3"],
    "CP1": ["CP3", "CPz", "C1"],
    "CPz": ["CP1", "CP2", "Cz"],
    "CP2": ["CPz", "CP4", "C2"],
    "CP4": ["CP2", "CP6", "C4"],
    "CP6": ["CP4", "C6"],
}


def _normalize_ch(name):
    """Normaliza nombre de canal: 'C3..' → 'C3', 'Cz..' → 'Cz', 'Cp3.' → 'CP3', 'Cpz.' → 'CPz'."""
    name = name.replace(".", "").replace(" ", "").strip()
    name = name.upper().replace("Z", "z")
    return name


def compute_laplacian_matrix(channel_names, debug=False):
    """
    Construye la matriz L del Small Laplacian para un subconjunto
    de canales del montaje 10-10.

    Parámetros
    ----------
    channel_names : list[str]
        Nombres de canales en el mismo orden que el eje 1 de X.
        Acepta formato BCI2000 con puntos extra (p.ej. 'C3..').

    debug : bool
        Si True, imprime vecinos encontrados y la matriz L.

    Retorna
    -------
    L : np.ndarray, shape (C, C)  dtype float32
        Matriz Laplaciana precalculada.
        Uso offline (batch): np.einsum('cd,ndt->nct', L, X)
        Uso online (ventana): L @ X

    Notas
    -----
    - Canal con un solo vecino en el subconjunto: peso -1 (diferencia
      simple), comportamiento correcto para extremos de la franja.
    - Canal sin ningún vecino en el subconjunto: se deja como identidad
      y se emite un warning — no se filtra, pero tampoco rompe nada.
    - La matriz es estática; calcúlala UNA VEZ y reutilízala en todos
      los folds, sujetos y ventanas online.
    """
    norm_names = [_normalize_ch(ch) for ch in channel_names]
    C = len(norm_names)
    ch_idx = {name: i for i, name in enumerate(norm_names)}

    L = np.zeros((C, C), dtype=np.float32)

    for i, ch in enumerate(norm_names):
        all_neighbors   = _LSF_NEIGHBORS_1010.get(ch, [])
        present_neighbors = [n for n in all_neighbors if n in ch_idx]
        n_nb = len(present_neighbors)

        if n_nb == 0:
            warnings.warn(
                f"[LSF] Canal '{ch}' no tiene vecinos conocidos en el "
                "subconjunto. Se deja sin filtrar (fila identidad). "
                "Agrega sus vecinos a _LSF_NEIGHBORS_1010 si es necesario."
            )
            L[i, i] = 1.0
        else:
            L[i, i] = 1.0
            w = 1.0 / n_nb
            for nb in present_neighbors:
                L[i, ch_idx[nb]] = -w

        if debug:
            w_show = -1.0 / n_nb if n_nb > 0 else 0.0
            print(f"  {ch:>6}: vecinos_activos={present_neighbors}  peso={w_show:.3f}")

    if debug:
        print(f"\n[LSF] Matriz L ({C}×{C}):")
        print(np.round(L, 3))

    return L


def apply_laplacian(X, L):
    """
    Aplica el filtro Laplaciano a una ventana o a un batch.

    Parámetros
    ----------
    X : np.ndarray
        (N, C, T)  — batch  (entrenamiento / evaluación)
        (C, T)     — ventana única (online)

    L : np.ndarray, shape (C, C)
        Salida de compute_laplacian_matrix().

    Retorna
    -------
    X_lap : np.ndarray, misma shape que X
    """
    if X.ndim == 2:          # online: (C, T)
        return L @ X
    return np.einsum('cd,ndt->nct', L, X)   # batch: (N, C, T)