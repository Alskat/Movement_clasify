#Generador de clases para procesamiento

#Primero, importamos todas las librerías que ya teníamos antes 
#Clásicas
from pdb import run

import numpy as np
import pandas as pd

#Para directorios 
import sys
import os




#Librerías de sklearn y tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
    precision_recall_fscore_support
)

#0 Clase para almacenar los datos EEG de manera más fácil
class EEGSubset:
    def __init__(self, data_dict):
        self.X = data_dict["X"]
        self.y = data_dict["y"]
        self.sub = data_dict["sub"]
        self.trial = data_dict["trial"]
        self.run = data_dict["run"]

#1 Vamos a crear la clase donde procesaremos todos los datos con el pick, el dataset
class Selection:

    """La función de este nuevo objeto va a ser obtener los datos, pickear las clases que queremos, undersamplear
    la clase 0 (rest) si es necesario, y luego fusionar las clases motoras e imaginarias, para finalmente retornar
    los datos ya procesados, listos para el entrenamiento."""
    def __init__(self, pick=None, undersample_rest=True, fusionar=True, random_state=42, Debug=False):
        
        self.pick = pick
        self.undersample_rest = undersample_rest #Esto nos ayuda a undersamplear la clase 0 a las demás clases
        self.fusionar = fusionar #Fusionar las clases mecánicas con las imaginarias
        self.random_state = random_state
        self.Debug = Debug

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
        
    def load(self, path):   

        #Creamos nuestro self.data acá

        data = np.load(path, allow_pickle=True)

        self.X = data["X"]
        self.y = data["y"]
        self.sub = data["sub"]
        self.run = data["run"]
        self.trial = data["trial"]

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

        return self
    
    def pick_fine(self, subject_id):

        mask_model = self.sub != subject_id
        mask_fine  = self.sub == subject_id


        data_model = {
            "X": self.X[mask_model],
            "y": self.y[mask_model],
            "sub": self.sub[mask_model],
            "trial": self.trial[mask_model],
            "run": self.run[mask_model]
        }

        data_fine = {
            "X": self.X[mask_fine],
            "y": self.y[mask_fine],
            "sub": self.sub[mask_fine],
            "trial": self.trial[mask_fine],
            "run": self.run[mask_fine]
        }

        if self.Debug:
            print(f"✔ Sujetos entrenamiento: {np.unique(data_model['sub'])}")
            print(f"✔ Sujeto fine-tune: {subject_id}")
            print(f"Shape modelo: {data_model['X'].shape}")
            print(f"Shape fine: {data_fine['X'].shape}")

        self.data_model = EEGSubset(data_model)
        self.data_fine = EEGSubset(data_fine)

        self.fine_subject = subject_id

        return self.data_model, self.data_fine
    
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
    
        # Paso 1: Pickeo 
        X = self.X
        y = self.y
        sub = self.sub
        trial = self.trial
        run = self.run

        if undersample_rest is not None:
            self.undersample_rest = undersample_rest
            if self.Debug:
                print(f"No se va a aplicar undersample!")

    

        # --- 1) Elegir clases a conservar ---
        if self.pick is not None:
            if self.Debug:
                print(f"✔ Aplicando pick: {self.pick}. . . ")
            pick = list(self.pick)
            mask = np.isin(y, pick)
            X = X[mask]
            y = y[mask]
            sub = sub[mask]
            trial = trial[mask]
            run = run[mask]

            if X.shape[0] == 0:
                raise ValueError(f"No hay muestras para las clases {pick}.")
        else:
            # si no se especifica pick, usamos todas las clases presentes
            pick = sorted(np.unique(y).tolist())
            if self.Debug:
                print("✔ No se especificó pick, usando todas las clases presentes:", pick, ". . .")

        # --- 2) Submuestreo adicional por cantidad n (opcional) ---
        total = len(y)
        if n is not None and n < total:
            if self.Debug:
                print(f"✔ Aplicando submuestreo adicional: limitando a los primeros {n} de {total} datos. . . ")
            X = X[:n]
            y = y[:n]
            sub = sub[:n]
            trial = trial[:n]
            run = run[:n]
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

            # aplicar undersampling si corresponde
            if self.undersample_rest:
                self._undersample_rest(X, y_final, sub, trial, run)
            else:
                self._build(X, y_final, sub, trial, run)

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
                        

        # --- 4) Undersamplear la clase rest de ser necesario ---
        if self.undersample_rest and 0 in pick:
            if self.Debug:
                print(f"✔ Datos antes del undersampling en 0: {np.sum(self.y==0)}")
            self._undersample_rest(X, y_final, sub, trial, run) #Si queremos undersamplear la clase 0, lo hacemos después de fusionar, para que se ajuste a las clases fusionadas
            if self.Debug:
                print(f"Datos después de undersampling - 0: {np.sum(self.y==0)}")
        else: 
            self._build(X, y_final, sub, trial, run) #Ahora construimos los atributos principaples

        


    
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
    

    def _undersample_rest(self, X, y, sub, trial, run): 
    
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
                rng = np.random.default_rng(self.random_state)
                idx_rest_sel = rng.choice(idx_rest, size=max_other, replace=False)

                idx_keep = np.concatenate([idx_rest_sel, idx_non_rest])
                rng.shuffle(idx_keep)

                X = X[idx_keep]
                y = y[idx_keep]
                sub = sub[idx_keep]
                trial = trial[idx_keep]
                run = run[idx_keep]

                print(f"Undersampling: clase 0 recortada de {len(idx_rest)} a {max_other} muestras.")

        self._build(X, y, sub, trial, run)


    def _build(self, X, y, sub, trial, run):
        self.X = X
        self.y = y
        self.sub = sub  
        self.trial = trial
        self.run = run
        

        
        
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
                run_train=None, run_val=None, run_test=None):
        self.X = XYGroup(X_train, X_val, X_test)
        self.y = XYGroup(y_train, y_val, y_test)
        self.sub = XYGroup(sub_train, sub_val, sub_test)
        self.trial = XYGroup(trial_train, trial_val, trial_test)
        self.run = XYGroup(run_train, run_val, run_test)


#1) Splitear los datos 

def split_eeg(X, y, sub, trial, run, test_size=0.2, val_size=0.1, mode = None, debug=False): #chatgpt me puso esta función más bonita 

    assert len(X) == len(y) == len(sub) == len(trial) == len(run), "X, y, sub, trial y run deben tener la misma longitud!"

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
        X, y, sub, trial, run, group,
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

def _split_by_group(X, y, sub, trial, run, group, test_size=0.1, val_size=0.1, random_state=42, debug=False): #Para qué nos vamos a mentir, yo hice el código pero Claude lo optimizó

    # Validación
    assert len(X) == len(y) == len(group), "X, y y group deben tener mismo largo"
    if group is None:
        raise ValueError("group no puede ser None en _split_by_group (No debería salir nunca este eror igual :P)")

    # 1. Grupos únicos
    unique_groups = np.unique(group)
    

    # 2. Split train vs temp
    train_groups, temp_groups = train_test_split(
        unique_groups,
        test_size=test_size + val_size,
        random_state=random_state
    )

    # 3. Split temp → val + test
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=test_size / (test_size + val_size),
        random_state=random_state
    )

 

    # 4. Máscaras
    mask_train = np.isin(group, train_groups)
    mask_val   = np.isin(group, val_groups)
    mask_test  = np.isin(group, test_groups)
    

    # 5. Aplicar
    X_train, y_train, g_train = X[mask_train], y[mask_train], group[mask_train]
    X_val, y_val, g_val       = X[mask_val], y[mask_val], group[mask_val]
    X_test, y_test, g_test    = X[mask_test], y[mask_test], group[mask_test]

    sub_train, trial_train, run_train = sub[mask_train], trial[mask_train], run[mask_train]
    sub_val, trial_val, run_val       = sub[mask_val], trial[mask_val], run[mask_val]
    sub_test, trial_test, run_test    = sub[mask_test], trial[mask_test], run[mask_test]

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
        run_train, run_val, run_test
    )


    
    
#2) Normalización


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
        split.run.train, split.run.val, split.run.test
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
        split.y.train,
        split.y.val,
        split.y.test,
        split.sub.train,
        split.sub.val,
        split.sub.test,
        split.trial.train,
        split.trial.val,
        split.trial.test,
        split.run.train,
        split.run.val,
        split.run.test
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

def eeg_train(data_obj, mode="subject", test_size=0.1, classes=None, epochs=20, 
              debug=False, use_class_weight=True, sfreq=160):
    
    #0) Definir variables
    X, y, sub, trial, run = data_obj.X, data_obj.y, data_obj.sub, data_obj.trial, data_obj.run

    if classes is None:
        classes = len(np.unique(y))

    #1 Split 
    split, info = split_eeg(X, y, sub, trial, run, mode = mode, test_size=test_size, debug=debug)

    #2 Normalización: importante sólo se normaliza X

    norm_split, channel_mean, channel_std = normalizar_split(split)

    #3 Ajustar forma para Keras (Sólo X también)

    keras_split = prepare_keras_split(norm_split)

    if debug: 
        print("Antes de normalizar:", split.X.train.shape, "Después de normalizar:", norm_split.X.train.shape, "Después de ajustar para Keras:", keras_split.X.train.shape)

    #4 Construir modelo EEGnet
    chans = keras_split.X.train.shape[1]
    signal_len = keras_split.X.train.shape[2]

    kern_length = int(sfreq // 2)   
    modelo = build_eegnet(classes, chans, signal_len, kern_length=kern_length)

    #5) Compilar el modelo
    modelo.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(filepath="EEGNet_best.keras",
                        monitor='val_accuracy', save_best_only=True,
                        save_weights_only=False, mode='max'),
        
    ]

    #6 Entrenar el modelo 

    if debug:
        verbose = 1
    else:
        verbose = 0
    class_weight = None
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        classes_present = np.unique(keras_split.y.train)
        weights = compute_class_weight('balanced',
                                       classes=classes_present,
                                       y=keras_split.y.train)
        class_weight = dict(zip(classes_present, weights))
        
    """if debug:
        print("Nombres de los hiperparámetros:")
        print(f"Learning rate: {K.get_value(modelo.optimizer.lr)}")
        print(f"Kernel length: {kern_length}")
        print(f"mean: {channel_mean} y std: {channel_std}")
        print(f"Class weights: {class_weight}")
        print(f"signal_len: {signal_len}")
        print(f"chans: {chans}")
        print(f"")"""

    history = modelo.fit(
        keras_split.X.train, keras_split.y.train,
        epochs=epochs,
        batch_size=32,          # ✅ bajar de 64 a 32
        validation_data=(keras_split.X.val, keras_split.y.val),
        callbacks=callbacks,
        class_weight=class_weight,  # ✅ añadir
        verbose=verbose
    )

    test_loss, test_acc = modelo.evaluate(keras_split.X.test, keras_split.y.test, verbose=0)
    print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    

    return modelo, history, channel_mean, channel_std, keras_split, info

def eeg_fine(base, dataset_fine, mean = None, std = None, epochs = 20, debug =False, test_size=0.2,
              unfreeze_last_n=16, lr=5e-4, use_class_weight=True, batch_size=32):

    #Paso 0: Definir variables
    X, y, sub, trial, run = dataset_fine.X, dataset_fine.y, dataset_fine.sub, dataset_fine.trial, dataset_fine.run
    #Paso 1: split 
    if(len(np.unique(sub)) != 1):
        raise ValueError("El finetunning debe ser de un sólo sujeto WEON")
   

    split_fine, _ = split_eeg(X, y, sub, trial, run, mode = "trial", test_size=test_size, debug=debug)

    if debug: 
        print("Tamaño del train:", split_fine.X.train.shape, "Tamaño del val:", split_fine.X.val.shape, "Tamaño del test:", split_fine.X.test.shape)

    #Paso 2: Normalización con media y desviación dada (si no se da, se calcula con el mismo método que antes)

    #if mean is None or std is None:
    mean, std = calc_stats(split_fine.X.train)

    X_train_norm = normalizar_por_canal(split_fine.X.train, mean, std)
    X_val_norm = normalizar_por_canal(split_fine.X.val, mean, std)
    X_test_norm = normalizar_por_canal(split_fine.X.test, mean, std)

    #Paso 3: Ajustar forma para Keras

    X_train_keras = ajustar_keras(X_train_norm)
    X_val_keras = ajustar_keras(X_val_norm)
    X_test_keras = ajustar_keras(X_test_norm)
    if debug:
        print("Antes de normalizar:", split_fine.X.train.shape, "Después de ajustar para Keras:", X_train_keras.shape)   
    
    #Paso 4: Clonar modelo para finetunning (para no afectar el modelo original)
    model_ft = clone_model(base)
    model_ft.set_weights(base.get_weights())

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
        classes_present = np.unique(split_fine.y.train)
        weights = compute_class_weight('balanced', 
                                        classes=classes_present, 
                                        y=split_fine.y.train)
        class_weight = dict(zip(classes_present, weights))
        if debug:
            print("Class weights:", class_weight)

    if debug:
        verbose = 1
    else:
        verbose = 0

    #Paso 6: callbacks específicos para fine-tuning
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(
            filepath="EEGNet_finetuned_subject.keras",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
    ]



    #Paso 7: entrenamiento de fine-tuning
    history = model_ft.fit(
        X_train_keras, split_fine.y.train,
        validation_data=(X_val_keras, split_fine.y.val),
        callbacks=callbacks,
        epochs=epochs,
        batch_size=batch_size,          
        class_weight=class_weight,
        verbose=verbose
    )

    #Paso 8: evaluación en test del sujeto
    test_loss, test_acc = model_ft.evaluate(X_test_keras, split_fine.y.test, verbose=0)
    print(f"[Fine-tune] Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    keras_split_fine = DataSplit(
        X_train_keras, X_val_keras, X_test_keras,
        split_fine.y.train, split_fine.y.val, split_fine.y.test,
        split_fine.sub.train, split_fine.sub.val, split_fine.sub.test,
        split_fine.trial.train, split_fine.trial.val, split_fine.trial.test,
        split_fine.run.train, split_fine.run.val, split_fine.run.test
    )

    return model_ft, history, mean, std, keras_split_fine

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

def evaluate(modelo, keras_split, label_map=None, title="EEGNet"):
    """
    Evaluar directamente un modelo de manera más rápida
    """

    # 1. Datos
    X_test = keras_split.X.test
    y_test = keras_split.y.test

    # 2. Predicción
    y_prob = modelo.predict(X_test)

    # 3. Nombres de clases
    if label_map is not None:
        class_names = [label_map[i] for i in sorted(label_map.keys())]
    else:
        n_classes = y_prob.shape[1]
        class_names = [str(i) for i in range(n_classes)]

    # 4. Evaluación completa
    resultados = evaluar_modelo_multiclase(
        y_true=y_test,
        y_prob=y_prob,
        class_names=class_names,
        title_prefix=title
    )

    return resultados
def evaluar_modelo_multiclase(y_true, y_prob, class_names=None, title_prefix="EEGNet"):
    """
    y_true : array (N,) con etiquetas enteras 0..C-1
    y_prob : array (N, C) con probabilidades (salida de model.predict)
    class_names : lista opcional de nombres para las clases en orden 0..C-1
                  (si es None, usa "0","1",...)
    title_prefix : texto para los títulos de los plots
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    #1) Obtener predicciones discretas
    y_pred = np.argmax(y_prob, axis=1)

    #2) Definir labels automáticamente
    n_classes = y_prob.shape[1]
    labels = np.arange(n_classes)

    if class_names is None:
        class_names = [str(i) for i in labels]

    print(f"\n=== Evaluación {title_prefix} ===")
    print("Clases (índice -> nombre):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    #3) Matriz de confusión (no normalizada)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nMatriz de confusión (conteos):")
    print(cm)

    #4) Reporte de clasificación
    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0
        )
    )

    # 5) Precision, recall, F1 por clase
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        zero_division=0
    )

    # ---- Plot F1 por clase ----
    plt.figure(figsize=(6, 4))
    plt.bar(labels, f1)
    plt.xticks(labels, class_names, rotation=0)
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.title(f"F1 por clase ({title_prefix})")
    plt.tight_layout()
    plt.show()

    # ---- Matriz de confusión normalizada por fila ----
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

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

    
    return {
        "y_pred": y_pred,
        "cm": cm,
        "cm_norm": cm_norm,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

def plot_history(history, metrics=("accuracy",), title_prefix="EEGNet"): #Lo más sencillo del mundo, la saqué de chatgpt
    """
    history : objeto History de Keras (lo que devuelve model.fit)
    metrics : tupla/lista de métricas a graficar además de la loss
              (por defecto solo 'accuracy')
    title_prefix : texto para los títulos de las figuras
    """
    hist_dict = history.history

    # --------- 1) Pérdida (loss) ---------
    plt.figure(figsize=(6, 4))
    plt.plot(hist_dict["loss"], label="train loss")
    if "val_loss" in hist_dict:
        plt.plot(hist_dict["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} – Curva de pérdida")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------- 2) Métricas (accuracy, etc.) ---------
    for m in metrics:
        # Soportar nombres antiguos tipo 'acc'/'val_acc'
        train_key = m if m in hist_dict else f"{m}"
        val_key   = f"val_{m}" if f"val_{m}" in hist_dict else None

        if train_key not in hist_dict:
            print(f"[plot_history] Métrica '{m}' no encontrada en history, me la salto.")
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(hist_dict[train_key], label=f"train {m}")
        if val_key and val_key in hist_dict:
            plt.plot(hist_dict[val_key], label=f"val {m}")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title(f"{title_prefix} – {m} por epoch")
        plt.legend()
        plt.tight_layout()
        plt.show()

def save_model(model, mean, std, path = None, name_model = "model_EEGnet.keras", name_params = "params_EEGnet.npz"):
    if path is None:
        print("no se ha especificado el path, se guardará en el directorio actual")
        path = os.getcwd()
    if not os.path.exists(os.path.dirname(path)):
        print(f"creando el directorio {os.path.dirname(path)}")
        os.makedirs(os.path.dirname(path))
    model.save(os.path.join(path, name_model))
    np.savez(os.path.join(path, name_params), mean=mean, std=std)
    
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
    

