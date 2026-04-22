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

#0 

#1 Vamos a crear la clase donde procesaremos todos los datos con el pick, el dataset
class Selection:

    """La función de este nuevo objeto va a ser obtener los datos, pickear las clases que queremos, undersamplear
    la clase 0 (rest) si es necesario, y luego fusionar las clases motoras e imaginarias, para finalmente retornar
    los datos ya procesados, listos para el entrenamiento."""
    def __init__(self, pick=None, undersample_rest=True, fusionar=True, random_state=42):
        
        self.pick = pick
        self.undersample_rest = undersample_rest #Esto nos ayuda a undersamplear la clase 0 a las demás clases
        self.fusionar = fusionar #Fusionar las clases mecánicas con las imaginarias
        self.random_state = random_state

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

        #Ahora ejercemos nuestro pipeline, el cual era: pick, fusionar, luego después se elegirán los sujetos de entrenamiento
        
    def load(self, path):   

        #Creamos nuestro self.data acá

        data = np.load(path, allow_pickle=True)

        self.X = data["X"]
        self.y = data["y"]
        self.sub = data["sub"]
        self.run = data["run"]
        self.trial = data["trial"]

        self.class_names = list(data["class_names"])
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

        # Dimensiones
        print("\n📐 Dimensiones:")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
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
        print(f"Número de clases: {len(self.class_names)}")
        print(f"Lista: {self.class_names}")

        # Distribución de clases
        print("\n📊 Distribución de clases:")
        unique, counts = np.unique(self.y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"{self.class_names[u]:10s}: {c}")

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
            
    def pipeline(self, n=None):
        #El pipeline principal consiste en aplicar un pick y luego una fusión, cosas que ya tenemos en otras clases 
        #n sólo es para limitar la cantidad de datos totales, no recuerdo porque lo quise añadir pero ahí está
    
        # Paso 1: Pickeo 
        X = self.X
        y = self.y
        sub = self.sub
        trial = self.trial
        run = self.run

        if len(sub) != len(y):
            #Cortamos para que sean iguales
            min_len = min(len(sub), len(y))
            X = X[:min_len]
            y = y[:min_len]
            sub = sub[:min_len]
            trial = trial[:min_len]
            run = run[:min_len]

        # --- 1) Elegir clases a conservar ---
        if self.pick is not None:
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

        # --- 2) Undersampling de la clase 0 (rest), si procede ---
        """if self.undersample_rest and 0 in pick:
            rest_label = 0
            idx_rest = np.where(y == rest_label)[0]
            idx_non_rest = np.where(y != rest_label)[0]

            # conteos de las clases distintas de 0 (en el espacio original)
            counts_others = []
            for cls in pick:
                if cls == rest_label:
                    continue
                counts_others.append(np.sum(y == cls))

            if counts_others:  # por si solo hay la clase 0
                max_other = max(counts_others)

                if len(idx_rest) > max_other:
                    rng = np.random.default_rng(self.random_state)
                    idx_rest_sel = rng.choice(idx_rest, size=max_other, replace=False)

                    idx_keep = np.concatenate([idx_rest_sel, idx_non_rest])
                    rng.shuffle(idx_keep)

                    X = X[idx_keep]
                    y = y[idx_keep]
                    sub = sub[idx_keep]

                    print(f"Undersampling: clase 0 recortada de {len(idx_rest)} a {max_other} muestras.")"""
            # si counts_others está vacío, significa que solo hay clase 0, no hacemos nada

        # --- 3) Submuestreo adicional por cantidad n (opcional) ---
        total = len(y)
        if n is not None and n < total:
            X = X[:n]
            y = y[:n]
            sub = sub[:n]
            trial = trial[:n]
            run = run[:n]
            print(f"Seleccionando los primeros {n} de {total} datos tras el filtrado.")

        # --- 4) Remapeo de etiquetas a 0..C-1 ---
        #Acá reordenamos todas las etiquetas de y
        unique_sorted = sorted(pick)
        mapa = {old: new for new, old in enumerate(unique_sorted)}
        y_new = np.vectorize(mapa.get)(y)

        #return X, y_new, sub
        #------------------------------------------------------

        #Paso 2: Fusión de clases motoras e imaginarias 
        """Recordemos las clases:
    
        "rest" -> 0
        "right_i" -> 1
        "left_i"->2
        "hands_i" -> 3
        "feet_i" -> 4
        "right_m" -> 5
        "left_m" -> 6
        "hands_m" -> 7
        "feet_m" -> 8"""

        if self.fusionar: 
            y_lr = y_new.copy()

            if len(np.unique(y_new)) == 9: #Son 5 clases en total, fusionamos motoras e imaginarias
                y_lr[y_new==5] = 1
                y_lr[y_new==6] = 2
                y_lr[y_new==7] = 3
                y_lr[y_new==8] = 4

            elif len(np.unique(y_new)) == 5: #3 clases en total (manos y pies o derecha e izquierda)
                y_lr[y_new==3] = 1
                y_lr[y_new==4] = 2

        else: 
            y_lr = y_new



        if self.undersample_rest and 0 in pick:
            self._undersample_rest(X, y_lr, sub, trial, run) #Si queremos undersamplear la clase 0, lo hacemos después de fusionar, para que se ajuste a las clases fusionadas
        else: 
            self._build(X, y_lr, sub, trial, run) #Ahora construimos los atributos principaples

    def _undersample_rest(self, X, y, sub, trial, run): 
        
        pick = list(self.pick)
        rest_label = 0
        idx_rest = np.where(y == rest_label)[0]
        idx_non_rest = np.where(y != rest_label)[0]

        # conteos de las clases distintas de 0 (en el espacio original)
        counts_others = []
        for cls in pick:
            if cls == rest_label:
                continue
            counts_others.append(np.sum(y == cls))

        if counts_others:  # por si solo hay la clase 0
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
            # si counts_others está vacío, significa que solo hay clase 0, no hacemos nada
        self._build(X, y, sub, trial, run) #Ahora construimos los atributos principaples

    def _build(self, X, y, sub, trial, run):
        self.X = X
        self.y = y
        self.sub = sub  
        self.trial = trial
        self.run = run
        

        #Esto en teoría es sencillo, pero ahora, quiero hacer una tabla de clases únicamente porque sí, para esto necesitamos saber pick de nuevo 
        if self.fusionar is False: 
            self.clases = "Chingue a su madre" #Vamos a fusionar en todo momento, así que no nos importa esto
            
            
        else:
            """Ahora, tengamos en cuenta lo siguiente, si: 
            pick = [0,3,4,7,8] Rest, Manos y pies 
            pick = [0,1,2,5,6] Rest, Derecha e Izquierda
            pick = [0,1,2,3,4,5,6,7,8] o None, Todas las clases 
            """
            unique_labels = sorted(np.unique(y).tolist())

            if len(unique_labels) == 3:
                # distinguir si son izquierda/derecha o manos/pies
                pick_set = set(self.pick) if self.pick is not None else {0,1,2,3,4,5,6,7,8} #Ver la lista del pick namás

                if pick_set == {0, 1, 2, 5, 6}:
                    nombres = ["Rest", "Derecha", "Izquierda"]

                elif pick_set == {0, 3, 4, 7, 8}:
                    nombres = ["Rest", "Manos", "Pies"]

                else:
                    nombres = ["Rest", "Clase 1", "Clase 2"]

            elif len(unique_labels) == 5:
                nombres = ["Rest", "Derecha", "Izquierda", "Manos", "Pies"]
            else: 
                nombres = [f"Clase {lbl}" for lbl in unique_labels]

            self.clases = pd.DataFrame({
            "codigo": unique_labels,
            "nombre": nombres
            }, index=unique_labels)
        
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
    if mode is None and len(unique_subj) >= 10:
        mode = "subject"
        group = sub
    elif mode is None and len(unique_subj) < 10:
        mode = "trial"
        group = trial

    elif mode == "subject":
        group = sub
    elif mode == "trial":
        group = trial
    elif mode == "run":
        group = run
    else: 
        raise ValueError(f"Modo de split desconocido: {mode}. Opciones válidas: 'subject', 'trial', 'run' o None (auto).")



    #Split universal
    split = _split_by_group(
        X, y, group,
        test_size=test_size,
        val_size=val_size
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

    return split, split_info

def _split_by_group(X, y, group, test_size=0.2, val_size=0.1, random_state=42, debug=False): #Para qué nos vamos a mentir, yo hice el código pero Claude lo optimizó

    # Validación
    assert len(X) == len(y) == len(group), "X, y y group deben tener mismo largo"

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

    if debug:
        print("Split por grupo:")
        print(f"  Train: {len(X_train)} muestras, grupos: {np.unique(g_train)}")
        print(f"  Val:   {len(X_val)} muestras, grupos: {np.unique(g_val)}")
        print(f"  Test:  {len(X_test)} muestras, grupos: {np.unique(g_test)}")

    return DataSplit(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        g_train, g_val, g_test
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
        split.y.train, split.y.val, split.y.test
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
        split.y.test
    )    

#4 Construir modelo EEGnet
def build_eegnet(classes, chans, signal_len,
                 dropout_rate=0.5, kern_length=64,
                 F1=16, D=2, norm_rate=0.25,
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

def eeg_train(X, y, sub, trial, run, test_size=0.2, classes=None, epochs=20, debug=False):
    if classes is None:
        classes = len(np.unique(y))

    #1 Split 
    split, info = split_eeg(X, y, sub, trial, run, test_size=test_size, debug=debug)

    #2 Normalización: importante sólo se normaliza X

    norm_split, channel_mean, channel_std = normalizar_split(split)

    #3 Ajustar forma para Keras (Sólo X también)

    keras_split = prepare_keras_split(norm_split)

    if debug: 
        print("Antes de normalizar:", split.X.train.shape, "Después de normalizar:", norm_split.X.train.shape, "Después de ajustar para Keras:", keras_split.X.train.shape)

    #4 Construir modelo EEGnet
    chans = keras_split.X.train.shape[1]
    signal_len = keras_split.X.train.shape[2]

    modelo = build_eegnet(classes, chans, signal_len)

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
        verbose = 2
    else:
        verbose = 1
    #Entrenamiento 
    history = modelo.fit(
        keras_split.X.train, keras_split.y.train,
        epochs=epochs,
        batch_size=64,   
        validation_data=(keras_split.X.val, keras_split.y.val),
        callbacks=callbacks,
        verbose=verbose
    )

    test_loss, test_acc = modelo.evaluate(keras_split.X.test, keras_split.y.test, verbose=0)
    print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    

    return modelo, history, channel_mean, channel_std, keras_split, info

def eeg_fine(base, X, y, sub, mean = None, std = None, epochs = 20, debug =False, test_size=0.2):

    #Paso 1: split 
    if(len(np.unique(sub)) != 1):
        raise ValueError("El finetunning debe ser de un sólo sujeto WEON")
   

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y 
    )

    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,     
    random_state=152,
    stratify=y_temp
    )
    if debug: 
        print("Tamaño del train:", X_train.shape, "Tamaño del val:", X_val.shape, "Tamaño del test:", X_test.shape)

    #Paso 2: Normalización con media y desviación dada (si no se da, se calcula con el mismo método que antes)

    if mean is None or std is None:
        mean, std = calc_stats(X_train)

    X_train_norm = normalizar_por_canal(X_train, mean, std)
    X_val_norm = normalizar_por_canal(X_val, mean, std)
    X_test_norm = normalizar_por_canal(X_test, mean, std)

    #Paso 3: Ajustar forma para Keras

    X_train_keras = ajustar_keras(X_train_norm)
    X_val_keras = ajustar_keras(X_val_norm)
    X_test_keras = ajustar_keras(X_test_norm)
    if debug:
        print("Antes de normalizar:", X_train.shape, "Después de ajustar para Keras:", X_train_keras.shape)   
    
    #Paso 4: Clonar modelo para finetunning (para no afectar el modelo original)
    model_ft = clone_model(base)
    model_ft.set_weights(base.get_weights())

    for layer in model_ft.layers:
        layer.trainable = True


    #Paso 5: recompilar con LR más bajo para fine-tuning

    model_ft.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=5e-5),
        metrics=['accuracy']
    )

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
        X_train_keras, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val_keras, y_val),
        callbacks=callbacks
    )

    #Paso 8: evaluación en test del sujeto
    test_loss, test_acc = model_ft.evaluate(X_test_keras, y_test, verbose=0)
    print(f"[Fine-tune] Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Opcional: devolver también los splits por si quieres calcular F1 afuera
    X_list = [X_train_keras, X_val_keras, X_test_keras]
    y_list = [y_train, y_val, y_test]

    return model_ft, history, mean, std, X_list, y_list

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

