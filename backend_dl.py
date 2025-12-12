"""Backend para el DL, acá estarán todas las funciones para interactuar con el """

import numpy as np
import tensorflow as tf



from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import clone_model


import os
from time import perf_counter
from EEGModels import EEGNet  
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

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

def eeg_fine(base, X, y,sub,  classes=3, test_size=0.2, epochs=60):

    """Función a utilizar cuando tenemos aplicar un fine tunning a un modelo general
    base = modelo general
    X,y, sub = las señales del sujeto (Pronto lo ajustaré para aceptar un sub = 10)
    clases = las clases del modelo a detectar
    epochs: Al ser de un dataset pequeño, recomiendo 150"""

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

    #Normalizamos los datos en base a train
    channel_mean = X_train.mean(axis=(0, 2))  
    channel_std  = X_train.std(axis=(0, 2))   

    eps=1e-6
    def normalize_by_channel(X, mean, std, eps=1e-6):
        # X: (N, 8, 160)
        # mean, std: (8,)
        return (X - mean[None, :, None]) / (std[None, :, None] + eps)
    
    X_train_norm = normalize_by_channel(X_train, channel_mean, channel_std, eps)
    X_test_norm  = normalize_by_channel(X_test,  channel_mean, channel_std, eps)
    X_val_norm   = normalize_by_channel(X_val,   channel_mean, channel_std)

    #Ajustar dimensiones para keras (N, C, T, 1)
    X_train_norm = X_train_norm[:, :, :, None]
    X_test_norm  = X_test_norm[:, :, :, None]
    X_val_norm   = X_val_norm[:,   :, :, None]

    # Paso 3: clonar el modelo base para no modificarlo in-place
    model_ft = clone_model(base)
    model_ft.set_weights(base.get_weights())

    for layer in model_ft.layers:
        layer.trainable = True

    # Paso 4: congelar capas iniciales y dejar entrenables sólo las últimas
    # Aquí dejo una regla simple: congelar todo menos las 3 últimas capas.
    # Ajusta este número según tu EEGNet real.
    # n_layers = len(model_ft.layers)
    # freeze_until = max(0, n_layers - 3)

    # for i, layer in enumerate(model_ft.layers):
    #     layer.trainable = (i >= freeze_until)



    # Paso 5: recompilar con LR más bajo para fine-tuning
    model_ft.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=5e-5),
        metrics=['accuracy']
    )

    # Paso 6: callbacks específicos para fine-tuning
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



    # Paso 7: entrenamiento de fine-tuning
    history = model_ft.fit(
        X_train_norm, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val_norm, y_val),
        callbacks=callbacks
    )

    # Paso 8: evaluación en test del sujeto
    test_loss, test_acc = model_ft.evaluate(X_test_norm, y_test, verbose=0)
    print(f"[Fine-tune] Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Opcional: devolver también los splits por si quieres calcular F1 afuera
    X_list = [X_train_norm, X_val_norm, X_test_norm]
    y_list = [y_train, y_val, y_test]

    return model_ft, history, channel_mean, channel_std, X_list, y_list


def eeg_train(X, y, sub, test_size = 0.2, classes=9, epochs=20, debug = False):
    """
    Entrenamiento de un modelo general, donde se recomienda entrenar con un modelo con más de 10 sujetos, en caso de tener pocos sujetos,
    se estratificará por clases
    Se usan épocas más chicas dado que ay muchos más datos
    """
    
    # 1) Sujetos únicos
    unique_subj = np.unique(sub)

    if (len(unique_subj) < 10):
        #Armamos el train con stratify en clases y no sujetos
                
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y  # aquí sí estratificamos por clase
        )

        X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,       # 15% y 15% del total
        random_state=152,
        stratify=y_temp
        )
    


    else:
        # 2) Split en sujetos train y test
        train_subj, test_subj = train_test_split(
            unique_subj,
            test_size=test_size,
            random_state=13
        )



        # 3) Máscaras por ventana usando esos sujetos
        mask_train = np.isin(sub, train_subj)
        mask_test  = np.isin(sub, test_subj)

        X_train, y_train = X[mask_train], y[mask_train]
        X_test,  y_test  = X[mask_test],  y[mask_test]

        if debug:
            print("Tamaño de train vs test:", X_train.shape, X_test.shape)

        # 4) Split interno train/val (por ventanas) dentro de los sujetos de entrenamiento
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.15,
            random_state=42,
            stratify=y_train   # aquí sí estratificamos por clase
        )
    
    #Normalizamos los datos en base a train
    channel_mean = X_train.mean(axis=(0, 2))  
    channel_std  = X_train.std(axis=(0, 2))   

    eps=1e-6
    def normalize_by_channel(X, mean, std, eps=1e-6):
        # X: (N, 8, 160)
        # mean, std: (8,)
        return (X - mean[None, :, None]) / (std[None, :, None] + eps)
    
    X_train_norm = normalize_by_channel(X_train, channel_mean, channel_std, eps)
    X_test_norm  = normalize_by_channel(X_test,  channel_mean, channel_std, eps)
    X_val_norm   = normalize_by_channel(X_val,   channel_mean, channel_std)

    if debug: 
        print("Antes de normalizar:")
        print("  Media por canal:", np.mean(X_train.mean(axis=(0,2))))

        print("Después de normalizar:")
        print("  Media por canal:", np.mean(X_train_norm.mean(axis=(0,2))))

    #Ajustar dimensiones para keras (N, C, T, 1)
    X_train_norm = X_train_norm[:, :, :, None]
    X_test_norm  = X_test_norm[:, :, :, None]
    X_val_norm   = X_val_norm[:,   :, :, None]

    #largo de las señales 
    signal_len = X_train_norm.shape[2]
    chans = X_train_norm.shape[1]
    ker = signal_len // 2  
    ker = 64  # en vez de signal_len // 2
    F1   = 16  # más filtros de entrada
    D    = 2
    F2   = F1 * D  # 32
    model = EEGNet( #Probemos a ver qué tal
        nb_classes=classes,
        Chans=chans,
        Samples=signal_len,
        dropoutRate=0.5,
        kernLength=ker,   
        F1=F1,
        D=D,
        F2=F2,           # normalmente F1*D
        norm_rate=0.25,
        dropoutType='Dropout'   # o 'SpatialDropout2D'
    )
    if debug:
        model.summary()

    #Compilamos el modelo

    model.compile(
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

    if debug:
        verbose = 2
    else:
        verbose = 1
    #Entrenamiento 
    history = model.fit(
        X_train_norm, y_train,
        epochs=epochs,
        batch_size=64,   # baja a 32 si sientes la CPU muy exigida
        validation_data=(X_val_norm, y_val),
        callbacks=callbacks,
        verbose=verbose
    )

    test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
    print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    #Crear tupla para las salidas 
    
    X_return = [X_train_norm, X_val_norm, X_test_norm]
    y_return = [y_train, y_val, y_test]
 

    return model, history, channel_mean, channel_std, X_return, y_return

def pick(data, pick=None, n=None, undersample_rest=True, random_state=42):
    """
    Filtra el dataset por clases y, opcionalmente, hace undersampling de la clase 'rest'
    Va a agarrar el vector completo y va a seleccionar todos las clases elegidas en pick, enc caso de no elegir pick, se usan todas las clases

    """

    X = data["X"]
    y = data["y"]
    sub = data["subject"]



    if len(sub) != len(y):
        #Cortamos para que sean iguales
        min_len = min(len(sub), len(y))
        X = X[:min_len]
        y = y[:min_len]
        sub = sub[:min_len]

    # --- 1) Elegir clases a conservar ---
    if pick is not None:
        pick = list(pick)
        mask = np.isin(y, pick)
        X = X[mask]
        y = y[mask]
        sub = sub[mask]
        if X.shape[0] == 0:
            raise ValueError(f"No hay muestras para las clases {pick}.")
    else:
        # si no se especifica pick, usamos todas las clases presentes
        pick = sorted(np.unique(y).tolist())

    # --- 2) Undersampling de la clase 0 (rest), si procede ---
    if undersample_rest and 0 in pick:
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
                rng = np.random.default_rng(random_state)
                idx_rest_sel = rng.choice(idx_rest, size=max_other, replace=False)

                idx_keep = np.concatenate([idx_rest_sel, idx_non_rest])
                rng.shuffle(idx_keep)

                X = X[idx_keep]
                y = y[idx_keep]
                sub = sub[idx_keep]

                print(f"Undersampling: clase 0 recortada de {len(idx_rest)} a {max_other} muestras.")
        # si counts_others está vacío, significa que solo hay clase 0, no hacemos nada

    # --- 3) Submuestreo adicional por cantidad n (opcional) ---
    total = len(y)
    if n is not None and n < total:
        X = X[:n]
        y = y[:n]
        print(f"Seleccionando los primeros {n} de {total} datos tras el filtrado.")

    # --- 4) Remapeo de etiquetas a 0..C-1 ---
    # Usamos el orden de 'pick' ordenado para que 0 siga siendo 0 si estaba en pick
    unique_sorted = sorted(pick)
    mapa = {old: new for new, old in enumerate(unique_sorted)}
    y_new = np.vectorize(mapa.get)(y)

    return X, y_new, sub

def pick_old(data, n_samples=0, pick = None): #Función vieja que ya no se usa
    """Función para seleccionar una cantidad de datos específicos de un dataset grande.
    Si pick es None, se seleccionan los primeros 'n_samples' datos.
    Si pick es una lista de índices, se seleccionan esos índices específicos.
    """

    #Elegir todos las señales si n_samples es None 



    if pick is None:

        if n_samples == 0:
            n_samples = len(data["y"])
        
        X_new = data["X"][:n_samples]
        y_new = data["y"][:n_samples]
        if n_samples > len(y_new):
             raise ValueError(f"'n_samples' es mayor que la cantidad de datos disponibles después de filtrar por 'pick'. La cantidad aceptada es de {n_samples(y_new)} datos.")
        elif n_samples < len(y_new):
            print(f"Seleccionando los primeros {n_samples} datos del dataset.")

        
    else:

    

        X = data["X"] #Todas las señales 
        y = data["y"]

        mask = np.isin(y, pick)
        X_new = X[mask]
        y_new = y[mask]
        
        if n_samples == 0:
            n_samples = len(y_new)
            print(n_samples)
        #elegir los primeros n_samples datos 
        if n_samples <= len(y_new):
            #len_new = len(y_new)
            X_new = X_new[:n_samples]
            y_new = y_new[:n_samples]
            print(f"Seleccionando los primeros {n_samples} datos del dataset filtrado.")

            mapa = {old: new for new, old in enumerate(sorted(pick))}
            y_new = np.vectorize(mapa.get)(y_new)
        elif n_samples > len(y_new):
            raise ValueError(f"'n_samples' es mayor que la cantidad de datos disponibles después de filtrar por 'pick'. La cantidad aceptada es de {n_samples(y_new)} datos.")
    
    return X_new, y_new

def norest(data):
    """
    Lo usamos cuando queremos hacer clases binarias (Rest y no rest)
    """
    X = data["X"]
    y = data["y"]
    sub = data["subject"]

    y_bin = (y != 0).astype(int)  # 0 si y==0, 1 en cualquier otro caso

    return X, y_bin, sub


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

def fusionar(y): 
    """Fusionar las clases imaginarias y motoras 
    
        "rest" -> 0
    "right_i" -> 1
    "left_i"->2
    "hands_i" -> 3
    "feet_i" -> 4
    "right_m" -> 5
    "left_m" -> 6
    "hands_m" -> 7
    "feet_m" -> 8"""

    y_lr = y.copy()

    if len(np.unique(y)) == 9: #Son 5 clases en total
        y_lr[y==5] = 1
        y_lr[y==6] = 2
        y_lr[y==7] = 3
        y_lr[y==8] = 4

    elif len(np.unique(y)) == 5: #3 clases en total (manos y pies o derecha e izquierda)
        y_lr[y==3] = 1
        y_lr[y==4] = 2
    
    

    return y_lr





from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

def evaluar_rf(y_true, y_pred, class_names=None, title_prefix="Random Forest"):
    """
    y_true : array (N,) con etiquetas enteras 0..C-1
    y_pred : array (N,) con predicciones enteras 0..C-1
    class_names : lista opcional de nombres para las clases en orden 0..C-1
                  (si es None, usa "0","1",...)
    title_prefix : texto para los títulos de las figuras
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Detectamos clases presentes a partir de y_true (o el rango completo)
    labels = np.unique(y_true)
    n_classes = len(labels)

    if class_names is None:
        class_names = [str(l) for l in labels]

    print(f"\n=== Evaluación {title_prefix} ===")
    print("Clases (índice interno -> nombre):")
    for lab, name in zip(labels, class_names):
        print(f"  {lab}: {name}")

    # 1) Matriz de confusión (conteos)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nMatriz de confusión (conteos):")
    print(cm)

    # 2) Reporte de clasificación
    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0
        )
    )

    # 3) F1 por clase
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        zero_division=0
    )

    # ---- Plot F1 por clase ----
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(n_classes), f1)
    plt.xticks(np.arange(n_classes), class_names)
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.title(f"F1 por clase ({title_prefix})")
    plt.tight_layout()
    plt.show()

    # ---- Matriz de confusión normalizada ----
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
        "cm": cm,
        "cm_norm": cm_norm,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "labels": labels,
    }
