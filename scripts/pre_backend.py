#Nueva versión de la librería de preprocesamiento, ahora con clases para nuestro dataset :P 

import os
from typing import Optional, Tuple, Union
import numpy as np
import mne
from time import perf_counter
from pathlib import Path
from typing import List
from sklearn.preprocessing import StandardScaler
import re #Para buscar el número final
import glob #Para buscar archivos con tipos específicos
import pandas as pd
import matplotlib.pyplot as plt
class EEGPreprocess: 

    def __init__(self, channels = None, l_freq = 8, h_freq = 30.0, use_notch = True, notch_freqs = (60,), window_size = 1.0, window_step = 0.5,
                new_freq = None, tmin = 0.5, tmax =3.5, Debug = False, path = None, config = None, Fs = None):

        #Parámetros de salida de nuestro array:
        self.X = None
        self.y = None
        self.sub = None
        self.name = None
        self.run = None #Sección del sujeto
        self.trial = None #ID de la partición de la prueba}
        self.mode = None #Modo de la prueba (LR_M, LR_I, etc.)

        #Parámetros de entrada

        self.channels = channels
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.use_notch = use_notch
        self.path = path
        self.new_freq = new_freq #Recordatorio: crear función de resampleo en caso de 
        self.tmin = tmin
        self.tmax = tmax
        self.windows = window_size
        self.window_step = window_step
        self.Debug = Debug
        self.notch_freqs = notch_freqs
        self.Fs = Fs

        self.dropout_df = None

        #Metadaatos

        self.class_names = None
        self.channels_names = None
        self.sfreq = None
        self.n_channels = None
        self.n_samples = None
        

        #Validación de la configuración de eventos

        self.config = config
        self.allowed_classes = {"rest", "left_i", "right_i", "hands_i", "feet_i", "left_m", "right_m", "hands_m", "feet_m"} #Por el momento
        #Sólo vamos a aceptar estas clases con esos nombres específicos
        if not isinstance(self.config, dict):
            raise TypeError(
                "La configuración debe ser un diccionario."
            )
        
        self.event_lookup = {} 
        self.available_events = set() 

        for run_group, cfg in self.config.items():

            #run_group es la id de los runs (El número final del nombre del archivo, que corresponde al tipo de evento) 
            #cfg es la configuración de cada run, que debe contener un diccionario con "events" y "mode"

            if self.Debug:
                print(f"Validando configuración para run_group: {run_group}")
                print(f"Configuración encontrada: {cfg}")

            if not isinstance(cfg, dict):

                raise TypeError(
                    f"Config inválida en {run_group}"
                )

            if "events" not in cfg:

                raise ValueError(
                    f"Falta 'events' en {run_group}"
                )

            if "mode" not in cfg:

                raise ValueError(
                    f"Falta 'mode' en {run_group}"
                )

            event_map = cfg["events"] #La configuración de los eventos, ej: T0: rest, T1: left_i, etc

            if self.Debug:
                print(f"Validando 'events' en {run_group}: {event_map}")

            if not isinstance(event_map, dict):

                raise TypeError(
                    f"'events' debe ser dict en {run_group}"
                )

            # ---------------------------------------------
            # Validar labels
            # ---------------------------------------------

            for ann_name, label in event_map.items():
                #ann_name es la anotación del evento (T0, T1, T2) y label es la clase que corresponde a esa anotación (rest, left_i, right_i, etc.)

                if self.Debug:
                    print(f"Validando anotación '{ann_name}' con label '{label}' en {run_group}")

                # Validar anotación

                if not isinstance(ann_name, str):

                    raise TypeError(
                        f"Evento inválido: {ann_name}"
                    )

                if not re.match(r"^T\d+$", ann_name):

                    raise ValueError(
                        f"Formato inválido de anotación: {ann_name}"
                    )

                # Validar label

                if label not in self.allowed_classes:

                    raise ValueError(
                        f"Clase no permitida: {label}"
                    )

            # ---------------------------------------------
            # Construir lookup rápido
            # ---------------------------------------------

            for run_id in run_group:

                if run_id in self.event_lookup:

                    raise ValueError(
                        f"Run duplicado: {run_id}"
                    )

                self.event_lookup[run_id] = cfg
                self.available_events.add(run_id)

                #self.event_lookup[n] va a tener la configuración de cada evento, es decir, un diccionario con "events" y "mode", que a su vez "events" es un diccionario con la anotación (T0, T1, T2) y su respectiva clase (rest, left_i, etc.)
                #self.available_events es un set con todos los runs disponibles, es decir, los números finales de los archivos que corresponden a eventos que sí vamos a procesar (3,4,5,6,7,8,9,10,11,12,13,14)
                #por ejemplo self.event_lookup[3] va a tener la configuración de los eventos con terminación 3, que corresponde a movimiento real mano izquierda o derecha, es decir, un diccionario con "events" y "mode", donde "events" es un diccionario con la anotación (T0, T1, T2) y su respectiva clase (rest, left_m, right_m) y "mode" es "LR_M"

        if self.Debug:
            print(f"Configuración de eventos validada.")

            print(f"event_lookup: {self.event_lookup}")
            print(f"available_events: {self.available_events}")

    def load(self, path = None, show_channels = False, Debug = None):

        if Debug is None: 
            Debug = self.Debug

        if self.path is None and path is not None:
            self.path = path
        elif self.path is not None and path is not None and self.path != path:
            print(f"⚠️ Advertencia: Se proporcionó un nuevo path '{path}' pero ya existe un path en la instancia ('{self.path}'). Se usará el path de la instancia.")
        elif self.path is not None and path is None:
            pass #Usamos el path que ya está en la instancia
        else:
            raise ValueError("No se ha proporcionado un path para cargar los datos. Por favor, especifica un path válido.") 

        subject_dirs = [os.path.join(self.path, d) for d in os.listdir(self.path)]
        subject_dirs = [d for d in subject_dirs if os.path.isdir(d)] #Creamos un array con todas las carpetas

        unknown_events = set()


        
        if Debug:
            print(f"Total de directorios encontrados: {len(subject_dirs)}")
            print(subject_dirs)
         
        self.array = []

        for sujeto in subject_dirs: 
            if Debug:
                print(f"📂 Revisando directorio: {sujeto}")
            
            subject_name = os.path.basename(sujeto)   # e.g. "S001"
            subject_id = int(subject_name[1:])    # 1..109

            edf_files = glob.glob(os.path.join(sujeto, "*.edf")) #Busca todos los archivos adentro que terminen en edf
            #IMPORTANTE QUE TERMINEN EN EDF!!!!

           
            for f in edf_files:
                ff = os.path.basename(f)
                #f= la dirección del archivo
                #ff= El nombre del archivo como tal 

                

                n_2 = re.search(r'(\d{2})(?=\.edf$)', ff, flags=re.IGNORECASE) #m son los dos últimos digitos antes del .edf, que corresponde al tipo de evento

                """Según nuestro dataset original, las terminaciones de los archivos son:
                1: ojos abiertos
                2: ojos cerrados
                3,7,11: Movimiento real mano derecha o izquierda
                4, 8, 12: movimiento imaginado mano derecha o izquierda
                5, 9, 13: movimiento real ambos brazos o ambos pies
                6, 10, 14: movimiento imaginado ambos brazos o ambos pies
                """

                if Debug:
                    print(f"🔍 Procesando archivo: {ff}")
                    
                
                if not n_2:
                    continue


                n_1 = int(n_2.group(1)) #n muestra 
                

                if n_1 not in self.available_events:

                    if Debug:
                        print(
                            f"⚠️ Evento {n_1} ignorado "
                            f"(no definido en dataset_config)"
                        )
                    unknown_events.add(n_1)

                    continue
                    
                if not f.lower().endswith(".edf"): #Asegurarnos de ue es un .edf y no un .event (Aunque ya lo verificamos antes )
                    print(f"Se coló un archivo no .edf = {ff}. REVISAR CÓDIGO")
                    continue

                #Si llegamos aquí, es porque el archivo es un .edf y corresponde a un evento que nos interesa
                if Debug:
                    print(f"✅ Archivo {n_1} válido para procesamiento.")

                    
                record_n = {
                "path": f,
                "event_id": n_1,
                "subject": subject_id
                }

                self.array.append(record_n) #Agregamos el diccionario a nuestro array de archivos válidos
            

                if Debug:
                    print(f"Archivo {ff} agregado para procesamiento. Total archivos válidos hasta ahora: {len(self.array)}")

        if show_channels:
            print("\n🔍 Canales encontrados en el primer archivo válido:")
            if self.array:
                first_file = self.array[0]['path']
                raw = mne.io.read_raw_edf(first_file, preload=False, verbose='ERROR')
                print(raw.ch_names)
            else:
                print("No se encontraron archivos válidos para mostrar canales.")
        if unknown_events:

            print(
                f"⚠️ Eventos ignorados no definidos "
                f"en config: {sorted(unknown_events)}"
            )




        return self.array #Entregaremos para procesar una lista con todos los directorios y sus respectivos tipos de eventos y sujetos, para luego procesarlos en la función de procesamiento.
    
    def build(self, array: List[dict], Debug = False, reasons_only = False,
               show_dropouts = 0, dropout_rate = 150e-6, dropout = True,
               save_df=True, channels = None,
               l_freq = None, h_freq =None, use_notch = None, notch_freqs = None, #Ahora los nuevos parámetros en caso de que no se hayan declarado antes
               window_size = None, window_step = None, tmin = None, tmax = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        
        if self.channels is None: #Nuevas declaraciones en caso de que no se bhyaan declarado en el init
            self.channels = channels 
        if l_freq is not None:
            self.l_freq = l_freq
        if h_freq is not None:
            self.h_freq = h_freq
        if use_notch is not None:
            self.use_notch = use_notch
        if notch_freqs is not None:
            self.notch_freqs = notch_freqs
        if window_size is not None:
            self.windows = window_size
        if window_step is not None:
            self.window_step = window_step
        if tmin is not None:
            self.tmin = tmin
        if tmax is not None:
            self.tmax = tmax
        

        X_list = [] #Lista para guardar los datos de cada archivo
        y_list = [] #Lista para guardar las etiquetas de cada archivo
        sub_list = [] #Lista para guardar el ID del sujeto de cada archivo
        run_list = [] #Lista para guardar la sección del sujeto de cada archivo
        trial_list = [] #Lista para guardar el ID de la partición de la prueba de cada archivo
        mode_list = [] #Lista para guardar el modo de la prueba de cada archivo

        global_trial_offset = 0
        fit_Fs = 0

        if Debug:
            print(f"📂 Procesando {len(array)} archivos...")

        sub_counts_before = {}
        sub_counts_after = {}



        #Bien, llegó la hora de llenar cada uno de estos arrays B) 

        print(f"Total de archivos a procesar: {len(array)} en {len(set([rec['subject'] for rec in array]))} sujetos")
        if show_dropouts:
            print(f"Dropout rate: {dropout_rate}")
        if self.channels is not None:
            print(f"Canales seleccionados: {self.channels}")
        

        for record in array: #Recorre 
            path = record["path"]
            event_id = record["event_id"]
            subject_id = record["subject"]

            n_raw = subject_id + event_id

            if Debug:
                print(f"Procesando archivo: {path} | Evento: {event_id} | Sujeto: {subject_id}")

            
            #1) Cargar el archivo edf
            
            raw = self._load_raw(path)

            if self.Fs is None and fit_Fs == 0: 
                self.Fs = raw.info['sfreq'] #Tomamos la Fs del primer archivo que cargamos y no lo volvemos a tocar 
                fit_Fs = 1 #No volvemos a tocar la Fs
                if Debug:
                    print(f"Frecuencia de muestreo establecida en {self.Fs} Hz a partir del primer archivo cargado.")
            
            if self.Fs is not None and raw.info['sfreq'] != self.Fs:

                print(f"⚠️ Advertencia: Archivo {path} tiene frecuencia de muestreo {raw.info['sfreq']} Hz, pero se esperaba {self.Fs} Hz. Se realizará resampleo.")
                raw.resample(self.Fs, verbose='ERROR')
                if Debug:
                    print(f"Archivo {path} resampleado a {raw.info['sfreq']} Hz.")


            #2) Obtener labels y demás metadatos

            labels, mode, event_map = self._get_labels(event_id, Debug = Debug) #Genial! ya tenemos etiquetas y el raw! vamos con lo siguiente:
            if Debug:
                print(f"Etiquetas obtenidas: {labels}")
                print(f"Modo de la prueba: {mode}")
                print(f"Mapeo de eventos: {event_map}")

            #3) Preprocesar el raw 

            clean, elapsed = self._preprocess(raw)

            #4) Epocar el raw limpio con las etiquetas obtenidas

            epochs = self._epochs(clean, event_map = event_map, tmin=self.tmin, tmax=self.tmax, debug=Debug, n_raw = n_raw) #Obtenemos las épocas con sus respectivas etiquetas, listas para ser windowizadas

            if dropout:
                #5) Eliminar épocas con artefactos
                n_before = len(epochs)
                sub_counts_before[subject_id] = sub_counts_before.get(subject_id, 0) + n_before

                epochs.drop_bad(reject=dict(eeg=dropout_rate), verbose = "Error")

                n_after = len(epochs)
                sub_counts_after[subject_id] = sub_counts_after.get(subject_id, 0) + n_after

                n_dropped = n_before - n_after

                if show_dropouts>1 and n_dropped > 0:
                    
                    ratio = n_dropped / n_before * 100
                    print(f"⚠️ Dropout: {n_dropped}/{n_before} ({ratio:.1f}%) en {os.path.basename(path)}. De un total de {n_before} epocas.")
                    
                    reasons = {}
                    drop_log = epochs.drop_log

                    for log in drop_log:
                        if len(log) == 0:
                            continue
                        key = tuple(log)
                        reasons[key] = reasons.get(key, 0) + 1

                    if reasons and reasons_only:
                        print("🔍 Razones de eliminación:")
                        for k, v in reasons.items():
                            print(f"{k}: {v}")


            if len(epochs) == 0: 
                continue
            

            #Habemus primeros datos para las listas!

            X, y, trial = self._window(epochs, size=self.windows, step=self.window_step, debug = Debug)

            #Faltan sub, run y trial

            sub = np.full(len(y), subject_id)
            run = np.full(len(y), event_id)

            assert len(X) == len(y) == len(sub) == len(run) == len(trial), "Inconsistencia en la longitud de los arrays generados!"

            trial_file = trial + global_trial_offset
            global_trial_offset += max(trial) + 1

            X_list.append(X)
            y_list.append(y)
            sub_list.append(sub)
            run_list.append(run)
            trial_list.append(trial_file)
            mode_arr = np.full(len(y), mode)    
            mode_list.append(mode_arr)
            if Debug: 
                print("===================")

            assert len(X_list) == len(y_list) == len(sub_list) == len(run_list) == len(trial_list) == len(mode_list), "Inconsistencia en la longitud de las listas acumuladoras!"

        self.X, self.y, self.sub, self.run, self.trial, self.mode = self._stack(X_list, y_list, sub_list, run_list, trial_list, mode_list)

        assert len(self.X) == len(self.y) == len(self.sub) == len(self.run) == len(self.trial), "Inconsistencia en la longitud de los arrays finales después del stack!"
        

        if show_dropouts > 0:
    
            print("\n📊 Dropout por sujeto:")

            for sub_id in sub_counts_before.keys():
                before = sub_counts_before.get(sub_id, 0)
                after = sub_counts_after.get(sub_id, 0)

                if before == 0:
                    continue

                dropped = before - after
                ratio = dropped / before * 100

                
                if ratio > 90:
                    print(f"Sujeto {sub_id:03d}: {before} → {after} | Dropout: {dropped} ({ratio:.1f}%)💀")
                elif ratio > 50:
                    print(f"Sujeto {sub_id:03d}: {before} → {after} | Dropout: {dropped} ({ratio:.1f}%)⚠️")
                else: 
                    print(f"Sujeto {sub_id:03d}: {before} → {after} | Dropout: {dropped} ({ratio:.1f}%)")
        
        #Ahora procedemos a guardar nuestra tabla en un formato pandas para, si queremos, guardarlo en un futuro 

        if save_df: 
            rows = []

            for sub_id in sub_counts_before.keys():
                before = sub_counts_before.get(sub_id, 0)
                after = sub_counts_after.get(sub_id, 0)

                if before == 0:
                    continue

                dropped = before - after
                ratio = dropped / before * 100

                rows.append({
                    "subject": sub_id,
                    "before_epochs": before,
                    "after_epochs": after,
                    "dropped_epochs": dropped,
                    "dropout_%": ratio
                })

            df_dropout = pd.DataFrame(rows)
            self.dropout_df = df_dropout
            
            if show_dropouts > 0:
                print(f"\n📊 Tabla de Dropout ({dropout_rate}) por sujeto:")
            
                df_dropout.sort_values("dropout_%", ascending=False)

                df_dropout["dropout_%"].describe()
                df_dropout["dropout_%"].hist(bins=20)
                plt.title("Distribución de Dropout por Sujeto")
                plt.xlabel("Dropout (%)")
                plt.ylabel("Cantidad de sujetos")
                plt.show()

            self.n_subjects = len(set([rec['subject'] for rec in array]))
            self.dropout_name_str = str(dropout_rate)
            name_channels = len(self.channels) if self.channels is not None else "all"

            path_base = os.getcwd()
            df_name = os.path.join(path_base, f"dropout_rate/{self.n_subjects}_subjects/{name_channels}_channels/dropout_{self.dropout_name_str}.csv")

            if not os.path.exists(os.path.dirname(df_name)):
                os.makedirs(os.path.dirname(df_name))

            df_dropout.to_csv(df_name, index=False)
            print(f"Tabla guardada en {df_name}!")





        

        #Ahora debemos transformar y a un formato numérico

        self._idx(self.y)

        print(f"Datos procesados: {self.X.shape[0]} ventanas, {self.X.shape[1]} canales, {self.X.shape[2]} muestras por ventana.")

        #Y crear diccionario para los canales
        self.channels_names = epochs.ch_names
        

        #Y terminamos de llenar el resto de metadatos 

        
        self.n_channels = self.X.shape[1]
        self.n_samples = self.X.shape[0]

        #Detectar inconsistencias en datos y levantar valores 

        if len(self.y) != self.X.shape[0]:
             raise ValueError(f"Longitud de y ({len(self.y)}) no coincide con número de ventanas en X ({self.X.shape[0]}).")

        return self
            

    def _load_raw(self, path: str) -> mne.io.Raw:


        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')

        if self.channels is not None:
                
            pick_avail = [ch for ch in self.channels if ch in raw.ch_names]

            

            if not pick_avail:
                raise ValueError(f"Ningún canal de {self.channels} está en los datos: {raw.ch_names}")
            #Ver si hay algún canal que no esté
            if len(pick_avail) < len(self.channels):
                missing = set(self.channels) - set(pick_avail)
                print(f"Advertencia: faltan canales {missing} en los datos.")

            raw.pick(pick_avail, verbose='ERROR')

            
            #Si picks es None, entonces se cargan todos los canales disponibles

        return raw
    
    def _get_labels(self, event_id: int, Debug = False):

        #Primero verificamos que el evento exista

        if event_id not in self.event_lookup:

            raise ValueError(
                f"Evento {event_id} no definido "
                f"en dataset_config."
            )

        #Obtenemos configuración, cfg es un diccionario

        cfg = self.event_lookup[event_id]

        mode = cfg["mode"]

        event_map = cfg["events"]

        #Obtenemos etiquetas

        labels = list(dict.fromkeys(event_map.values()))

        # ==========================================
        # Debug
        # ==========================================

        if Debug:
            print(f"Evento {event_id} → Modo: {mode}, Etiquetas: {labels}, Mapeo: {event_map}")

        return labels, mode, event_map
    
    def _preprocess( #Preprocesamiento: filtrado, notch y CAR
        self,
        
        raw: mne.io.Raw,
        ref: str = 'average',
        filter_method: str = 'fir'  #'fir' (MNE por defecto) o 'iir'
    ):
        t0 = perf_counter()
        
        raw_clean = raw.copy().load_data()

        #Filtrado
        raw_clean.filter(l_freq=self.l_freq, h_freq=self.h_freq, method=filter_method, verbose='ERROR')

        #notch
        if self.use_notch:
            raw_clean.notch_filter(freqs=self.notch_freqs, verbose='ERROR')

        
        #Rereferenciado 
        raw_clean.set_eeg_reference(ref, verbose='ERROR')

        self.sfreq = raw_clean.info['sfreq']

        if self.new_freq is not None:
            if self.new_freq != self.sfreq:
                print(f"Se cambiará la frecuencia de muestreo de {self.sfreq} Hz a {self.new_freq} Hz (upsampling).")
                raw_clean.resample(self.new_freq, verbose='ERROR')
                self.sfreq = self.new_freq
            else: 
                print(f"La frecuencia de muestreo ya es {self.sfreq} Hz, no se realizará resampleo.")
        self.samples = int(round(self.windows * self.sfreq))

        elapsed_ms = (perf_counter() - t0) * 1000.0
        return raw_clean, elapsed_ms
    
    def _epochs(

        self,
        raw_clean: mne.io.BaseRaw,
        event_map: dict,
        tmin: float = 0.5,
        tmax: float = 3.5,
        preload: bool = True,
        debug: bool = False,
        n_raw = None,
        verbose: str = "ERROR"

    ):

        #Leer los eventos del edf y mapearlos a nuestras etiquetas internas

        events, ann_dict = mne.events_from_annotations(
            raw_clean,
            verbose="ERROR"
        )

        inv_ann = {v: k for k, v in ann_dict.items()}

        #Pbtenemos las clases en orden

        wanted_labels = list(
            dict.fromkeys(event_map.values())
        )

        label2code = {
            lab: idx
            for idx, lab in enumerate(wanted_labels)
        }

        #Mapear eventos

        mapped = []

        detected_annotations = set()

        for sample, _, code_int in events:

            name = inv_ann.get(code_int, None)

            if name is None:
                continue

            name = name.strip().upper()

            detected_annotations.add(name)

            

            if name not in event_map: #Si el evento no se encuentra en el diccionario

                print(
                    f"""
                        Evento inesperado detectado en EDF.

                        Evento encontrado:
                        {name}

                        Eventos esperados:
                        {list(event_map.keys())}
                        """
                                    )
                continue

            #Asignar los labels correspondientes a cada evento

            label = event_map[name]

            tgt = label2code[label]

            mapped.append([sample, 0, tgt])

        #Eventos faltantes

        missing = set(event_map.keys()) - detected_annotations

        if missing:

            print(
                f"⚠️ Eventos configurados "
                f"pero no encontrados: {missing}"
            )

        #Buscar si hay eventos faltantes

        if len(mapped) == 0:

            raise ValueError(
                "No se encontraron eventos válidos."
            )

        #Convertir a numpy array

        mapped = np.array(mapped, dtype=int)

        #Crear épocas con los eventos mapeados
        epochs = mne.Epochs(

            raw=raw_clean,

            events=mapped,

            event_id=label2code,

            tmin=0,

            tmax=max(tmax, self.windows),

            baseline=None,

            preload=preload,

            verbose=verbose
        )

        #Recortamos las épocas al rango deseado (tmin, tmax)

        epochs.crop(
            tmin=tmin,
            tmax=tmax
        )

        #Comprobar que las clases sean las esperadas

        present_codes = np.unique(epochs.events[:, 2])

        present_labels = {

            lab
            for lab, code in label2code.items()
            if code in present_codes
        }

        expected_labels = set(wanted_labels)

        if present_labels != expected_labels:

            raise ValueError(
                f"""
                Mismatch de clases detectado.

                Clases esperadas:
                {expected_labels}

                Clases detectadas:
                {present_labels}
                """
                        )

        #Debug

        if debug:

            

            print(f"RAW: {n_raw}")

            print(f"Eventos detectados: {detected_annotations}")

            print(f"Eventos esperados: {set(event_map.keys())}")

            print(f"Clases finales: {wanted_labels}")

            print(f"N epochs: {len(epochs)}")

            

        return epochs
    
    def _window( #Aplicar muchas ventanas de tiempo y solape determinado
        self,
        epochs: mne.Epochs,
        size: float = 1.0,
        step: float = 0.5, 
        debug = False

        
    ):
        """Convierte una ventana épocas de t tiempo en multiples ventanas de size segundos con overlap, asegurándose siempre
        de terminar siempre con una ventana que llegue hasta el final del raw y ni se sobrepase, es decir, si cada época tiene 4 segundos la
        última ventana será de 3-4 segundos.
        
        epochs: objeto done se encuentra todas las épocas

        Retorna: 

            X_win : np.ndarray
            Ventanas con shape (N_ventanas_totales, n_channels, win_samp).
            y_win : np.ndarray
            Etiquetas por ventana, shape (N_ventanas_totales,).
        """

        #Datos base 
        X = epochs.get_data() #Obtiene todos los datos 
        y_ids = epochs.events[:, 2] #Obtiene los eventos 
        sfreq = epochs.info['sfreq']
        

        n_epochs, n_channels, n_times = X.shape

        

        #Sacar las clases

        event_id = epochs.event_id
        


        id_to_event = {v: k for k, v in event_id.items()}
        y_epoch = np.vectorize(id_to_event.get)(y_ids)  # (n_epochs,)

        # print(y_ids)
        # print(sfreq)
        # print(id_to_class)

        # --- Tamaños en muestras ---
        win_samp  = int(round(size  * sfreq))
        step_samp = int(round(step * sfreq))

        

        if win_samp > n_times:
            raise ValueError(f"La ventana ({win_samp} muestras) es más larga que el epoch ({n_times}).")

        X_windows = []
        y_windows = []
        trial_windows=[]

        for ep_idx in range(n_epochs):
            ep_data = X[ep_idx]         # (n_channels, n_times)
            label  = y_epoch[ep_idx]
            

            # recorre inicio en muestras: 0, step, 2*step, ...
            for start in range(0, n_times - win_samp + 1, step_samp):
                end = start + win_samp
                window = ep_data[:, start:end]  # (n_channels, win_samp)

                X_windows.append(window)
                y_windows.append(label)
                trial_windows.append(ep_idx)

        X_windows = np.stack(X_windows, axis=0)   # (N_ventanas, n_channels, win_samp)
        y_windows = np.array(y_windows)
        trial_local = np.array(trial_windows)

        if debug: 
            print(f"Archivo con {n_epochs} épocas → {len(X_windows)} ventanas (size={size}s, step={step}s)")
        

        if not (len(X_windows) == len(y_windows) == len(trial_local)):
            raise ValueError(f"Inconsistencia en número de ventanas: {len(X_windows)} vs {len(y_windows)} vs {len(trial_local)} ERROR EN _WINDOWS")
        

        return X_windows, y_windows, trial_local
    
    def _stack(self, X_list: List[np.ndarray],
                y_list: List[np.ndarray], sub_list: List[np.ndarray], run_list: List[np.ndarray], trial_list: List[np.ndarray], mode_list: List[np.ndarray]):
        """
        Apila tensores 3D ignorando entradas inválidas.
        
        X_list: [ (ni, C, T), ... ]
        y_list: [ (ni,), ... ]

        Devuelve:
            X_total: (N_total, C, T)
            y_total: (N_total,)
        """

        X_ok = []
        y_ok = []
        sub_ok = []
        run_ok = []
        trial_ok = []
        mode_ok = []

        
        ref_shape = X_list[0].shape[1:]  # (C, T)

        

        for i in range(len(X_list)):
            X = X_list[i]

            if X.shape[1:] != ref_shape:
                if self.Debug:  
                    print(f"⚠️  Aviso: Se descarta X_list[{i}] con shape {X.shape[1:]}, ")
                    print(f"se esperaba {ref_shape}")

                
                continue

            X_ok.append(X)
            y_ok.append(y_list[i])
            sub_ok.append(sub_list[i])
            run_ok.append(run_list[i])
            trial_ok.append(trial_list[i])
            mode_ok.append(mode_list[i])

        X = np.concatenate(X_ok, axis=0)
        y = np.concatenate(y_ok, axis=0)
        sub = np.concatenate(sub_ok, axis=0)
        run = np.concatenate(run_ok, axis=0)
        trial = np.concatenate(trial_ok, axis=0)
        mode = np.concatenate(mode_ok, axis=0)

        return X, y, sub, run, trial, mode


    def _idx(self, y: np.ndarray) -> None:

        """
        Convierte etiquetas string a índices numéricos canónicos fijos.
        El mapping es siempre el mismo independientemente de qué clases
        estén presentes en el dataset:

            rest     → 0
            right_i  → 1
            left_i   → 2
            hands_i  → 3
            feet_i   → 4
            right_m  → 5
            left_m   → 6
            hands_m  → 7
            feet_m   → 8
        """

        # =====================================================
        # Mapping canónico fijo
        # =====================================================

        LABEL_TO_IDX = {
            "rest":    0,
            "right_i": 1,
            "left_i":  2,
            "hands_i": 3,
            "feet_i":  4,
            "right_m": 5,
            "left_m":  6,
            "hands_m": 7,
            "feet_m":  8,
        }

        IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

        # =====================================================
        # Validar que todas las etiquetas en y sean conocidas
        # =====================================================

        present = set(y)
        unknown = present - set(LABEL_TO_IDX)

        if unknown:
            raise ValueError(
                f"Etiquetas desconocidas en y: {unknown}. "
                f"Solo se aceptan: {set(LABEL_TO_IDX)}"
            )

        # class_names solo refleja las clases que realmente aparecen,
        # pero sus índices numéricos respetan siempre el mapping canónico.
        self.class_names = IDX_TO_LABEL  # dict completo idx→label

        # =====================================================
        # Debug
        # =====================================================

        if self.Debug:

            print("====================================")
            print("Mapping canónico (completo):")
            for label, idx in LABEL_TO_IDX.items():
                marker = " ✓" if label in present else ""
                print(f"  {idx}: {label}{marker}")
            print(f"Clases presentes en este dataset: {sorted(present)}")
            print("====================================")

        # =====================================================
        # Convertir labels
        # =====================================================

        self.y = np.array(
            [LABEL_TO_IDX[label] for label in y],
            dtype=np.int32
        )

        # =====================================================
        # Guardar mapping
        # =====================================================

        self.label_to_idx = LABEL_TO_IDX
        self.idx_to_label = IDX_TO_LABEL
    
    def resume(self):
        print("\n" + "="*50)
        print("🧠 RESUMEN DEL DATASET EEG")
        print("="*50)

        # Estado
        if self.X is None:
            print("❌ Dataset no construido aún.")
            return

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
        print(f"Número de clases: {len(self.class_names)}")
        print(f"Lista: {self.class_names}")

        # Distribución de clases
        print("\n📊 Distribución de clases presentes:")
        unique, counts = np.unique(self.y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u} ({self.idx_to_label[u]:10s}): {c}")

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

    def save(self, path: Union[str, Path], name: str ) -> None:
        if self.X is None or self.y is None:
            raise ValueError("No hay datos para guardar. Asegúrate de haber llamado a build() primero.")
        
        if name is None:
            name = f"eeg_dataset_{self.name}.npz"

        if path is None:
            path = f"./{name}"
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        np.savez_compressed(
            path,
            name = name,
            X=self.X.astype(np.float32),
            y=self.y.astype(np.int32),
            sub=self.sub.astype(np.int32),
            run=self.run.astype(np.int32),
            trial=self.trial.astype(np.int32),
            class_names=np.array(
                [self.idx_to_label[i] for i in range(len(self.idx_to_label))],
                dtype=object
            ),
            channel_names=np.array(self.channels_names, dtype=object),
            sfreq=np.array([self.sfreq], dtype=np.float32),
            window_size=np.array([self.windows], dtype=np.float32),
            window_step=np.array([self.window_step], dtype=np.float32),
            tmin=np.array([self.tmin], dtype=np.float32),
            tmax=np.array([self.tmax], dtype=np.float32),
            mode = np.array(self.mode, dtype=str)
        )

        print(f"✔️ Dataset {name} guardado en: {path}!")


    