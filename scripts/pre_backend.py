#Nueva versión de la librería de preprocesamiento, ahora con clases para nuestro dataset :P 

import os
from typing import Optional, Tuple, Union
import numpy as np
import mne
import matplotlib.pyplot as plt
from time import perf_counter
from pathlib import Path
from typing import List
from sklearn.preprocessing import StandardScaler
import re #Para buscar el número final
import glob #Para buscar archivos con tipos específicos

class EEGPreprocess: 

    def __init__(self, channels = None, l_freq = 8, h_freq = 30.0, use_notch = True, notch_freqs = (50,), resample_freq = False, new_freq = 128, Window_size = 1.0, tmin = 0.5, tmax =3.5, Debug = False):

        #Parámetros de salida de nuestro array:
        self.X = None
        self.y = None
        self.sub = None
        self.data_dir = None
        self.run = None #Sección del sujeto
        self.trial = None #ID de la partición de la prueba}

        #Parámetros de entrada

        self.channels = channels
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.use_notch = use_notch
        self.resample_freq = resample_freq
        self.new_freq = new_freq
        self.Window_size = Window_size
        self.tmin = tmin
        self.tmax = tmax
        self.Debug = Debug
        self.notch_freqs = notch_freqs

        #Metadaatos

        self.class_names = None
        self.channels_names = None
        self.sfreq = None
        self.n_channels = None
        self.n_samples = None
        self.name = None

    def load(self, path: Union[str, Path]):

        subject_dirs = [os.path.join(path, d) for d in os.listdir(path)]
        subject_dirs = [d for d in subject_dirs if os.path.isdir(d)] #Creamos un array con todas las carpetas


        
        if self.Debug:
            print(f"Total de directorios encontrados: {len(subject_dirs)}")
            print(subject_dirs)
         
        self.array = []

        for sujeto in subject_dirs: 
            if self.Debug:
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

                if self.Debug:
                    print(f"🔍 Procesando archivo: {ff}")
                    
                
                if not n_2:
                    continue


                n_1 = int(n_2.group(1)) #n muestra 
                
                if self.Debug:
                    print(f" Analizando evento: {n_1}")

                if n_1 in {1, 2}:
                    

                    if self.Debug:
                        print(f" Evento {n_1}  ignorado.")

                    continue
                    
                if not f.lower().endswith(".edf"): #Asegurarnos de ue es un .edf y no un .event (Aunque ya lo verificamos antes )
                    print(f"Se coló un archivo no .edf = {ff}. REVISAR CÓDIGO")
                    continue

                #Si llegamos aquí, es porque el archivo es un .edf y corresponde a un evento que nos interesa
                if self.Debug:
                    print(f"✅ Archivo {ff} válido para procesamiento.")

                    
                record_n = {
                "path": f,
                "event_id": n_1,
                "subject": subject_id
                }

                self.array.append(record_n) #Agregamos el diccionario a nuestro array de archivos válidos
            

                if self.Debug:
                    print(f"Archivo {ff} agregado para procesamiento. Total archivos válidos hasta ahora: {len(self.array)}")




        return self.array #Entregaremos para procesar una lista con todos los directorios y sus respectivos tipos de eventos y sujetos, para luego procesarlos en la función de procesamiento.
    
    def build(self, array: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        

        X_list = [] #Lista para guardar los datos de cada archivo
        y_list = [] #Lista para guardar las etiquetas de cada archivo
        sub_list = [] #Lista para guardar el ID del sujeto de cada archivo
        run_list = [] #Lista para guardar la sección del sujeto de cada archivo
        trial_list = [] #Lista para guardar el ID de la partición de la prueba de cada archivo

        global_trial_offset = 0



        #Bien, llegó la hora de llenar cada uno de estos arrays B) 

        for record in array:
            path = record["path"]
            event_id = record["event_id"]
            subject_id = record["subject"]

            if self.Debug:
                print(f"Procesando archivo: {path} | Evento: {event_id} | Sujeto: {subject_id}")

            
            #1) Cargar el archivo edf
            raw = self._load_raw(path)

            #2) Obtener labels y demás metadatos

            labels, mode = self._get_labels(raw, event_id) #Genial! ya tenemos etiquetas y el raw! vamos con lo siguiente:}

            #3) Preprocesar el raw 

            clean, elapsed = self._preprocess(raw)

            #4) Epocar el raw limpio con las etiquetas obtenidas

            epochs = self._epochs(clean, labels)

            #5) Eliminar épocas con artefactos

            epochs.drop_bad(reject=dict(eeg=150e-6), verbose = "Error")


            if len(epochs) == 0: 
                continue
            

            #Habemus primeros datos para las listas!

            X, y, trial = self._window(epochs)

            #Faltan sub, run y trial

            sub = np.full(len(y), subject_id)
            run = np.full(len(y), event_id)

            trial_file = trial + global_trial_offset
            global_trial_offset += max(trial) + 1

            X_list.append(X)
            y_list.append(y)
            sub_list.append(sub)
            run_list.append(run)
            trial_list.append(trial_file)

        self.X, self.y = self._stack_3d(X_list, y_list)
        self.sub = np.concatenate(sub_list, axis=0)
        self.run = np.concatenate(run_list, axis=0)
        self.trial = np.concatenate(trial_list, axis=0)

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
    
    def _get_labels(self, raw: mne.io.Raw, event_id: int) -> mne.io.Raw:

        if self.Debug:
            print(f" Evento ID: {event_id}, Sujeto ID: {event_id}")

        if event_id in {3,7,11}:
            labels =  ['rest','left_m','right_m']
            mode = "LR_M"

        if event_id in {4,8,12}:
            labels = ['rest', 'left_i', 'right_i']
            mode = "LR_I"

        if event_id in {6,10,14}:
            labels= ['rest', 'hands_i', 'feet_i']
            mode = "HF_I"

        if event_id in {5,9,13}:
            labels =  ['rest', 'hands_m', 'feet_m']
            mode = "HF_M"

        return labels, mode
    
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

        elapsed_ms = (perf_counter() - t0) * 1000.0
        return raw_clean, elapsed_ms
    
    def _epochs( #Epocar
        
        self,
        raw_clean: mne.io.BaseRaw,
        wanted_labels: List[str],
        tmin: float = -0.5,
        tmax: float = 2.5,
        preload: bool = True,
        scale: str = 'Medium', # 'Small', 'Medium', 'Large',
        show: bool = False,
        start: float = 0.0,
        duration: float = 20.0,
        debug: bool = False,
        verbose: str = 'ERROR'

):
    
        # 1) Eventos y diccionario de anotaciones
        events, ann_dict = mne.events_from_annotations(raw_clean, verbose='ERROR') #Events nos entrega los tiempos y ann el diccionario
        inv_ann = {v: k for k, v in ann_dict.items()}  

        label2code = {lab: idx for idx, lab in enumerate(wanted_labels)}

        # ──────────────────────────────────────────────
        # 3) Función para convertir el código T0/T1/T2 a nuestras clases
        def map_event(code_int):
            """
            Convierte el entero del evento (ej. 2, 3, etc.)
            al código interno de nuestras clases (0,1,2,...)
            """
            # Recuperar nombre textual: "T0", "T1", "T2"
            name = inv_ann.get(code_int, None)
            if name is None:
                return None  # evento desconocido

            name = name.strip().upper()

            if name == "T0" and "rest" in label2code:
                return label2code["rest"]

            if name == "T1":
                # Si usuario pidió left
                if "left_i" in label2code:
                    return label2code["left_i"]
                # Si usuario pidió hands
                if "hands_i" in label2code:
                    return label2code["hands_i"]
                            
                if "left_m" in label2code:
                    return label2code["left_m"]
                # Si usuario pidió hands
                if "hands_m" in label2code:
                    return label2code["hands_m"]

            if name == "T2":
                # Si usuario pidió right
                if "right_i" in label2code:
                    return label2code["right_i"]
                # Si usuario pidió feet
                if "feet_i" in label2code:
                    return label2code["feet_i"]
                    
                if "right_m" in label2code:
                    return label2code["right_m"]
                # Si usuario pidió feet
                if "feet_m" in label2code:
                    return label2code["feet_m"]

            return None
    


        #Mapear eventos
        mapped = []
        for sample, _, code_int in events:
            tgt = map_event(code_int)
            if tgt is not None:
                mapped.append([sample, 0, tgt])


        
        #print("Ejemplo de eventos mapeados:", mapped[:10])


        # Extraer épocas 
        mapped = np.array(mapped, dtype=int)

        # Filtrar solo las clases presentes
        present_codes = np.unique(mapped[:, 2])
        present_labels = [lab for lab, code in label2code.items() if code in present_codes]
        event_id = {lab: label2code[lab] for lab in present_labels}

        # Crear épocas
        epochs = mne.Epochs(
            raw=raw_clean, 
            events=mapped,
            event_id=event_id,
            tmin=tmin, 
            tmax=tmax,
            baseline=(None, 0.0),  # baseline hasta evento, más estándar
            preload=preload, 
            verbose=verbose
        )

        if debug:
            print(f"Épocas extraídas: {len(epochs)}")
            print(f"Clases presentes: {present_labels}")
            print(epochs)

        if scale == 'small':
            scaling = {'eeg': 100e-6}
        elif scale == 'medium':
            scaling = {'eeg': 50e-6}
        elif scale == 'large':
            scaling = {'eeg': 20e-6}
        else: 
            scaling = None


        #Plotear 10 segundos de los canales con sus eventos (Sanity check)
        #Asegurarse de que cada señal esté bien alineada a su celda y no se mezclen ni superpongan
        if show:
            raw_clean.plot(duration=duration, start=start, scalings=scaling,remove_dc=True,show_scrollbars=False,block=True)
        else:
            pass
        #Retornamos las épocas
        return epochs
    
    def _window( #Aplicar muchas ventanas de tiempo y solape determinado
        self,
        epochs: mne.Epochs,
        size: float = 1.0,
        step: float = 0.5

        
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

            # --- Datos base ---
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

        return X_windows, y_windows, trial_local
    
    def _stack_3d(self, X_list: List[np.ndarray],
                y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]: #Apilar para DL
        """
        Apila tensores 3D ignorando entradas inválidas.
        
        X_list: [ (ni, C, T), ... ]
        y_list: [ (ni,), ... ]

        Devuelve:
            X_total: (N_total, C, T)
            y_total: (N_total,)
        """

        if len(X_list) != len(y_list):
            raise ValueError("X_list y y_list deben tener el mismo largo")

        if len(X_list) == 0:
            raise ValueError("X_list está vacío")

        # shape de referencia
        ref_shape = X_list[0].shape[1:]  # (C, T)

        X_clean = []
        y_clean = []

        for i, (X, y) in enumerate(zip(X_list, y_list)):

            # Validar dimensión
            if X.ndim != 3:
                print(f"⚠️  Aviso: Se descarta X_list[{i}] con shape {X.shape} (no es 3D)")
                continue

            # Validar canales/tiempos
            if X.shape[1:] != ref_shape:
                print(
                    f"⚠️  Aviso: Se descarta X_list[{i}] con shape {X.shape[1:]}, "
                    f"se esperaba {ref_shape}"
                )
                continue

            # Si pasó la validación, se agrega
            X_clean.append(X)
            y_clean.append(y)

        if len(X_clean) == 0:
            raise RuntimeError("❌ Ningún archivo válido quedó después del filtrado.")

        # Apilar
        X_total = np.concatenate(X_clean, axis=0)
        y_total = np.concatenate(y_clean, axis=0)

        print(f"✔️ stack_3d: Datos válidos: {X_total.shape[0]} ventanas.")
        return X_total, y_total




      