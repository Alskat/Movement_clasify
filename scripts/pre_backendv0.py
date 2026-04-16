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

    def __init__(self, data_dir: Union[str, Path], channels = None, l_freq = 8, h_freq = 30.0, use_notch = True, resample_freq = False, new_freq = 128, Debug = False, start = True): 


        #Empezamos con los los datos que se necesitan en el array

        self.X = None
        self.y = None
        self.sub = None
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.classes = None
        self.run = None #ID de la sesión, para diferenciar entre sesiones de un mismo sujeto
        self.trial = None #ID de cada muestra diferente
        self.class_names = None #diccionario para mapear las clases 
        self.channels_names = None #nombres de los canales, para mapearlos con los datos

        self.l_freq = l_freq
        self.h_freq = h_freq

        #Ahora debemos examinar la carpeta de datos, nuestro datasir es un directorio con muchas carpetas!!!! 

        subject_dirs = [os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir)]
        subject_dirs = [d for d in subject_dirs if os.path.isdir(d)] #Creamos un array con todas las carpetas

        if Debug:
            print(f"Total de directorios encontrados: {len(subject_dirs)}")
            print(subject_dirs)
         
        f_array =[] #Arreglo para guardar los archivos que sí cumplen con el formato correcto, para luego procesarlos

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
                
                if Debug:
                    print(f" Analizando evento: {n_1}")

                if n_1 in {1, 2}:
                    

                    if Debug:
                        print(f" Evento {n_1}  ignorado.")

                    continue
                    
                if not f.lower().endswith(".edf"): #Asegurarnos de ue es un .edf y no un .event (Aunque ya lo verificamos antes )
                    print(f"Se coló un archivo no .edf = {ff}. REVISAR CÓDIGO")
                    continue

                #Si llegamos aquí, es porque el archivo es un .edf y corresponde a un evento que nos interesa
                if Debug:
                    print(f"✅ Archivo {ff} válido para procesamiento.")

                f_array.append((f, n_1, subject_id)) #Guardamos la dirección del archivo, el tipo de evento y el ID del sujeto para procesarlo después

                if Debug:
                    print(f"Archivo {ff} agregado para procesamiento. Total archivos válidos hasta ahora: {len(f_array)}")

        self.array = f_array #Guardamos el array con los archivos válidos para procesarlos después

        #Ahora aplicamos el pipeline 

        if start:
            self.load(self.array, picks = self.channels, Debug = Debug) #Cargamos los datos de los archivos válidos, con los canales que nos interesan (si se especificaron)
            self.preprocess(raw_1s = self.raw_clean, l_freq = self.l_freq, h_freq = self.h_freq, use_notch = use_notch, Debug = Debug)
            self.epochs(raw_clean = self.raw_clean, wanted_labels = self.class_names, tmax=3.5, tmin=0.5, baseline=None, show=False, debug=Debug)




        #self.load(self.array, picks = self.channels, Debug = Debug) #Cargamos los datos de los archivos válidos, con los canales que nos interesan (si se especificaron)
    
    def load(self, f_array, picks = None, Debug = True):

        #Todos los paths
        paths = [f[0] for f in f_array]

        f2_array = [None] * len(f_array) #Creamos un nuevo array para guardar la información adicional de labels y mode, que se añadirá después de cargar los datos

        for path in paths:
            if Debug:
                print(f"📂 Cargando archivo: {path}")
            raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')

            if picks is not None:
                
                pick_avail = [ch for ch in picks if ch in raw.ch_names]

                if not pick_avail:
                    raise ValueError(f"Ningún canal de {picks} está en los datos: {raw.ch_names}")
                #Ver si hay algún canal que no esté
                if len(pick_avail) < len(picks):
                    missing = set(picks) - set(pick_avail)
                    print(f"Advertencia: faltan canales {missing} en los datos.")

                raw.pick(pick_avail, verbose='ERROR')
                #Si picks es None, entonces se cargan todos los canales disponibles

            #n de fila del path actual
            idx = paths.index(path)
            event_id = f_array[idx][1] #Tipo de evento, que corresponde a la clase
            subject_id = f_array[idx][2] #ID del sujeto, para diferenciar entre sujetos
            if Debug:
                print(f" Evento ID: {event_id}, Sujeto ID: {subject_id}")

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

        #Añadimos labels y mode a nuestro array
            f2_array[idx] = (path, event_id, subject_id, labels, mode)
            
        self.array = f2_array #Actualizamos el array con la información adicional de labels y mode, para usarla después en el procesamiento

        self.preprocess(raw, l_freq = self.l_freq, h_freq = self.h_freq, use_notch = True, Debug = Debug)

    def preprocess( #Preprocesamiento: filtrado, notch y CAR
        self,
        
        raw_1s: mne.io.Raw,
        l_freq: float = 0.5,
        h_freq: float = 40.0,
        notch_freqs = (50,),
        use_notch: bool = False,
        ref: str = 'average',
        filter_method: str = 'fir'  # 'fir' (MNE por defecto) o 'iir'
    ):
        """
        Preprocesa un Raw de ~1 s: band-pass [l_freq, h_freq], notch opcional y referencia promedio.
        Devuelve (raw_clean, elapsed_ms).
        """
        raw_clean = raw_1s.copy().load_data()
    
        #Filtro pasabanda 
        raw_clean.filter(l_freq=l_freq, h_freq=h_freq, method=filter_method, verbose='ERROR')

        # notch
        if use_notch and notch_freqs:
            raw_clean.notch_filter(freqs=notch_freqs, verbose='ERROR')

        # referencia
        raw_clean.set_eeg_reference(ref, verbose='ERROR')

    
        self.raw_clean = raw_clean

    

    def epochs( #Epocar
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