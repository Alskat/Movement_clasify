### Dentro de esta función vamos a recibir una ventana de N muestras y preprocesarlo, normalizarlo y estandarizalo listo para modelo keras
### Es decir, pasar de una ventana del tipo (C, N) a una ventana del tipo (1, C, N 1)
import numpy as np
from scipy.signal import (
    firwin,
    iirnotch,
    filtfilt,
    resample
)
import warnings
from time import perf_counter

from tensorflow.keras.models import load_model
from pathlib import Path

class Preprocessor:

    def __init__(self, use_notch=True, lowcut=8.0, highcut=30.0, notch_freq=50.0,
                 current_Fs = 160, current_duration = 1.0, model_path=None):
    
        self.use_notch = use_notch

        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.use_notch = use_notch

        #Metadatos que necesitamos para el modelo 

        self.target_Fs = 160
        self.target_duration= 1.0
        self.target_samples = int(self.target_Fs * self.target_duration)
        self.current_Fs = current_Fs
        self.current_duration = current_duration

        #Diseñamos los filtros de una vez para ejecutarlos sólo una vez 

        self.fir_coeff = firwin(
            31,
            [8.0, 30.0],
            pass_zero=False,
            fs=self.current_Fs
        )
        if self.use_notch:
            self.notch_b, self.notch_a = iirnotch(
                self.notch_freq,
                Q=30,
                fs=self.current_Fs
            )
        #Cargar modelos para la predicción


        model_path = Path(model_path)
        params_path = model_path.with_suffix(".npz")

        self.model = load_model(model_path)

        params = np.load(params_path)

        self.mean = params["mean"]
        self.std = params["std"]
        self.class_names = list(params["class_names"])
                

    def preprocess_window(self, window):
        #Acá vamos a aplicar todo el pipeline de procesamiento del buffer que recibimos 

        #Empezamos a contar el tiempo
        init_time = perf_counter()

        #0) Validamos que la ventana tenga la forma correcta (C, N) y que tenga la cantidad de muestras correcta}
        if window.ndim != 2:
            raise ValueError(f"Se esperaba una ventana de 2 dimensiones (C, N), pero se recibió {window.ndim} dimensiones. No se aplicará ningún preprocesamiento.")
            #Nos salimos de la función
            
        
        if self.current_Fs != self.target_Fs:
            warnings.warn(f"Se recibió una frecuencia de {self.current_Fs}Hz. Se resampleará a {self.target_Fs}Hz.")
        if self.current_duration != self.target_duration:
            raise ValueError(f"Se recibió una duración de {self.current_duration}s. No se aplicará ningún preprocesamiento.")   
            
        #1) Filtramos la ventana con bandpass y Notch

        window = self.filt_window(window)

        #2) Aplicamos CAR

        window = self.car_window(window)

        #3) Se resamplea la ventana a la frecuencia objetivo (160Hz)

        if self.current_Fs != self.target_Fs:
            window = resample(
                window,
                self.target_samples,
                axis=1
            )

        #4) Normalizamos la señal 

        window = self.normalize_window(
            window,
            mean=self.mean,
            std=self.std
        )

        #5) Reshapeamos a la forma que espersa Keras: 

        window = window[np.newaxis, :, :, np.newaxis]

        #6) Aplicamos el modelo para obtener la predicción

        prediction = self.model.predict(window, verbose=0)

        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]
        predicted_name = self.class_names[predicted_class]

        confidence = np.max(prediction)


        end_time = perf_counter()
        processing_time = (end_time - init_time) * 1000

        return {
                "window": window,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "raw_prediction": prediction,
                "predicted_name": predicted_name,
                "processing_time_ms": processing_time
            }
        


    def filt_window(self, window):

        #Aplicamos el filtro FIR a cada canal 
        filtered_window = filtfilt(
            self.fir_coeff,
            [1.0],
            window,
            axis=1
        )
        if self.use_notch:
            filtered_window = filtfilt(
                self.notch_b,
                self.notch_a,
                filtered_window,
                axis=1
            )

        return filtered_window
    
    def car_window(self, window):

        mean_ref = np.mean(
            window,
            axis=0,
            keepdims=True
        )

        window = window - mean_ref

        return window
    
    def normalize_window(self, window, mean=None, std=None, eps=1e-6):

        if mean is None or std is None:
            warnings.warn("No se proporcionaron valores de media y desviación estándar para la normalización. Se calcularán a partir de la ventana actual, lo que puede introducir data leakage. Se recomienda proporcionar valores pre-calculados.")
            mean = np.mean(window, axis=1, keepdims=True)
            std = np.std(window, axis=1, keepdims=True)
            
        mean = np.asarray(mean).reshape(-1, 1)
        std = np.asarray(std).reshape(-1, 1)

        if mean.shape[0] != window.shape[0]:
            raise ValueError(
                f"Mean/std corresponden a {mean.shape[0]} canales, "
                f"pero la ventana tiene {window.shape[0]}."
            )
        normalized_window = (window - mean) / (std + eps)

        return normalized_window


