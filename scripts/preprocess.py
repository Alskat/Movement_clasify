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

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

class Preprocessor:

    def __init__(self, use_notch=True, lowcut=8.0, highcut=30.0, notch_freq=50.0,
                 current_Fs = 160, current_duration = 1.0, model_path=None,
                 aux_path = None, use_bin = False):
    
        self.use_notch = use_notch

        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.use_notch = use_notch
        self.use_bin = use_bin

        #Metadatos que necesitamos para el modelo 

        self.target_Fs = 160
        self.target_duration= 2.0
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

        self.ea_applied = bool(params["ea_applied"][0]) if "ea_applied" in params else False
        self.R_invsqrt = params["R_invsqrt"] if self.ea_applied else None

        if self.ea_applied:
            print(f"[Preprocessor] EA activo — R_invsqrt shape={self.R_invsqrt.shape}")

        #Hacer lo mismo con el auxiliar
        if use_bin:

            if aux_path is None:
                 raise ValueError( "Si use_bin=True, debes proporcionar aux_path.")


            aux_model_path = Path(aux_path)
            aux_params_path = aux_model_path.with_suffix(".npz")

            self.aux_model = load_model(aux_model_path)

            aux_params = np.load(aux_params_path)

            self.aux_mean = aux_params["mean"]
            self.aux_std = aux_params["std"]
            self.aux_class_names = list(aux_params["class_names"])

            self.aux_ea_applied = bool(aux_params["ea_applied"][0]) if "ea_applied" in aux_params else False
            self.aux_R_invsqrt = aux_params["R_invsqrt"] if self.aux_ea_applied else None

            self.class_names = [ self.class_names[0],*self.aux_class_names]
            print(f"Clases finales: {self.class_names}")
                

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

        #4) Euclidean Alignment + Normalización + Reshape
        if not self.use_bin:
            if self.ea_applied:
                window = self.R_invsqrt @ window

            window = self.normalize_window(
                window,
                mean=self.mean,
                std=self.std
            )

            window = window[np.newaxis, :, :, np.newaxis]

        #5) Aplicamos el modelo para obtener la predicción

        predicted_class, predicted_name, confidence, prediction = self.predict(window)

        

        


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

    def predict(self, window):
        if not self.use_bin:
            prediction = self.model.predict(window, verbose=0)  

            predicted_class = np.argmax(
            prediction,
            axis=1
            )[0]

            predicted_name = self.class_names[predicted_class]

            confidence = np.max(prediction)

         
        
        else:
            #Hay que aplicar varios modelos, por lo cual debemos hacer unos pasos extras

            window_bin = window.copy()
            if self.ea_applied:
                window_bin = self.R_invsqrt @ window_bin

            window_bin = self.normalize_window(window_bin,mean=self.mean,std=self.std)

            window_bin = window_bin[np.newaxis, :, :, np.newaxis]


            prediction_bin = self.model.predict(window_bin, verbose=0)
            predicted_class_bin = np.argmax(
            prediction_bin,
            axis=1
            )[0]

            if predicted_class_bin == 0:
                predicted_class = predicted_class_bin
                prediction = prediction_bin
                predicted_name = self.class_names[predicted_class_bin]

                confidence = np.max(prediction_bin)
            else:
                #Debemos hacer el mismo proceso con la ventana ahora con el modelo auxiliar

                window_aux = window.copy()
                if self.aux_ea_applied:
                    window_aux = self.aux_R_invsqrt @ window_aux

                window_aux = self.normalize_window(window_aux,mean=self.aux_mean,std=self.aux_std)

                window_aux = window_aux[np.newaxis, :, :, np.newaxis]
                prediction = self.aux_model.predict(window_aux, verbose=0)  

                aux_predicted_class = np.argmax(
                prediction,
                axis=1
                )[0]
                predicted_class = aux_predicted_class + 1

                predicted_name = self.class_names[predicted_class]

                confidence = np.max(prediction)
        return predicted_class, predicted_name, confidence, prediction
        

def resumir_ventanas(
    y_true,
    y_pred,
    class_names,
    total_windows=0
):
    """
    Imprime métricas solo sobre ventanas estables.
    Las ventanas de transición se contabilizan aparte,
    porque contienen muestras de más de un evento.
    """

    if len(y_true) == 0:
        print("\n[SUMMARY] No existen ventanas estables para evaluar.")
        print(f"[SUMMARY] Ventanas totales: {total_windows}")
        
        return

    labels = list(class_names)

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    stable_windows = len(y_true)

    print("\n" + "=" * 65)
    print("RESUMEN DE VALIDACIÓN ONLINE - VENTANAS ESTABLES")
    print("=" * 65)

    print(f"Ventanas totales       : {total_windows}")
    print(f"Ventanas estables      : {stable_windows}")
    

    print(f"\nAccuracy estable       : {accuracy:.4f}")
    print(f"Macro F1 estable       : {macro_f1:.4f}")

    print("\nAciertos por clase:")

    for idx, class_name in enumerate(labels):

        total_class = int(cm[idx, :].sum())
        correct_class = int(cm[idx, idx])

        if total_class == 0:
            print(f"  {class_name:>8}: sin ventanas evaluadas")
            continue

        class_accuracy = correct_class / total_class

        print(
            f"  {class_name:>8}: "
            f"{correct_class}/{total_class} "
            f"({class_accuracy * 100:.1f}%)"
        )

    print("\nMatriz de confusión:")
    print("Filas = real | Columnas = predicción")
    print(f"Orden: {labels}")
    print(cm)

    print("\nReporte por clase:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0
        )
    )

    print("=" * 65)