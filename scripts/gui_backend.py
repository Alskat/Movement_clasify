
# ============================================
# EEG: GFP y STFT
# ============================================
from typing import Optional, Tuple, Union
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, butter, sosfiltfilt
from mne.decoding import CSP
from time import perf_counter
from pathlib import Path
from typing import List
from sklearn.preprocessing import StandardScaler
import re #Para buscar el número final
import glob #Para buscar archivos con tipos específicos



ArrayLike = Union[np.ndarray, mne.io.BaseRaw, mne.Epochs]

class EEG: #Esta clase se usa para la visialización de features, está desactualizado y ya no se usa
    """
    Utilidades de visualización para ventanas Raw (~1s) o Epochs:
      - GFP: Global Field Power (desvío estándar entre canales a lo largo del tiempo)
      - STFT: Espectrograma (promedio sobre canales o por canal)
    """
    def __init__(self, fmax: float = 40.0):
        self.fmax = float(fmax)

    # -------- Helpers --------
    @staticmethod
    def _as_data_and_times(x: ArrayLike) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Devuelve (data, times, sfreq)
          - Raw: data shape (n_ch, n_samp), times shape (n_samp,)
          - Epochs: toma la PRIMERA época para visualizar (n_ch, n_samp)
        """
        if isinstance(x, mne.io.BaseRaw):
            data = x.get_data()
            times = x.times
            sf = x.info['sfreq']
        elif isinstance(x, mne.Epochs):
            data = x.get_data()[0]       # 1ra época
            times = x.times
            sf = x.info['sfreq']
        else:
            raise TypeError("x debe ser mne.io.Raw o mne.Epochs")
        return data, times, float(sf)

    # -------- GFP --------
    @staticmethod
    def gfp(epochs, cond, picks=None, ci="sem", mode="epochs"):
        """
        GFP por condición con dos modos:
        - mode='epochs': heatmap ÉPOCAS×tiempo + promedio (GFP clásico).
        - mode='channels': heatmap CANALES×tiempo (no es GFP; muestra PSD/var por canal).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if cond not in epochs.event_id:
            raise ValueError(f"Condición {cond} no está en epochs: {list(epochs.event_id.keys())}")

        eps = epochs[cond]
        if isinstance(picks, (list, tuple)) and picks and isinstance(picks[0], str):
            data = eps.get_data(picks=picks)   # (n_ep, n_ch, n_t)
            ch_names = picks
        else:
            data = eps.get_data(picks=picks)   # (n_ep, n_ch, n_t)
            ch_names = [epochs.ch_names[i] for i in (picks if picks is not None else range(data.shape[1]))]

        times = eps.times
        n_ep, n_ch, n_t = data.shape

        if mode.lower() == "epochs":
            # GFP por época = sd espacial en cada t
            gfp_ep = data.std(axis=1)                  # (n_ep, n_t)
            mean_gfp = gfp_ep.mean(axis=0)             # (n_t,)
            spread = (gfp_ep.std(axis=0, ddof=1) / np.sqrt(n_ep)) if ci == "sem" else gfp_ep.std(axis=0, ddof=1)
            lower, upper = mean_gfp - spread, mean_gfp + spread

            fig = plt.figure(figsize=(8,5))
            gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.25)

            # Heatmap épocas×tiempo
            ax0 = fig.add_subplot(gs[0])
            extent = [times[0], times[-1], 0, n_ep]
            im = ax0.imshow(gfp_ep, aspect="auto", origin="lower", extent=extent)
            ax0.axvline(0, linestyle="--", linewidth=1)
            ax0.set_title(f"EEG (GFP) – {cond}")
            ax0.set_ylabel("Épocas")
            cbar = fig.colorbar(im, ax=ax0); cbar.set_label("µV")

            # Promedio + banda
            ax1 = fig.add_subplot(gs[1], sharex=ax0)
            ax1.plot(times, mean_gfp, linewidth=1.5)
            ax1.fill_between(times, lower, upper, alpha=0.3)
            ax1.axvline(0, linestyle="--", linewidth=1)
            ax1.set_xlabel("Tiempo (s)"); ax1.set_ylabel("µV")
            plt.show()

            return fig  # << no plt.show() aquí

        elif mode.lower() == "channels":
            # Heatmap canal×tiempo (zscore por canal para comparabilidad visual)
            X = data.mean(axis=0)                         # (n_ch, n_t) promedio sobre épocas
            X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            extent = [times[0], times[-1], 0, n_ch]
            im = ax.imshow(X, aspect="auto", origin="lower", extent=extent, cmap="viridis")
            ax.axvline(0, linestyle="--", linewidth=1, color='w', alpha=0.7)
            ax.set_title(f"EEG (canales z-score) – {cond}")
            ax.set_xlabel("Tiempo (s)")
            ax.set_yticks(np.arange(0.5, n_ch + 0.5, 1))
            ax.set_yticklabels(ch_names)                  # << nombres de canal en Y
            cbar = fig.colorbar(im, ax=ax); cbar.set_label("z")
            fig.tight_layout()
            plt.show()
            return fig

        else:
            raise ValueError("Debe ser 'epochs' o 'channels'")

    # -------- STFT --------
    @staticmethod
    def stft(
        epochs,
        cond: str,
        canal: str,
        TMIN: float = -0.5,
        win_sec: float = 0.5,        # duración de ventana (s)
        overlap: float = 0.90,       # 90% solapamiento
        pad_factor: int = 4,         # zero-padding => nfft = pad_factor * nperseg (≥2)
        fmin: float = 1.0,
        fmax: float = 40.0,
        use_imshow: bool = True,     # True => píxel nítido
        dpi: int = 150,
        vmin_pct: float = 5,         # percentiles para rango dinámico
        vmax_pct: float = 95,
    ):
        eps = epochs[cond]
        sf = eps.info["sfreq"]

        # (n_ep, n_t) del canal
        X = eps.get_data(picks=[canal])[:, 0, :]
        x = X.mean(axis=0)  # promedio sobre épocas (o elige una época si quieres)

        # Parámetros STFT
        nperseg  = max(16, int(win_sec * sf))
        noverlap = int(overlap * nperseg)
        nfft     = int(2 ** np.ceil(np.log2(max(nperseg, pad_factor * nperseg))))  # potencia de 2

        # STFT (spectrogram ya devuelve densidad PSD)
        f, t, Sxx = spectrogram(
            x, fs=sf, window="hann",
            nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            detrend=False, scaling="density", mode="psd"
        )

        # Recorte de banda y dB
        m = (f >= fmin) & (f <= fmax)
        f, Sxx = f[m], Sxx[m]
        Sxx_dB = 10*np.log10(Sxx + 1e-20)

        # Contraste robusto (evita “lavado”)
        vmin = np.percentile(Sxx_dB, vmin_pct)
        vmax = np.percentile(Sxx_dB, vmax_pct)

        # Tiempo relativo a época
        t_plot = t + TMIN

        # Plot nítido
        plt.figure(figsize=(8, 3.6), dpi=dpi)
        if use_imshow:
            # pixeles nítidos
            extent = [t_plot[0], t_plot[-1], f[0], f[-1]]
            plt.imshow(Sxx_dB, origin="lower", aspect="auto", extent=extent,
                    interpolation="nearest", vmin=vmin, vmax=vmax)
        else:
            # suavizado (si lo prefieres)
            plt.pcolormesh(t_plot, f, Sxx_dB, shading="gouraud", vmin=vmin, vmax=vmax)

        plt.axvline(0, ls="--", lw=1, color="k")
        plt.title(f"Espectrograma (STFT) – {canal} – {cond}")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Frecuencia (Hz)")
        cbar = plt.colorbar(); cbar.set_label("PSD (dB)")
        plt.tight_layout()
        plt.show()

       
    def welch_raw(self, raw_1s: mne.io.BaseRaw,
                  bands: Tuple[Tuple[str, float, float], ...] = (('alpha',8,12),('beta',14,26)),
                  plot: bool = False, show: bool = True) -> Tuple[np.ndarray, float]:
        """Welch sobre Raw (~1 s). Retorna (vector_features, ms). Plot opcional de barras por banda."""
        sf = raw_1s.info['sfreq']; X = raw_1s.get_data()
        t0 = perf_counter()
        freqs, psd = welch(X, fs=sf, nperseg=X.shape[-1], axis=-1)
        feat, per_band = [], []
        for name, f1, f2 in bands:
            m = (freqs >= f1) & (freqs <= f2)
            bp = np.trapz(psd[..., m], freqs[m], axis=-1)  # (n_ch,)
            feat.extend(bp.tolist())
            per_band.append((name, bp))
        ms = (perf_counter() - t0) * 1000.0

        if plot:
            fig, ax = plt.subplots(1,1, figsize=(7,3))
            idx = np.arange(X.shape[0])
            width = 0.8 / max(1, len(per_band))
            for i, (name, bp) in enumerate(per_band):
                ax.bar(idx + i*width, bp, width=width, label=name)
            ax.set_xticks(idx + width*(len(per_band)-1)/2)
            ax.set_xticklabels(raw_1s.ch_names, rotation=90)
            ax.set_ylabel("Bandpower (uV²)")
            ax.set_title("Welch bandpower por canal")
            ax.legend(); plt.tight_layout()
            if show: plt.show() 
            else: plt.close(fig)

        return np.asarray(feat, dtype=float), ms

    
    def welch_epochs(self, epochs: mne.Epochs,
                     bands: Tuple[Tuple[str, float, float], ...] = (('alpha',8,12),('beta',14,26)),
                     plot: bool = False, cond: Optional[str] = None, show: bool = True) -> np.ndarray:
        """Welch por época. Retorna matriz (n_epochs, n_features). Plot opcional promedio por canal."""
        eps = epochs if cond is None else epochs[cond]
        sf = eps.info['sfreq']; X = eps.get_data()  # (n_ep, n_ch, n_samp)
        feats, per_band_mean = [], []
        for ep in X:
            freqs, psd = welch(ep, fs=sf, nperseg=ep.shape[-1], axis=-1)
            vec = []
            for name, f1, f2 in bands:
                m = (freqs >= f1) & (freqs <= f2)
                bp = np.trapz(psd[..., m], freqs[m], axis=-1)
                vec.extend(bp.tolist())
            feats.append(np.asarray(vec, float))
        F = np.vstack(feats)

        if plot:
            # promedio por canal y banda sobre épocas
            per_band_mean = []
            freqs, psd_all = welch(X.mean(axis=0), fs=sf, nperseg=X.shape[-1], axis=-1)
            fig, ax = plt.subplots(1,1, figsize=(7,3))
            idx = np.arange(X.shape[1]); width = 0.8 / max(1, len(bands))
            for i,(name,f1,f2) in enumerate(bands):
                m = (freqs >= f1) & (freqs <= f2)
                bp = np.trapz(psd_all[..., m], freqs[m], axis=-1)
                ax.bar(idx + i*width, bp, width=width, label=name)
            ax.set_xticks(idx + width*(len(bands)-1)/2)
            ax.set_xticklabels(eps.ch_names, rotation=90)
            ax.set_ylabel("Bandpower (uV²)"); ax.set_title("Welch (promedio épocas)")
            ax.legend(); plt.tight_layout()
            if show: plt.show()
            else: plt.close(fig)

        return F


def preprocess( #Preprocesamiento: filtrado, notch y CAR
    raw_1s,
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
    t0 = perf_counter()

    #Filtro pasabanda 
    raw_clean.filter(l_freq=l_freq, h_freq=h_freq, method=filter_method, verbose='ERROR')

    # notch
    if use_notch and notch_freqs:
        raw_clean.notch_filter(freqs=notch_freqs, verbose='ERROR')

    # referencia
    raw_clean.set_eeg_reference(ref, verbose='ERROR')

    elapsed_ms = (perf_counter() - t0) * 1000.0
    return raw_clean, elapsed_ms

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

DEFAULT_PICKS = ['Fp1.','Fp2.','F3..','F4..','C3..','C4..','O1..','O2..']
mov = ['C3..','C4..','Cz..','Fc3.','Fc4.','Cp3.','Cp4.','Pz..']

def load_and_pick(EDF_PATH: str, PICKS=DEFAULT_PICKS): #Elegir los canales que queremos utilizar



    raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose='ERROR')
    

    # Elegir canales disponibles del subconjunto deseado
    pick_avail = [ch for ch in PICKS if ch in raw.ch_names]
    if not pick_avail:
        raise ValueError(f"Ningún canal de {PICKS} está en los datos: {raw.ch_names}")
    #Ver si hay algún canal que no esté
    if len(pick_avail) < len(PICKS):
        missing = set(PICKS) - set(pick_avail)
        print(f"Advertencia: faltan canales {missing} en los datos.")

    if pick_avail:
        raw.pick(pick_avail, verbose='ERROR')

    m = re.search(r'(\d{2})(?=\.edf$)', EDF_PATH, flags=re.IGNORECASE) #Ni idea de qué hacía esto

    if m:
        n = int(m.group(1)) #
    else:
        # fallback: 1 dígito
        m = re.search(r'(\d)(?=\.edf$)', EDF_PATH, flags=re.IGNORECASE)

        n = int(m.group(1))

    # excluir 1 y 2
    if n in {1}:
        labels = {'open'} 
        mode = "OPEN"
    if n in {2}:
        labels = {'close'} 
        mode = "CLOSE"


    if n in {3,7,11}:
        labels =  ['rest','left_m','right_m']
        mode = "LR_M"

    if n in {4,8,12}:
        labels = ['rest', 'left_i', 'right_i']
        mode = "LR_I"

    if n in {6,10,14}:
        labels= ['rest', 'hands_i', 'feet_i']
        mode = "HF_I"

    if n in {5,9,13}:
        labels =  ['rest', 'hands_m', 'feet_m']
        mode = "HF_M"


    return raw, labels, mode

#----------- Nuevas funciones jocosas
#-----Vectorizar epochs a matriz 2D (n_epochs, n_features)
# 1) ---------- Vectorizar ----------
def vectorizar(data3d: np.ndarray) -> np.ndarray: #Vectorizar para ML
    """
    Convierte (n_epochs, n_channels, n_times) -> (n_epochs, n_channels*n_times).
    No cambia orden; solo aplana por época.
    """
    if data3d.ndim != 3:
        raise ValueError(f"Esperaba 3D, llegó {data3d.shape}")
    n_e, n_c, n_t = data3d.shape
    return data3d.reshape(n_e, n_c * n_t)

#2)------- Apilar tanto los features como el target-----


def stack(X_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]: #Apilar distintas señales en un único tensor
    """
    Recibe múltiples matrices 2D de features y vectores 1D de targets,
    y los apila en una sola matriz X y vector y.
    
    X_list: [ (n1, d), (n2, d), ... ]
    y_list: [ (n1,),  (n2,),  ... ]

    return: X_total, y_total
    """
    if len(X_list) != len(y_list):
        raise ValueError("X_list y y_list deben tener el mismo largo")

    # Validamos dimensiones
    for X in X_list:
        if X.ndim != 2:
            raise ValueError(f"Cada X debe ser 2D, llegó {X.shape}")

    X_total = np.vstack(X_list)
    y_total = np.concatenate(y_list)

    return X_total, y_total


# 3) ---------- Normalizar ----------
def normalizar(X: np.ndarray) -> Tuple[StandardScaler, np.ndarray]: #Normalización para ML

    """
    Ajusta StandardScaler y normaliza X.
    devueve scaler y X_norm
    """
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return scaler, X_norm


"----------------Funciones para Deep Learning------------------"

def ventana( #Aplicar muchas ventanas de tiempo y solape determinado
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

    for ep_idx in range(n_epochs):
        ep_data = X[ep_idx]         # (n_channels, n_times)
        label  = y_epoch[ep_idx]

        # recorre inicio en muestras: 0, step, 2*step, ...
        for start in range(0, n_times - win_samp + 1, step_samp):
            end = start + win_samp
            window = ep_data[:, start:end]  # (n_channels, win_samp)

            X_windows.append(window)
            y_windows.append(label)

    X_windows = np.stack(X_windows, axis=0)   # (N_ventanas, n_channels, win_samp)
    y_windows = np.array(y_windows)

    return X_windows, y_windows



# ----------Stackear ventanas 3d----------

def stack_3d(X_list: List[np.ndarray],
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


def plot_windows_check(X, fs, n_windows=5, start=0):
    """
    Muestra n_windows ventanas consecutivas.
    
    X: array (N, C, T) = ventanas ya generadas
    fs: frecuencia de muestreo
    n_windows: cuántas ventanas consecutivas graficar
    """

    # Seleccionamos un índice aleatorio para ver un grupo consecutivo
    
    
    plt.figure(figsize=(14, 10))

    time = np.arange(X.shape[2]) / fs  # vector de tiempo en segundos

    for i in range(n_windows):
        ax = plt.subplot(n_windows, 1, i+1)
        data = X[start + i]

        # Plot canal por canal
        offset = 0
        for ch in range(data.shape[0]):
            ax.plot(time, data[ch] + offset, linewidth=0.8)
            offset += np.ptp(data[ch]) * 1.2  # separación vertical útil

        ax.set_title(f"Ventana {start + i}  (t = {time[0]:.2f} → {time[-1]:.2f} s)")
        ax.set_ylabel("Canales")
        ax.grid(True)

    plt.xlabel("Tiempo dentro de la ventana (s)")
    plt.tight_layout()
    plt.show()


# Orden CANÓNICO de clases para TODO el proyecto DL
CLASS_NAMES = [
    "rest", #0
    "right_i", #1
    "left_i", #2
    "hands_i", #3
    "feet_i", #4
    "right_m", #5
    "left_m", #6
    "hands_m", #7
    "feet_m", #8
]

LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
