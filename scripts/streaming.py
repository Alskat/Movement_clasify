import time
import json
import socket
import numpy as np
import mne

#Cargamos todo lo relacionaod a la señal
EDF_PATH = "dataset_physionet/files/S007/S007R03.edf"
type = 1 
if type == 1:
    event_map = {
    "T0": "rest",
    "T1": "left",
    "T2": "right"
}
else:
    event_map = {
    "T0": "rest",
    "T1": "hands",
    "T2": "feet"
}
    

#Todo lo relacionado al stream 

HOST = "127.0.0.1"

STREAM_PORT = 5005
CONTROL_PORT = 5006

sock_stream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock_control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_control.bind((HOST, CONTROL_PORT))

print("[STREAMER] Esperando READY del receiver...")

data, addr = sock_control.recvfrom(1024)

msg = data.decode("utf-8")

if msg != "READY":
    print("[STREAMER] Mensaje inválido.")
    exit()

# Después del handshake, el socket de control no debe bloquear el streaming.
# Solo revisaremos si ha llegado una orden STOP.
sock_control.setblocking(False)

print("[STREAMER] READY recibido.")
print("[STREAMER] Iniciando streaming...\n")

#Cargamos la señal a enviar

raw = mne.io.read_raw_edf(
    EDF_PATH,
    preload=True,
    verbose=False
)
#Seleccionamos canales de interés

channels = ["C3..", "C4..", "Cz.."] #Esto se va a poder seleccionar en el futuro en la GUI
available_channels = raw.ch_names #Validamos que existan en el archivo

for channel in channels:
    if channel not in available_channels:
        raise ValueError(f"Canal {channel} no encontrado en los datos.")
raw.pick_channels(channels)

events, event_id = mne.events_from_annotations(raw) 
#Events es un array de shape (n_events, 3) donde cada fila es [sample, 0, event_id], nos importan los samples 

reverse_event_map = {v: k for k, v in event_id.items()}


fs = int(raw.info["sfreq"])

data = raw.get_data()

ch_names = raw.ch_names

n_channels, n_samples = data.shape

duration = n_samples / fs

print(
    f"[STREAMER] "
    f"{n_channels} canales | "
    f"{n_samples} muestras | "
    f"{duration:.2f}s"
)

t0 = time.perf_counter()
send_data=1
event_pointer = 0

stopped_by_user = False

for i in range(n_samples):


    # Revisar si el receiver pidió detener el streaming
    try:
        control_data, control_addr = sock_control.recvfrom(1024)
        control_msg = control_data.decode("utf-8")

        if control_msg == "STOP":
            print("\n[STREAMER] STOP recibido. Streaming detenido por el receiver.")
            stopped_by_user = True
            break

    except BlockingIOError:
        # No llegó ningún mensaje de control; continuamos transmitiendo.
        pass

    sample = data[:, i].astype(float)

    #1) Creamos el paquete con los metadatos 

    if send_data:
        packet_data = {
            "type": "CONFIG",
            "fs": fs,
            "channels": ch_names,
            "duration": duration
        }

        msg = json.dumps(packet_data).encode("utf-8")
        sock_stream.sendto(
            msg,
            (HOST, STREAM_PORT)
        )
        send_data =0

    #2) Se envía un paquete con el evento si corresponde

    if (event_pointer < len(events) and i == events[event_pointer][0]): #Observa si la muestra actual corresponde a un evento

        event_code = events[event_pointer][2]

        event_name = reverse_event_map[event_code]
        semantic_event = event_map[event_name]

        event_packet = {
            "type": "EVENT",
            "event": semantic_event,
            "sample_idx": i,
            "timestamp": time.time()
        }



        msg = json.dumps(event_packet).encode("utf-8")
        sock_stream.sendto(
            msg,
            (HOST, STREAM_PORT)
        )

        event_pointer += 1

    #3) Se envía el paquete con las muestras
    packet = {
        "type": "SAMPLE",
        "idx": i,
        "sample": sample.tolist(),
        "timestamp": time.time(),
    }

    msg = json.dumps(packet).encode("utf-8")

    sock_stream.sendto(
        msg,
        (HOST, STREAM_PORT)
    )


    #4 Se espera el paquete de control para sincronizar el streaming
    next_time = t0 + (i + 1) / fs

    sleep_time = (
        next_time - time.perf_counter()
    )

    if sleep_time > 0:
        time.sleep(sleep_time)

end_packet = {
    "type": "END"
}

sock_stream.sendto(
    json.dumps(end_packet).encode("utf-8"),
    (HOST, STREAM_PORT)
)

if stopped_by_user:
    print("[STREAMER] Stream cerrado manualmente.")
else:
    print("[STREAMER] Streaming terminado: fin del EDF.")