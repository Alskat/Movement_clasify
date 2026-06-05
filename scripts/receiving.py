import json
import socket
import numpy as np
import time
from preprocess import Preprocessor, resumir_ventanas

import argparse

parser = argparse.ArgumentParser(
    description="Receiver pseudo-online para clasificación EEG con EEGNet."
)



parser.add_argument( #Modelo principal el cual puede ser el triclase o binario
    "--model",
    dest="model_path",
    type=str,
    required=True,
    help="Ruta al modelo principal .keras."
)

parser.add_argument( #Modelo auxiliar el cual es solamente binario sin rest
    "--aux-model",
    dest="aux_model_path",
    type=str,
    default=None,
    help="Ruta al modelo auxiliar .keras cuando se usa jerarquía binaria."
)

parser.add_argument( #Permite activar la jerarquía binaria
    "--use-bin",
    dest="use_bin",
    action="store_true",
    help="Activa la clasificación mediante jerarquía binaria."
)

slide_group = parser.add_mutually_exclusive_group()

slide_group.add_argument(
    "--slide",
    dest="slide",
    action="store_true",
    help="Activa ventana deslizante con hop de 0.5 segundos."
)

slide_group.add_argument(
    "--no-slide",
    dest="slide",
    action="store_false",
    help="Desactiva ventana deslizante; usa ventanas consecutivas de 1 segundo."
)

parser.set_defaults(slide=True)

args = parser.parse_args()

model_path = args.model_path
aux_model_path = args.aux_model_path
use_bin = args.use_bin
slide = args.slide


# Validación mínima para jerarquía binaria
if use_bin and aux_model_path is None:
    parser.error(
        "Si activas --use-bin, debes proporcionar también --aux-model."
    )


print(f"[RECEIVER] Modelo principal: {model_path}")

if use_bin:
    print("[RECEIVER] Modo: jerarquía binaria")
    print(f"[RECEIVER] Modelo auxiliar: {aux_model_path}")
else:
    print("[RECEIVER] Modo: clasificación triclase")

print(
    "[RECEIVER] Slide/hop: "
    + ("activo, hop de 0.5 s" if slide else "inactivo, hop de 1.0 s")
)

#Variables fijas
HOST = "127.0.0.1"

STREAM_PORT = 5005
CONTROL_PORT = 5006

WINDOW_SAMPLES = 160

sock_stream = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

sock_stream.bind((HOST, STREAM_PORT))

sock_stream.settimeout(1.0)

sock_control = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

print("[RECEIVER] Enviando READY...")

sock_control.sendto(
    "READY".encode("utf-8"),
    (HOST, CONTROL_PORT)
)

print("[RECEIVER] Esperando stream...\n")

#Variables dinámicas

buffer = []

buffer_count = 1

channel_names = None

last_packet_time = time.time()

last_idx = -1


y_true_stable = []
y_pred_stable = []


total_windows = 0

event_buffer = []

try:
    while True:

        try:

            data, addr = sock_stream.recvfrom(65535)

            last_packet_time = time.time()

        except socket.timeout:

            elapsed = (
                time.time() - last_packet_time
            )

            

            if elapsed > 10:
                print(
                    "[ERROR] "
                    "No llegaron paquetes."
                )
                break

            continue

        packet = json.loads(
        
            data.decode("utf-8")
        )

        if packet.get("type") == "END":

            print(
                "\n[CONTROL] "
                "END recibido."
            )

            break

        if packet["type"] == "CONFIG":

            print("Se recibe una nueva señal:")
            channel_names = packet["channels"]

            print(
                f"[INFO] "
                f"Canales: {len(channel_names)}"
            )

            print(channel_names)
            fs = packet["fs"]
            duration = packet["duration"]
            
            half = int(fs/2)

            preprocessor = Preprocessor(
                current_Fs = fs,
                current_duration = 1.0,
                model_path = model_path,
                aux_path = aux_model_path if use_bin else None,
                use_bin = use_bin
            )
        
        elif packet["type"] == "EVENT":
            current_event = packet["event"]
            print(
                f"\nSe ha detectado un nuevo evento: "
                f"{current_event} | "
                f"sample_idx={packet['sample_idx']}  "
            )
            

        elif packet["type"] == "SAMPLE":
            sample = np.array(
                packet["sample"],
                dtype=float
            )


            if packet["idx"] != last_idx + 1:

                print(
                    f"[WARNING] "
                    f"Packet loss: "
                    f"{last_idx} -> {packet['idx']}"
                )

            last_idx = packet["idx"]

            latency = (
                time.time() - packet["timestamp"]
            ) * 1000

            buffer.append(sample)
            event_buffer.append(current_event)


            if len(buffer) == WINDOW_SAMPLES: #Se llena la ventana de 160 muestras

                total_windows += 1

                window = np.stack( #Creamos nuestra ventana 
                    buffer,
                    axis=1
                )

                
                if slide: 
                    print(
                        f"[WINDOW] "
                        f"{buffer_count} / {duration*2} | "
                        f"shape={window.shape}"
                    )
                else: 
                    print(
                        f"[WINDOW] "
                        f"{buffer_count} / {duration} | "
                        f"shape={window.shape}"
                    )

                #preprocesamos la ventana

                result = preprocessor.preprocess_window(window)

                predicted_name = result["predicted_name"]
                true_event = event_buffer[0]

                labels_in_window = set(event_buffer)

                y_true_stable.append(true_event)
                y_pred_stable.append(predicted_name)

                #print(f"Dimensiones de la ventana preprocesada: {result['window'].shape}")
                print(f"Predicción: clase={result['predicted_name']} | confidence={result['confidence']:.4f} |{result['processing_time_ms']:.2f} ms ")
                if result['predicted_name'] == current_event:
                    print(f"✅Predicción coincide con el evento actual")
                else:
                    print(f"❌Predicción NO coincide con el evento actual")

        
                
                buffer_count += 1
                if slide:
                    buffer = buffer[half:] #Deslizamiento del 50%
                    event_buffer = event_buffer[half:]
                else:
                    buffer = [] #Deslizamiento del 50%
                    event_buffer = []
except KeyboardInterrupt:
    print("\n[RECEIVER] Detenido por el usuario.")

finally:

    sock_control.sendto(
        "STOP".encode("utf-8"),
        (HOST, CONTROL_PORT)
    )



    resumir_ventanas(
        y_true_stable,
        y_pred_stable,
        class_names=preprocessor.class_names,
        total_windows=total_windows
    )

    sock_stream.close()
    sock_control.close()

    print("[RECEIVER] Receiver finalizado.")

