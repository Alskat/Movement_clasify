import json
import socket
import numpy as np
import time

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

buffer = []

buffer_count = 1

channel_names = None

last_packet_time = time.time()

last_idx = -1

current_event = "NONE"

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
    
    elif packet["type"] == "EVENT":
        current_event = packet["event"]
        print(
            f"\nSe ha detectado un nuevo evento: "
            f"{current_event} | "
            f"sample_idx={packet['sample_idx']} | "
            f"timestamp={packet['timestamp']}"
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


        if len(buffer) == WINDOW_SAMPLES: #Se llena la ventana de 160 muestras

            window = np.stack(
                buffer,
                axis=1
            )

            window = window[np.newaxis, :, :]

            

            print(
                f"[WINDOW] "
                f"{buffer_count} / {duration*2} | "
                f"shape={window.shape}"
            )
            buffer_count += 1
            buffer = buffer[half:] #Deslizamiento del 50%

print("\n[RECEIVER] Receiver finalizado.")