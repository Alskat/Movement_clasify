"""
gui.py
Interfaz mínima para la demostración pseudo-online EEG/BCI.

La GUI NO procesa EEG ni carga modelos. Solo:
  - selecciona EDF y modelos;
  - configura paradigma, jerarquía binaria y slide/hop;
  - lanza streaming.py y receiving.py mediante QProcess;
  - envía STOP por UDP al streamer.

Contrato esperado de los scripts:
  streaming.py:
    --edf <ruta.edf> --type {1,2}
    

  receiving.py:
    --model <ruta.keras> [--aux-model <ruta.keras>] [--use-bin]
    [--slide | --no-slide]
"""

import socket
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QProcess, QTimer
from PyQt5.QtGui import QCloseEvent, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


HOST = "127.0.0.1"
CONTROL_PORT = 5006

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STREAMING_SCRIPT = SCRIPT_DIR / "streaming.py"
RECEIVING_SCRIPT = SCRIPT_DIR / "receiving.py"


class EEGDemoWindow(QMainWindow):
    """Panel de control de la demo; el pipeline continúa viviendo en consola."""

    def __init__(self) -> None:
        super().__init__()

        self.edf_path: Optional[Path] = None
        self.model_path: Optional[Path] = None
        self.aux_model_path: Optional[Path] = None

        self.streamer_process = QProcess(self)
        self.receiver_process = QProcess(self)
        self._running = False
        self._stop_requested = False

        self._configure_processes()
        self._build_ui()
        self._connect_signals()
        self._set_status("DETENIDO", "#697386")
        self._set_running_state(False)

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.setWindowTitle("EEG/BCI Pseudo-Online Demonstration")
        self.setMinimumSize(720, 535)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(16)

        title = QLabel("EEG/BCI Pseudo-Online Demonstration System")
        title.setObjectName("title")
        subtitle = QLabel("EEGNet · Streaming simulado desde EDF · Clasificación en consola")
        subtitle.setObjectName("subtitle")
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("separator")
        main_layout.addWidget(line)

        data_group = QGroupBox("Configuración de adquisición")
        data_grid = QGridLayout(data_group)
        data_grid.setHorizontalSpacing(12)
        data_grid.setVerticalSpacing(12)

        self.edf_line = self._readonly_line("Seleccione un archivo EDF...")
        self.edf_button = QPushButton("Buscar EDF")
        self.paradigm_combo = QComboBox()
        self.paradigm_combo.addItem("Rest / Left / Right", "1")
        self.paradigm_combo.addItem("Rest / Hands / Feet", "2")

        data_grid.addWidget(QLabel("Archivo EDF"), 0, 0)
        data_grid.addWidget(self.edf_line, 0, 1)
        data_grid.addWidget(self.edf_button, 0, 2)
        data_grid.addWidget(QLabel("Paradigma"), 1, 0)
        data_grid.addWidget(self.paradigm_combo, 1, 1, 1, 2)
        main_layout.addWidget(data_group)

        model_group = QGroupBox("Configuración del modelo")
        model_grid = QGridLayout(model_group)
        model_grid.setHorizontalSpacing(12)
        model_grid.setVerticalSpacing(12)

        self.model_line = self._readonly_line("Seleccione un modelo .keras...")
        self.model_button = QPushButton("Buscar modelo")
        self.hierarchy_checkbox = QCheckBox("Usar jerarquía binaria")
        self.hierarchy_checkbox.setToolTip(
            "Modelo principal: rest vs no-rest; modelo auxiliar: discriminación motora."
        )
        self.aux_line = self._readonly_line("Desactivado")
        self.aux_button = QPushButton("Buscar auxiliar")
        self.slide_checkbox = QCheckBox("Ventana deslizante / hop de 0.5 s")
        self.slide_checkbox.setChecked(True)

        model_grid.addWidget(QLabel("Modelo principal"), 0, 0)
        model_grid.addWidget(self.model_line, 0, 1)
        model_grid.addWidget(self.model_button, 0, 2)
        model_grid.addWidget(self.hierarchy_checkbox, 1, 1, 1, 2)
        model_grid.addWidget(QLabel("Modelo auxiliar"), 2, 0)
        model_grid.addWidget(self.aux_line, 2, 1)
        model_grid.addWidget(self.aux_button, 2, 2)
        model_grid.addWidget(self.slide_checkbox, 3, 1, 1, 2)
        main_layout.addWidget(model_group)

        status_group = QGroupBox("Estado de ejecución")
        status_grid = QGridLayout(status_group)
        status_grid.setVerticalSpacing(8)

        self.status_label = QLabel()
        self.loaded_edf_label = QLabel("--")
        self.loaded_model_label = QLabel("--")
        self.loaded_aux_label = QLabel("--")

        status_grid.addWidget(QLabel("Sistema"), 0, 0)
        status_grid.addWidget(self.status_label, 0, 1)
        status_grid.addWidget(QLabel("EDF cargado"), 1, 0)
        status_grid.addWidget(self.loaded_edf_label, 1, 1)
        status_grid.addWidget(QLabel("Modelo cargado"), 2, 0)
        status_grid.addWidget(self.loaded_model_label, 2, 1)
        status_grid.addWidget(QLabel("Auxiliar"), 3, 0)
        status_grid.addWidget(self.loaded_aux_label, 3, 1)
        main_layout.addWidget(status_group)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.start_button = QPushButton("Start Streaming")
        self.start_button.setObjectName("startButton")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        main_layout.addLayout(buttons)

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #F5F7FA;
                color: #182230;
                font-family: "Segoe UI";
                font-size: 10.5pt;
            }
            QLabel#title {
                font-size: 18pt;
                font-weight: 700;
                color: #12233F;
            }
            QLabel#subtitle {
                color: #697386;
                font-size: 10pt;
            }
            QFrame#separator {
                color: #D9E0E8;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #D9E0E8;
                border-radius: 10px;
                margin-top: 10px;
                padding: 14px 12px 10px 12px;
                font-weight: 600;
                color: #12233F;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                background-color: white;
            }
            QLineEdit, QComboBox {
                min-height: 32px;
                padding: 0 9px;
                border: 1px solid #CDD6E0;
                border-radius: 6px;
                background: #FBFCFE;
            }
            QLineEdit:disabled, QComboBox:disabled {
                color: #8A94A6;
                background: #F0F3F7;
            }
            QPushButton {
                min-height: 34px;
                padding: 0 14px;
                border-radius: 7px;
                border: 1px solid #C8D1DC;
                background-color: white;
                font-weight: 600;
            }
            QPushButton:hover:!disabled {
                background-color: #EEF4FF;
                border-color: #91B2EA;
            }
            QPushButton:disabled {
                color: #95A1B2;
                background-color: #EEF1F5;
            }
            QPushButton#startButton {
                min-width: 150px;
                background-color: #1455A3;
                color: white;
                border: none;
            }
            QPushButton#startButton:hover:!disabled {
                background-color: #104789;
            }
            QPushButton#stopButton {
                min-width: 110px;
                background-color: #FFF4F3;
                color: #B42318;
                border: 1px solid #FDA29B;
            }
            QPushButton#stopButton:hover:!disabled {
                background-color: #FEE4E2;
            }
            QCheckBox {
                spacing: 8px;
                font-weight: 500;
            }
            """
        )

    @staticmethod
    def _readonly_line(placeholder: str) -> QLineEdit:
        line = QLineEdit()
        line.setReadOnly(True)
        line.setPlaceholderText(placeholder)
        return line

    def _configure_processes(self) -> None:
        # Toda la predicción y los logs permanecen visibles en la consola original.
        for process in (self.streamer_process, self.receiver_process):
            process.setWorkingDirectory(str(REPO_ROOT))
            process.setProcessChannelMode(QProcess.ForwardedChannels)

    def _connect_signals(self) -> None:
        self.edf_button.clicked.connect(self.select_edf)
        self.model_button.clicked.connect(self.select_model)
        self.aux_button.clicked.connect(self.select_aux_model)
        self.hierarchy_checkbox.toggled.connect(self.toggle_binary_hierarchy)
        self.start_button.clicked.connect(self.start_streaming)
        self.stop_button.clicked.connect(self.stop_streaming)

        self.streamer_process.finished.connect(self._process_finished)
        self.receiver_process.finished.connect(self._process_finished)
        self.streamer_process.errorOccurred.connect(self._process_error)
        self.receiver_process.errorOccurred.connect(self._process_error)

    # ------------------------------------------------------------------ #
    # Selección y validación de archivos
    # ------------------------------------------------------------------ #
    def select_edf(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar señal EDF",
            str(REPO_ROOT),
            "EDF files (*.edf);;Todos los archivos (*)",
        )
        if filename:
            self.edf_path = Path(filename).resolve()
            self.edf_line.setText(str(self.edf_path))
            self.loaded_edf_label.setText(self.edf_path.name)

    def select_model(self) -> None:
        selected = self._choose_keras_file("Seleccionar modelo principal")
        if selected is not None:
            self.model_path = selected
            self.model_line.setText(str(selected))
            self.loaded_model_label.setText(selected.name)

    def select_aux_model(self) -> None:
        selected = self._choose_keras_file("Seleccionar modelo auxiliar")
        if selected is not None:
            self.aux_model_path = selected
            self.aux_line.setText(str(selected))
            self.loaded_aux_label.setText(selected.name)

    def _choose_keras_file(self, caption: str) -> Optional[Path]:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            caption,
            str(REPO_ROOT),
            "Keras models (*.keras);;Todos los archivos (*)",
        )
        if not filename:
            return None

        path = Path(filename).resolve()
        params_path = path.with_suffix(".npz")
        if not params_path.exists():
            QMessageBox.warning(
                self,
                "Parámetros no encontrados",
                "El modelo fue seleccionado, pero no existe su archivo .npz asociado:\n\n"
                f"{params_path.name}\n\n"
                "Ambos archivos deben estar en el mismo directorio.",
            )
            return None
        return path

    def toggle_binary_hierarchy(self, enabled: bool) -> None:
        self.aux_button.setEnabled(enabled and not self._running)
        self.aux_line.setEnabled(enabled)
        if not enabled:
            self.aux_model_path = None
            self.aux_line.clear()
            self.aux_line.setPlaceholderText("Desactivado")
            self.loaded_aux_label.setText("--")
        else:
            self.aux_line.setPlaceholderText("Seleccione el modelo auxiliar .keras...")

    def _validate_configuration(self) -> bool:
        if self.edf_path is None or not self.edf_path.exists():
            self._show_missing("Debes seleccionar un archivo EDF válido.")
            return False

        if self.model_path is None or not self.model_path.exists():
            self._show_missing("Debes seleccionar un modelo principal .keras válido.")
            return False

        if not self.model_path.with_suffix(".npz").exists():
            self._show_missing("No se encontró el .npz asociado al modelo principal.")
            return False

        if self.hierarchy_checkbox.isChecked():
            if self.aux_model_path is None or not self.aux_model_path.exists():
                self._show_missing(
                    "La jerarquía binaria requiere seleccionar un modelo auxiliar."
                )
                return False
            if not self.aux_model_path.with_suffix(".npz").exists():
                self._show_missing("No se encontró el .npz del modelo auxiliar.")
                return False

        if not STREAMING_SCRIPT.exists() or not RECEIVING_SCRIPT.exists():
            self._show_missing(
                "gui.py debe estar en la misma carpeta que streaming.py y receiving.py."
            )
            return False

        return True

    def _show_missing(self, message: str) -> None:
        QMessageBox.warning(self, "Configuración incompleta", message)

    # ------------------------------------------------------------------ #
    # Ejecución
    # ------------------------------------------------------------------ #
    def start_streaming(self) -> None:
        if self._running or not self._validate_configuration():
            return

        self._running = True
        self._stop_requested = False
        self._set_running_state(True)
        self._set_status("INICIANDO...", "#B54708")

        stream_args = [
            str(STREAMING_SCRIPT),
            "--edf",
            str(self.edf_path),
            "--type",
            str(self.paradigm_combo.currentData()),
        ]

        print("\n[GUI] Lanzando streamer...")
        print(f"[GUI] EDF: {self.edf_path.name}")
        print(f"[GUI] Paradigma: {self.paradigm_combo.currentText()}")
        self.streamer_process.start(sys.executable, stream_args)

        if not self.streamer_process.waitForStarted(1500):
            self._abort_start("No fue posible iniciar streaming.py.")
            return

        # El receiver envía READY; el pequeño retardo garantiza que el
        # socket CONTROL_PORT del streamer ya esté enlazado.
        QTimer.singleShot(400, self._start_receiver)

    def _start_receiver(self) -> None:
        if not self._running or self._stop_requested:
            return

        receiver_args = [
            str(RECEIVING_SCRIPT),
            "--model",
            str(self.model_path),
        ]

        if self.slide_checkbox.isChecked():
            receiver_args.append("--slide")
        else:
            receiver_args.append("--no-slide")

        if self.hierarchy_checkbox.isChecked():
            receiver_args.extend(
                ["--use-bin", "--aux-model", str(self.aux_model_path)]
            )

        print("[GUI] Lanzando receiver...")
        print(
            "[GUI] Modo: "
            + ("jerarquía binaria" if self.hierarchy_checkbox.isChecked() else "triclase")
        )
        print(
            "[GUI] Slide/hop: "
            + ("activo (0.5 s)" if self.slide_checkbox.isChecked() else "inactivo (1.0 s)")
        )

        self.receiver_process.start(sys.executable, receiver_args)
        if not self.receiver_process.waitForStarted(1500):
            self._send_stop()
            self._abort_start("No fue posible iniciar receiving.py.")
            return

        self._set_status("STREAMING ACTIVO", "#027A48")

    def stop_streaming(self) -> None:
        if not self._running:
            return

        self._stop_requested = True
        self._set_status("DETENIENDO...", "#B54708")
        self.stop_button.setEnabled(False)
        self._send_stop()
        print("[GUI] STOP enviado al streamer.")

    @staticmethod
    def _send_stop() -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(b"STOP", (HOST, CONTROL_PORT))
        except OSError as exc:
            print(f"[GUI][WARNING] No se pudo enviar STOP: {exc}")

    def _abort_start(self, message: str) -> None:
        self._running = False
        self._set_running_state(False)
        self._set_status("ERROR", "#B42318")
        QMessageBox.critical(self, "Error de ejecución", message)

    def _process_error(self, _: QProcess.ProcessError) -> None:
        if self._running and not self._stop_requested:
            self._set_status("ERROR DE PROCESO", "#B42318")

    def _process_finished(self) -> None:
        if not self._running:
            return

        streamer_done = self.streamer_process.state() == QProcess.NotRunning
        receiver_done = self.receiver_process.state() == QProcess.NotRunning

        if streamer_done and receiver_done:
            self._running = False
            self._set_running_state(False)
            if self._stop_requested:
                self._set_status("DETENIDO POR USUARIO", "#697386")
            else:
                self._set_status("FINALIZADO", "#1455A3")

    def _set_running_state(self, running: bool) -> None:
        self.edf_button.setEnabled(not running)
        self.model_button.setEnabled(not running)
        self.paradigm_combo.setEnabled(not running)
        self.hierarchy_checkbox.setEnabled(not running)
        self.slide_checkbox.setEnabled(not running)
        self.aux_button.setEnabled(
            (not running) and self.hierarchy_checkbox.isChecked()
        )
        self.aux_line.setEnabled(self.hierarchy_checkbox.isChecked())
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-weight: 700; color: {color}; letter-spacing: 0.4px;"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._running:
            self._send_stop()
            for process in (self.receiver_process, self.streamer_process):
                if process.state() != QProcess.NotRunning:
                    process.waitForFinished(1200)
                if process.state() != QProcess.NotRunning:
                    process.kill()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = EEGDemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
