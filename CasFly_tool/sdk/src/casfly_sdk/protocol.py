"""CustomProtocol — UDP socket layer for device-to-device CasFly communication.

Each CasFlyDevice instance owns one CustomProtocol bound to its (host, port).
Packets are JSON-serialized dicts sent as UDP datagrams, matching the
communication layer used in the paper experiments.
"""
from __future__ import annotations

import json
import logging
import os
import socket


class CustomProtocol:
    """UDP send/receive wrapper matching the paper's device communication layer.

    Parameters
    ----------
    host:
        IP address to bind (e.g. ``"127.0.0.1"`` or a Raspberry Pi LAN IP).
    port:
        UDP port to bind.
    log_dir:
        Directory for the protocol log file.
    """

    def __init__(self, host: str, port: int, log_dir: str = "./logs") -> None:
        self.host = host
        self.port = port
        self.server_socket: socket.socket | None = None
        os.makedirs(log_dir, exist_ok=True)
        self._log = logging.getLogger(f"casfly.protocol.{host}.{port}")
        if not self._log.handlers:
            handler = logging.FileHandler(os.path.join(log_dir, "protocol_log.txt"))
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self._log.addHandler(handler)
            self._log.setLevel(logging.INFO)
        self._start_server()

    def _start_server(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            self.server_socket = sock
            self._log.info("Server started at %s:%s", self.host, self.port)
        except Exception as exc:
            self._log.error("Failed to start server at %s:%s — %s", self.host, self.port, exc)
            raise

    def send_packet(self, packet: dict, target_host: str, target_port: int) -> None:
        """Serialize *packet* to JSON and send it as a UDP datagram."""
        try:
            data = json.dumps(packet).encode("utf-8")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(data, (target_host, target_port))
            self._log.info("Packet sent to %s:%s", target_host, target_port)
        except Exception as exc:
            self._log.error("Failed to send to %s:%s — %s", target_host, target_port, exc)

    def receive_packet(
        self, timeout: float | None = None
    ) -> tuple[dict | None, tuple | None]:
        """Block until a UDP packet arrives (or *timeout* seconds elapses).

        Returns ``(packet_dict, sender_addr)`` or ``(None, None)`` on timeout/error.
        """
        if self.server_socket is None:
            return None, None
        if timeout is not None:
            self.server_socket.settimeout(timeout)
        try:
            data, addr = self.server_socket.recvfrom(65535)
            packet = json.loads(data.decode("utf-8"))
            return packet, addr
        except socket.timeout:
            return None, None
        except Exception as exc:
            self._log.error("Error receiving packet: %s", exc)
            return None, None

    def close(self) -> None:
        """Close the UDP socket."""
        if self.server_socket:
            self.server_socket.close()
            self._log.info("Server socket closed at %s:%s", self.host, self.port)
