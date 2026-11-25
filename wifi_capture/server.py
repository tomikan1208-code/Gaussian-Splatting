import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from ipaddress import IPv4Interface, IPv4Address
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from tls_setup import prepare_tls


HERE = Path(__file__).resolve().parent
SERVICE = "house-wifi-capture"
PUBLIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/capture.css": ("capture.css", "text/css; charset=utf-8"),
    "/capture.js": ("capture.js", "application/javascript; charset=utf-8"),
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".mov", ".mp4"}
PNG_HEADER = b"\x89PNG\r\n\x1a\n"
INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_DEVICES = re.compile(r"(?i:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])")


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def clean_name(name, length):
    return INVALID_NAME.sub("_", name)[:length].strip(" .")


def upload_name(value):
    try:
        name = unquote(value, errors="strict")
    except UnicodeError:
        raise RequestError(400, "ファイル名を読み取れませんでした。") from None
    if "/" in name or "\\" in name:
        raise RequestError(400, "フォルダを含まないファイル名を指定してください。")
    suffix = Path(name).suffix
    if suffix.lower() not in EXTENSIONS:
        raise RequestError(415, "PNG、JPEG、HEIC、HEIF、MOV、MP4を送信できます。")
    stem = clean_name(name[:-len(suffix)], 96) or "media"
    if WINDOWS_DEVICES.fullmatch(stem.split(".", 1)[0]):
        stem = "_" + stem
    return stem + suffix


def available_name(folder, name):
    candidate = folder / name
    index = 2
    while candidate.exists():
        candidate = folder / f"{Path(name).stem}_{index}{Path(name).suffix}"
        index += 1
    return candidate


def verify_media(path, suffix):
    with path.open("rb") as stream:
        header = stream.read(12)
    if suffix == ".png":
        valid = header.startswith(PNG_HEADER) and header[8:12] == b"\x00\x00\x00\x0d"
    elif suffix in {".jpg", ".jpeg"}:
        valid = header.startswith(b"\xff\xd8\xff")
    elif suffix in {".heic", ".heif"}:
        valid = header[4:8] == b"ftyp"
    else:
        valid = header[4:8] in {b"ftyp", b"wide", b"moov", b"mdat", b"free", b"skip", b"uuid"}
    if not valid:
        raise RequestError(415, "ファイルの形式を確認してください。")


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass
class Session:
    folder: Path
    kind: str
    lock: Lock = field(default_factory=Lock)
    files: dict = field(default_factory=dict)

    def save(self, staged, index, name, size, readings):
        with self.lock:
            if index in self.files:
                old_name, old_size, saved = self.files[index]
                if (name, size) != (old_name, old_size):
                    raise RequestError(409, "別のファイル番号で送り直してください。")
                return saved
            destination = available_name(self.folder, name)
            if readings is None:
                staged.replace(destination)
            else:
                row = json_bytes({
                    "file": destination.name, "sequence": index,
                    "received_at": timestamp(), **readings,
                }) + b"\n"
                with (self.folder / "frames.jsonl").open("ab") as log:
                    offset = log.tell()
                    try:
                        log.write(row)
                        log.flush()
                        staged.replace(destination)
                    except OSError:
                        log.truncate(offset)
                        raise
            self.files[index] = name, size, destination.name
            return destination.name


class CaptureApp:
    def __init__(self, output, interfaces, port):
        self.output = output
        self.interfaces = tuple(interfaces)
        self.port = port
        self.tls_port = None
        self.profile = None
        self.code = secrets.token_urlsafe(9)
        self.sessions = {}
        self.shutdown = Event()

    def accepts(self, address):
        peer = IPv4Address(address)
        return peer.is_loopback or any(peer in interface.network for interface in self.interfaces)

    def share(self, address):
        return f"http://{address}:{self.port}/#{self.code}"

    def state(self, local, paired):
        response = {
            "service": SERVICE, "local": local, "paired": paired,
            "tlsPort": self.tls_port,
        }
        if paired:
            response["output"] = str(self.output)
        if local:
            response["code"] = self.code
            response["links"] = [
                {"address": str(interface.ip), "url": self.share(interface.ip)}
                for interface in self.interfaces
            ]
        return response

    def create_session(self, data):
        kind = data.get("kind")
        label = data.get("label", "")
        if kind not in {"capture", "upload"} or not isinstance(label, str):
            raise RequestError(400, "撮影またはファイル送信を指定してください。")
        settings = data.get("settings")
        if kind == "capture":
            if not isinstance(settings, dict):
                raise RequestError(400, "撮影設定を指定してください。")
            edge, fps = settings.get("edge"), settings.get("fps")
            if type(edge) is not int or not 0 <= edge <= 4096:
                raise RequestError(400, "解像度を確認してください。")
            if type(fps) not in {int, float} or not 0.5 <= fps <= 30:
                raise RequestError(400, "fpsを確認してください。")
            settings = {"edge": edge, "fps": fps, "format": "PNG"}
        else:
            settings = None
        identifier = secrets.token_hex(12)
        name = clean_name(label, 40) or ("capture" if kind == "capture" else "upload")
        folder = self.output / f"{datetime.now():%Y%m%d-%H%M%S}_{name}_{identifier[:8]}"
        folder.mkdir()
        (folder / "session.json").write_bytes(json_bytes({
            "kind": kind, "started_at": timestamp(), "settings": settings,
        }))
        self.sessions[identifier] = Session(folder, kind)
        return {"id": identifier, "folder": str(folder)}


def qr_png(value):
    import qrcode
    from qrcode.image.pure import PyPNGImage

    buffer = BytesIO()
    qrcode.make(value, image_factory=PyPNGImage, box_size=6).save(buffer)
    return buffer.getvalue()


class Receiver(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HouseWiFiCapture"
    sys_version = ""

    @property
    def app(self):
        return self.server.app

    @property
    def local(self):
        return self.client_address[0] == "127.0.0.1"

    @property
    def paired(self):
        return secrets.compare_digest(
            self.headers.get("X-Pair-Code", "").encode("utf-8"), self.app.code.encode("ascii"),
        )

    def setup(self):
        super().setup()
        self.connection.settimeout(60)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except ConnectionError:
            # A peer can disconnect before dispatch or while receiving an error response.
            self.close_connection = True

    def log_message(self, *args):
        pass

    def send_bytes(self, status, body, media_type, **headers):
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' blob: data:; "
                         "media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
        self.send_header("Permissions-Policy", "camera=(self), microphone=(), "
                         "accelerometer=(self), gyroscope=(self), magnetometer=(self)")
        if status >= 400:
            self.close_connection = True
            self.send_header("Connection", "close")
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_bytes(status, json_bytes(value), "application/json; charset=utf-8")

    def read_length(self, maximum=None):
        raw = self.headers.get("Content-Length", "")
        if not raw or not raw.isascii() or not raw.isdecimal() or len(raw) > 18:
            raise RequestError(411, "ファイルサイズを取得できませんでした。")
        length = int(raw)
        if length < 1 or (maximum is not None and length > maximum):
            raise RequestError(413, "送信データのサイズを確認してください。")
        return length

    def read_json(self):
        try:
            data = json.loads(self.rfile.read(self.read_length(4096)))
        except (UnicodeError, ValueError):
            raise RequestError(400, "送信データを読み取れませんでした。") from None
        if not isinstance(data, dict):
            raise RequestError(400, "送信データを読み取れませんでした。")
        return data

    def allowed_host(self):
        hosts = {"localhost", "127.0.0.1", *(str(interface.ip) for interface in self.app.interfaces)}
        authority = self.headers.get("Host", "")
        return authority in {f"{host}:{self.server.server_port}" for host in hosts}

    def dispatch(self, method):
        try:
            if not self.app.accepts(self.client_address[0]) or not self.allowed_host():
                raise RequestError(403, "同じWi-FiからPCのURLを開いてください。")
            path = urlsplit(self.path)
            if method == "GET":
                self.read_route(path)
            else:
                if not self.paired:
                    raise RequestError(401, "PCの接続コードを確認してください。")
                self.write_route(path.path)
        except RequestError as error:
            self.send_json({"error": str(error)}, error.status)
        except (ConnectionError, TimeoutError):
            self.close_connection = True
        except OSError as error:
            print(f"Save or socket error: {error}", file=sys.stderr, flush=True)
            self.send_json({"error": "PCに保存できませんでした。空き容量と保存先を確認してください。"}, 500)

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def read_route(self, path):
        if path.path in PUBLIC:
            name, content_type = PUBLIC[path.path]
            self.send_bytes(200, (HERE / name).read_bytes(), content_type)
        elif path.path == "/api/state":
            self.send_json(self.app.state(self.local, self.local or self.paired))
        elif path.path == "/profile.mobileconfig" and self.app.profile:
            self.send_bytes(200, self.app.profile, "application/x-apple-aspen-config",
                            Content_Disposition='attachment; filename="House-WiFi-Capture.mobileconfig"')
        elif path.path == "/pair.png" and self.local:
            address = parse_qs(path.query).get("address", [""])[0]
            if address not in {str(interface.ip) for interface in self.app.interfaces}:
                raise RequestError(404, "この接続先は見つかりませんでした。")
            try:
                png = qr_png(self.app.share(address))
            except ImportError:
                raise RequestError(404, "表示されたURLを使ってください。") from None
            self.send_bytes(200, png, "image/png")
        else:
            raise RequestError(404, "このページは見つかりませんでした。")

    def write_route(self, path):
        if path == "/api/session":
            self.send_json(self.app.create_session(self.read_json()), 201)
        elif path == "/api/file":
            self.receive_file()
        elif path == "/api/stop" and self.local:
            self.send_json({"stopped": True})
            self.app.shutdown.set()
        else:
            raise RequestError(404, "このページは見つかりませんでした。")

    def receive_file(self):
        session = self.app.sessions.get(self.headers.get("X-Session", ""))
        if session is None:
            raise RequestError(409, "撮影またはファイル送信を開始し直してください。")
        raw_index = self.headers.get("X-Index", "")
        if not re.fullmatch(r"[1-9][0-9]{0,8}", raw_index):
            raise RequestError(400, "ファイル番号を確認してください。")
        index = int(raw_index)
        if session.kind == "capture":
            raw = self.headers.get("X-Frame-Info", "")
            try:
                readings = json.loads(raw) if len(raw) <= 2048 else None
                if not isinstance(readings, dict):
                    raise ValueError
                readings = {key: readings[key] for key in (
                    "capture_type", "captured_at", "elapsed_ms", "video_seconds", "width", "height", "orientation",
                ) if key in readings}
                json_bytes(readings)
            except (ValueError, RecursionError):
                raise RequestError(400, "撮影時刻や姿勢の情報を読み取れませんでした。") from None
            prefix = "photo" if readings.get("capture_type") == "photo" else "frame"
            name = f"{prefix}_{index:06}.png"
            length = self.read_length(48 * 1024 * 1024)
        else:
            name = upload_name(self.headers.get("X-File-Name", ""))
            readings = None
            length = self.read_length()
        if length > shutil.disk_usage(session.folder).free:
            raise RequestError(507, "PCの空き容量が不足しています。")

        temporary = None
        try:
            with NamedTemporaryFile(dir=session.folder, prefix=".receiving-", suffix=".part",
                                    delete=False) as stream:
                temporary = Path(stream.name)
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise RequestError(400, "受信が途中で止まりました。再送してください。")
                    stream.write(chunk)
                    remaining -= len(chunk)
            verify_media(temporary, Path(name).suffix.lower())
            saved = session.save(temporary, index, name, length, readings)
            self.send_json({"name": saved, "bytes": length})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class Server(ThreadingHTTPServer):
    allow_reuse_address = False


def main():
    parser = argparse.ArgumentParser(description="Receive phone photos and video over the local network.")
    parser.add_argument("--output", type=Path, default=HERE.parent / "wifi-captures")
    parser.add_argument("--state", type=Path,
                        default=Path(os.environ.get("LOCALAPPDATA", Path.home())) / "HouseWiFiCapture")
    parser.add_argument("--lan", action="append", type=IPv4Interface, default=[])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tls-port", type=int, default=8766)
    parser.add_argument("--openssl", type=Path)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = CaptureApp(output, args.lan, args.port)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    with ExitStack() as resources:
        http = resources.enter_context(Server((host, args.port), Receiver))
        servers = [http]
        if args.openssl:
            try:
                https = resources.enter_context(Server((host, args.tls_port), Receiver))
                addresses = ["127.0.0.1", *(str(interface.ip) for interface in args.lan)]
                context, app.profile = prepare_tls(args.openssl, args.state.resolve(), addresses)
                https.socket = context.wrap_socket(https.socket, server_side=True,
                                                   do_handshake_on_connect=False)
                app.tls_port = https.server_port
                servers.append(https)
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"HTTPS setup failed: {error}", file=sys.stderr, flush=True)
        for server in servers:
            server.app = app
            Thread(target=server.serve_forever, daemon=True).start()
        pc_url = f"http://127.0.0.1:{http.server_port}/"
        print(f"PC: {pc_url}", flush=True)
        for interface in args.lan:
            print(f"Phone: http://{interface.ip}:{http.server_port}/", flush=True)
        print(f"Save to: {output}", flush=True)
        if args.open:
            webbrowser.open(pc_url)
        try:
            app.shutdown.wait()
        except KeyboardInterrupt:
            pass
        finally:
            for server in servers:
                server.shutdown()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE / "vendor"))
    main()
