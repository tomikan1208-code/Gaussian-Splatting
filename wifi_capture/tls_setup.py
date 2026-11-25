from datetime import datetime
import hashlib
from pathlib import Path
import plistlib
import ssl
import subprocess
from tempfile import TemporaryDirectory
import uuid


CERT_NAME = "House Wi-Fi Capture"


def prepare_tls(openssl: Path, state: Path, addresses: list[str]):
    """Keep one local root so phones can reconnect after the PC's address changes."""
    state.mkdir(parents=True, exist_ok=True)
    root_key = state / "root.key"
    root_cert = state / "root.pem"

    def run(*args):
        return subprocess.run(
            [str(openssl), *map(str, args)], capture_output=True, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout

    with TemporaryDirectory(prefix="tls-", dir=state) as temporary:
        scratch = Path(temporary)
        if not (root_key.is_file() and root_cert.is_file()):
            new_key, new_cert = scratch / "root.key", scratch / "root.pem"
            run(
                "req", "-new", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
                "-days", 3650, "-subj", f"/CN={CERT_NAME}",
                "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign",
                "-addext", "subjectKeyIdentifier=hash",
                "-keyout", new_key, "-out", new_cert,
            )
            new_key.replace(root_key)
            new_cert.replace(root_cert)

        server_key, request = scratch / "server.key", scratch / "server.csr"
        leaf_cert, extensions = scratch / "server.pem", scratch / "extensions.ini"
        subject_alt_names = ["DNS:localhost", *(f"IP:{address}" for address in addresses)]
        extensions.write_text(
            "basicConstraints=critical,CA:FALSE\n"
            "keyUsage=critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
            f"subjectAltName={','.join(subject_alt_names)}\n",
            encoding="ascii",
        )
        run("req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={CERT_NAME}",
            "-keyout", server_key, "-out", request)
        run("x509", "-req", "-in", request, "-CA", root_cert, "-CAkey", root_key,
            "-set_serial", f"0x{uuid.uuid4().hex}", "-days", 180, "-sha256",
            "-extfile", extensions, "-out", leaf_cert)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(leaf_cert, server_key)

    der = ssl.PEM_cert_to_DER_cert(root_cert.read_text(encoding="ascii"))
    digest = hashlib.sha256(der).hexdigest()
    identifier = f"local.house-wifi-capture.{digest[:12]}"
    profile = plistlib.dumps({
        "PayloadType": "Configuration",
        "PayloadVersion": 1,
        "PayloadIdentifier": identifier,
        "PayloadUUID": str(uuid.uuid5(uuid.NAMESPACE_OID, identifier)).upper(),
        "PayloadDisplayName": CERT_NAME,
        "PayloadDescription": "Trust the local Wi-Fi Capture server on this PC.",
        "PayloadContent": [{
            "PayloadType": "com.apple.security.root",
            "PayloadVersion": 1,
            "PayloadIdentifier": identifier + ".root",
            "PayloadUUID": str(uuid.uuid5(uuid.NAMESPACE_OID, identifier + ".root")).upper(),
            "PayloadDisplayName": CERT_NAME,
            "PayloadCertificateFileName": "House-WiFi-Capture.cer",
            "PayloadContent": der,
        }],
    })
    return context, profile
