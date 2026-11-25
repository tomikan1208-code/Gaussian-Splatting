from datetime import datetime, timedelta, timezone
import hashlib
from ipaddress import IPv4Address
from pathlib import Path
import plistlib
import ssl
from tempfile import TemporaryDirectory
import uuid

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


CERT_NAME = "House Wi-Fi Capture"


def certificate(name, issuer, key, days):
    now = datetime.now(timezone.utc)
    return (x509.CertificateBuilder()
            .subject_name(name).issuer_name(issuer).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=days)))


def private_key_bytes(key):
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def key_usage(*, ca):
    return x509.KeyUsage(
        digital_signature=not ca, content_commitment=False, key_encipherment=not ca,
        data_encipherment=False, key_agreement=False, key_cert_sign=ca, crl_sign=ca,
        encipher_only=False, decipher_only=False,
    )


def prepare_tls(state: Path, addresses: list[str]):
    """Keep one local root so phones can reconnect after the PC's address changes."""
    state.mkdir(parents=True, exist_ok=True)
    root_key = state / "root.key"
    root_cert = state / "root.pem"

    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, CERT_NAME)])
    with TemporaryDirectory(prefix="tls-", dir=state) as temporary:
        scratch = Path(temporary)
        if root_key.is_file() and root_cert.is_file():
            ca_key = serialization.load_pem_private_key(root_key.read_bytes(), password=None)
            ca = x509.load_pem_x509_certificate(root_cert.read_bytes())
        else:
            ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            ca = (certificate(name, name, ca_key, 3650)
                  .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                  .add_extension(key_usage(ca=True), critical=True)
                  .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
                  .sign(ca_key, hashes.SHA256()))
            new_key, new_cert = scratch / "root.key", scratch / "root.pem"
            new_key.write_bytes(private_key_bytes(ca_key))
            new_cert.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
            new_key.replace(root_key)
            new_cert.replace(root_cert)

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        names = [x509.DNSName("localhost"), *(x509.IPAddress(IPv4Address(address)) for address in addresses)]
        leaf = (certificate(name, ca.subject, key, 180)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(key_usage(ca=False), critical=True)
                .add_extension(x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                .add_extension(x509.SubjectAlternativeName(names), critical=False)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                               critical=False)
                .sign(ca_key, hashes.SHA256()))
        server_key, leaf_cert = scratch / "server.key", scratch / "server.pem"
        server_key.write_bytes(private_key_bytes(key))
        leaf_cert.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(leaf_cert, server_key)

    der = ca.public_bytes(serialization.Encoding.DER)
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
