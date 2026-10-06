import base64
import json
import re
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

APP_SECRET = 'OV-SECURE-2026-!@#$%^&*()_+OFFLINE-VALIDATOR'
V1 = 'v1'
SHIFT_BASE = 7


def fnv1a_32(data: bytes) -> str:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def keystream_byte(secret: str, salt: str, i: int, position_mix: int = 0) -> int:
    a = ord(secret[i % len(secret)]) if secret else 0
    b = ord(salt[i % len(salt)]) if salt else 0
    m = (position_mix or 0) & 0xFF
    return ((a * 31) ^ (b * 17) ^ (i * 53) ^ (m * 7)) & 0xFF


def decode_token(token: str, expected_salt: str = None):
    if not isinstance(token, str):
        return None
    parts = token.split('$')
    if len(parts) != 4:
        return None
    ver, salt, checksum, payload_b64 = parts
    if ver != V1:
        return None
    if expected_salt and salt != expected_salt:
        return None
    try:
        raw = base64.b64decode(payload_b64)
    except Exception:
        return None
    if len(raw) < 8 or raw[0:2] != b'v1':
        return None
    if fnv1a_32(raw) != checksum:
        return None
    length = raw[2] | (raw[3] << 8)
    if length != len(raw) - 8:
        return None
    secret = APP_SECRET + ':' + salt
    out = bytearray(length)
    for i in range(length):
        b = raw[8 + i]
        shift = (SHIFT_BASE + (i % 5)) & 7
        b = ((b >> shift) | (b << (8 - shift))) & 0xFF
        b ^= keystream_byte(secret, salt, i)
        out[i] = b
    try:
        return out.decode('utf-8')
    except UnicodeDecodeError:
        return None


DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})')
TIME_RE = re.compile(r'(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})')


def _parse_date(s):
    try:
        return datetime.strptime(s.strip(), '%Y-%m-%d').date()
    except ValueError:
        return None


def _parse_time(s):
    try:
        return datetime.strptime(s.strip(), '%H:%M').time()
    except ValueError:
        return None


def check_field(f: dict, now: datetime):
    t = f.get('type')
    val = str(f.get('val', ''))

    if t == 'daterange':
        m = DATE_RE.search(val)
        if not m:
            return False, 'invalid daterange'
        d1, d2 = _parse_date(m.group(1)), _parse_date(m.group(2))
        if d1 is None or d2 is None:
            return False, 'invalid daterange'
        if not (d1 <= now.date() <= d2):
            return False, f'date out of range ({d1} to {d2})'
        return True, None

    if t == 'timerange':
        m = TIME_RE.search(val)
        if not m:
            return False, 'invalid timerange'
        t1, t2 = _parse_time(m.group(1)), _parse_time(m.group(2))
        if t1 is None or t2 is None:
            return False, 'invalid timerange'
        cur = now.time().replace(second=0, microsecond=0)
        if t1 <= t2:
            ok = t1 <= cur <= t2
        else:  # overnight span
            ok = cur >= t1 or cur <= t2
        if not ok:
            return False, f'time out of range ({t1:%H:%M} to {t2:%H:%M})'
        return True, None

    return True, None  # other types don't restrict validity


def validate_key(token: str, expected_salt: str = None, tz: str = 'Asia/Dhaka'):
    raw = decode_token(token, expected_salt=expected_salt)
    if raw is None:
        return False, 'invalid or tampered key', None
    try:
        payload = json.loads(raw)
    except ValueError:
        return False, 'key payload is not JSON', None

    fields = payload.get('fields')
    if not isinstance(fields, list) or not fields:
        return False, 'key has no fields', None

    timed = [f for f in fields if f.get('type') in ('daterange', 'timerange')]
    if not timed:
        return False, 'key has no date/time validity', None

    try:
        now = datetime.now(ZoneInfo(tz))
    except Exception:
        now = datetime.now()

    for f in fields:
        ok, reason = check_field(f, now)
        if not ok:
            return False, reason, payload
    return True, 'valid', payload
