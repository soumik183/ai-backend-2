import os
import time
import json
import logging
import requests
from flask import Flask, request, Response, stream_with_context, jsonify
from validator import validate_key

import sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("ai-backend")

app = Flask(__name__)

DEFAULT_CONFIG = {
    "default_model": "gemma-4-26b-a4b-it-4bit",
    "validator_salt": "",
    "key_timezone": "Asia/Dhaka",
    "upstream_timeout": 300,
    "upstream_retries": 1,
    "upstream_retry_backoff": 0.5,
    "cors_origin": "*",
    "port": 8000,
}

CONFIG_FILE = os.environ.get("CONFIG_FILE", "config.json")


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    except (OSError, ValueError) as e:
        log.warning("Cannot read %s (%s), using defaults", CONFIG_FILE, e)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass
    return cfg


CFG = load_config()

DEFAULT_MODEL = CFG["default_model"]
VALIDATOR_SALT = CFG["validator_salt"] or None
KEY_TIMEZONE = CFG["key_timezone"]
UPSTREAM_TIMEOUT = int(CFG["upstream_timeout"])
UPSTREAM_RETRIES = int(CFG["upstream_retries"])
UPSTREAM_RETRY_BACKOFF = float(CFG["upstream_retry_backoff"])
CORS_ORIGIN = CFG["cors_origin"]


def load_providers():
    env = os.environ.get("MODEL_PROVIDERS")
    if env:
        try:
            data = json.loads(env)
        except ValueError:
            log.error("MODEL_PROVIDERS env is not valid JSON, falling back to config")
            data = None
    else:
        data = None
    if data is None:
        data = CFG.get("providers")
    if not data:
        data = [{
            "name": "default",
            "url": os.environ.get("MODEL_SERVER_URL", "http://47.108.57.223:8000"),
            "models": [],
        }]
    out = []
    for p in data:
        if not p.get("url"):
            continue
        out.append({
            "name": p.get("name", p["url"]),
            "url": p["url"].rstrip("/"),
            "models": list(p.get("models", [])),
            "direct_only": bool(p.get("direct_only")),
        })
    return out


PROVIDERS = load_providers()
log.info("Loaded %d provider(s): %s", len(PROVIDERS), [p["name"] for p in PROVIDERS])

MODEL_INDEX = {}
for p in PROVIDERS:
    for m in p["models"]:
        MODEL_INDEX.setdefault(m, []).append(p)


def refresh_index_for(provider, model_ids):
    for mid in model_ids:
        MODEL_INDEX.setdefault(mid, [])
        if provider not in MODEL_INDEX[mid]:
            MODEL_INDEX[mid].append(provider)


def refresh_all_indexes():
    for p in PROVIDERS:
        try:
            r = requests.get(f"{p['url']}/v1/models", timeout=5)
            data = r.json()
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            refresh_index_for(p, ids)
        except (requests.exceptions.RequestException, ValueError):
            pass


refresh_all_indexes()


def providers_for_model(model):
    if model in MODEL_INDEX:
        return MODEL_INDEX[model]
    return []


def openai_error(message, status=500, err_type="server_error"):
    return jsonify({"error": {"message": message, "type": err_type, "code": None}}), status


@app.after_request
def harden(resp):
    resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


@app.before_request
def guard():
    if request.method == "OPTIONS":
        return Response(status=204)
    if request.path in ("/", "/health"):
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return openai_error("Missing API key", 401, "authentication_error")
    token = auth[len("Bearer "):].strip()
    ok, reason, _payload = validate_key(token, expected_salt=VALIDATOR_SALT, tz=KEY_TIMEZONE)
    if not ok:
        return openai_error(f"Invalid key: {reason}", 401, "authentication_error")
    return None


@app.errorhandler(400)
def bad_request(e):
    return openai_error("Bad request", 400, "invalid_request_error")


@app.errorhandler(404)
def not_found(e):
    return openai_error(f"Not found: {request.path}", 404, "invalid_request_error")


@app.errorhandler(405)
def method_not_allowed(e):
    return openai_error("Method not allowed", 405, "invalid_request_error")


@app.errorhandler(413)
def too_large(e):
    return openai_error("Request body too large", 413, "invalid_request_error")


@app.errorhandler(429)
def too_many(e):
    return openai_error("Rate limit exceeded", 429, "rate_limit_error")


@app.errorhandler(Exception)
def unhandled(e):
    log.exception("Unhandled error")
    return openai_error("Internal server error", 500)


@app.route("/")
def index():
    return jsonify({"status": "ok", "providers": [p["name"] for p in PROVIDERS], "models": len(MODEL_INDEX)})


@app.route("/health")
def health():
    report = []
    any_up = False
    for p in PROVIDERS:
        if p.get("direct_only"):
            any_up = True
            report.append({"name": p["name"], "up": "direct-only", "note": "use provider URL directly"})
            continue
        try:
            r = requests.get(f"{p['url']}/v1/models", timeout=3)
            up = r.status_code < 500
        except requests.exceptions.RequestException:
            up = False
        any_up = any_up or up
        report.append({"name": p["name"], "up": up})
    return jsonify({"status": "ok" if any_up else "degraded", "providers": report}), 200


def validate_chat_body(body):
    if not isinstance(body, dict):
        return "JSON object expected"
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "'messages' must be a non-empty list"
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return "each message needs 'role' and 'content'"
    if "max_tokens" in body and (not isinstance(body["max_tokens"], int) or body["max_tokens"] < 1):
        return "'max_tokens' must be a positive integer"
    return None


def upstream_request(base_url, method, path, body=None, params=None, stream=False):
    for attempt in range(UPSTREAM_RETRIES + 1):
        try:
            resp = requests.request(
                method,
                f"{base_url}{path}",
                json=body,
                params=params,
                headers={"Content-Type": "application/json"},
                stream=stream,
                timeout=(10, UPSTREAM_TIMEOUT),
            )
            if resp.status_code in (502, 503, 504) and attempt < UPSTREAM_RETRIES:
                log.warning("%s: upstream %s, retrying", base_url, resp.status_code)
                resp.close()
                time.sleep(UPSTREAM_RETRY_BACKOFF * (2 ** attempt))
                continue
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            log.warning("%s: %s (attempt %d)", base_url, type(e).__name__, attempt + 1)
            if attempt < UPSTREAM_RETRIES:
                time.sleep(UPSTREAM_RETRY_BACKOFF * (2 ** attempt))
                continue
            raise


def failover_post(path, body, stream):
    model = body.get("model") or DEFAULT_MODEL
    body["model"] = model
    candidates = providers_for_model(model)
    if not candidates:
        return None, openai_error(
            f"Unknown model '{model}'. Available: {sorted(MODEL_INDEX.keys())}", 400, "invalid_request_error")

    errors = []
    for p in candidates:
        if p.get("direct_only"):
            errors.append(f"{p['name']}: direct-only (use its own URL, not this gateway)")
            continue
        try:
            resp = upstream_request(p["url"], "POST", path, body=body, stream=stream)
            if stream and resp.status_code == 200:
                return (p, resp), None
            if resp.status_code in (502, 503, 504):
                errors.append(f"{p['name']}: HTTP {resp.status_code}")
                continue
            return (p, resp), None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            errors.append(f"{p['name']}: {type(e).__name__}")
            continue
    return None, openai_error("All providers failed for this model — " + "; ".join(errors), 502, "upstream_error")


def proxy(path, inject_default_model=False, validate=None):
    body = request.get_json(silent=True)
    if body is None:
        return openai_error("Invalid or missing JSON body", 400, "invalid_request_error")
    if validate:
        err = validate(body)
        if err:
            return openai_error(err, 400, "invalid_request_error")
    if inject_default_model and not body.get("model"):
        body["model"] = DEFAULT_MODEL

    result, err = failover_post(path, body, stream=bool(body.get("stream")))
    if err:
        return err
    provider, upstream = result

    try:
        if body.get("stream"):
            def generate():
                try:
                    for chunk in upstream.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                finally:
                    upstream.close()

            return Response(
                stream_with_context(generate()),
                status=200,
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        content = upstream.content
        try:
            return jsonify(json.loads(content)), upstream.status_code
        except (ValueError, UnicodeDecodeError):
            return Response(content, status=upstream.status_code,
                            content_type=upstream.headers.get("Content-Type", "application/json"))
    except requests.exceptions.RequestException as e:
        log.warning("Stream failed: %s", e)
        return openai_error("Upstream request failed", 502, "upstream_error")


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    return proxy("/v1/chat/completions", inject_default_model=True, validate=validate_chat_body)


@app.route("/v1/completions", methods=["POST"])
def completions():
    return proxy("/v1/completions", inject_default_model=True)


@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    return proxy("/v1/embeddings", inject_default_model=True)


@app.route("/v1/models", methods=["GET"])
def models():
    merged = []
    seen = set()
    provider_status = []
    for p in PROVIDERS:
        if p.get("direct_only"):
            provider_status.append({"name": p["name"], "up": "direct-only", "models": len(p["models"])})
            for mid in p["models"]:
                if mid in seen:
                    continue
                seen.add(mid)
                merged.append({
                    "id": mid, "object": "model", "owned_by": p["name"],
                    "provider": p["name"], "status": "direct-only — use provider URL directly",
                })
            continue
        try:
            r = requests.get(f"{p['url']}/v1/models", timeout=5)
            data = r.json()
            live = data.get("data", []) if isinstance(data, dict) else []
            provider_status.append({"name": p["name"], "up": True, "models": len(live)})
            refresh_index_for(p, [m.get("id") for m in live if m.get("id")])
            for m in live:
                mid = m.get("id")
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                m = dict(m)
                m["provider"] = p["name"]
                merged.append(m)
        except (requests.exceptions.RequestException, ValueError) as e:
            provider_status.append({"name": p["name"], "up": False, "models": 0})
            log.warning("Provider %s down for /v1/models: %s", p["name"], e)
            for mid in p["models"]:
                if mid in seen:
                    continue
                seen.add(mid)
                merged.append({
                    "id": mid, "object": "model", "owned_by": p["name"],
                    "provider": p["name"], "status": "provider-down",
                })
    return jsonify({
        "object": "list",
        "data": merged,
        "providers": provider_status,
    })


@app.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def catch_all(subpath):
    try:
        body = request.get_json(silent=True)
        model = (body or {}).get("model") or request.args.get("model") or DEFAULT_MODEL
        candidates = providers_for_model(model) or PROVIDERS
        last_err = None
        for p in candidates:
            try:
                r = upstream_request(p["url"], request.method, f"/v1/{subpath}", body=body, params=request.args)
                return Response(r.content, status=r.status_code,
                                content_type=r.headers.get("Content-Type", "application/json"))
            except requests.exceptions.RequestException as e:
                last_err = e
                continue
        raise last_err or requests.exceptions.ConnectionError("no provider")
    except requests.exceptions.Timeout:
        return openai_error("Upstream model server timed out", 504, "timeout_error")
    except requests.exceptions.RequestException:
        return openai_error("Upstream request failed", 502, "upstream_error")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", CFG["port"]))
    app.run(host="0.0.0.0", port=port, threaded=True)
