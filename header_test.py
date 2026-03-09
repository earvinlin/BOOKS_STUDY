#!/usr/bin/env python3
import requests
import json
from datetime import datetime

SECURITY_HEADERS = [
    "Content-Security-Policy",
    "Permissions-Policy",
    "Strict-Transport-Security",
    "X-Frame-Options",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Cross-Origin-Opener-Policy",
    "Cross-Origin-Resource-Policy",
    "Cross-Origin-Embedder-Policy"
]

def scan(url: str):
    print(f"Scanning: {url}")
    r = requests.get(url, timeout=10)
    headers = r.headers

    result = {
        "url": url,
        "timestamp": datetime.utcnow().isoformat(),
        "headers": {}
    }

    for h in SECURITY_HEADERS:
        result["headers"][h] = headers.get(h, None)

    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scan.py <url>")
        exit(1)

    scan(sys.argv[1])
