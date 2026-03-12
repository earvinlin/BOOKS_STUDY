#!/usr/bin/env bash

URL="$1"

if [ -z "$URL" ]; then
  echo "Usage: $0 <url>"
  exit 1
fi

echo "Scanning security headers for: $URL"
echo "-----------------------------------"

curl -s -D - "$URL" -o /dev/null | grep -Ei \
"content-security-policy|permissions-policy|strict-transport-security|x-frame-options|x-content-type-options|referrer-policy|cross-origin-opener-policy|cross-origin-resource-policy|cross-origin-embedder-policy"
