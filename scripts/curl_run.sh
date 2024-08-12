#!/bin/bash

port=8499

curl -X POST \
     -H "Content-Type: application/json" \
     -d @./scripts/req.json \
     http://localhost:$port/generate