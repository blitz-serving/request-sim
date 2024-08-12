#!/bin/bash

curl -X POST -H "Content-Type: application/json" -d '{"tokens":[1,2,3,4], "max_tokens":10}' http://localhost:5060