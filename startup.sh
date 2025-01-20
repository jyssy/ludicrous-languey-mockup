#!/bin/bash

# Create and start up
docker-compose -f llama_compose.yml pull
docker-compose -f llama_compose.yml build
docker-compose -f llama_compose.yml up -d --remove-orphans