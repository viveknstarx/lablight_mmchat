#!/bin/sh
uvicorn project.Main.api:app --host 0.0.0.0 --port 8000 &
ollama serve