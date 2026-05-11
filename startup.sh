#!/bin/bash
# Azure App Service startup script
# PORT is set automatically by Azure (default 8000)
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
