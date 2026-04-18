"""FastAPI backend for the Simtool desktop app.

The Electron main process launches `uvicorn simtool.api.server:app --port 0`
and reads the assigned port from stdout. Backend is strictly local — binds
127.0.0.1 only; no cross-origin access needed.
"""
