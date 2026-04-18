// Simtool desktop app — Electron main process.
//
// Responsibilities:
//   1. Spawn the Python backend (`python -m simtool.api.server`) as a child
//      process. The backend prints `SIMTOOL_API_PORT=<n>` on startup; we
//      parse that line to learn the local port.
//   2. Create a BrowserWindow that loads the backend's static renderer
//      mount at `/ui/` on that port. No CORS, no IPC bridge needed — the
//      renderer talks to the same origin it was served from.
//   3. Forward Python stdout/stderr to the Electron console for debugging.
//   4. Kill the backend when the app quits.

const { app, BrowserWindow, shell, Menu } = require('electron')
const { spawn } = require('node:child_process')
const path = require('node:path')
const fs = require('node:fs')

const REPO_ROOT = path.resolve(__dirname, '..')
const RENDERER_INDEX = path.join(__dirname, 'renderer', 'index.html')

let backend = null
let backendPort = null
let win = null

function resolvePython () {
  // Prefer the project's venv python so we get the installed deps.
  const venvPy = path.join(REPO_ROOT, '.venv', 'bin', 'python')
  if (fs.existsSync(venvPy)) return venvPy
  return process.env.PYTHON || 'python3'
}

function startBackend () {
  return new Promise((resolve, reject) => {
    const py = resolvePython()
    console.log(`[main] starting backend: ${py} -m simtool.api.server`)
    backend = spawn(py, ['-m', 'simtool.api.server'], {
      cwd: REPO_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: ['ignore', 'pipe', 'pipe']
    })

    let resolved = false
    const killTimer = setTimeout(() => {
      if (!resolved) reject(new Error('backend did not report a port within 15s'))
    }, 15000)

    backend.stdout.on('data', (buf) => {
      const text = buf.toString()
      process.stdout.write(`[py-out] ${text}`)
      if (!resolved) {
        const m = text.match(/SIMTOOL_API_PORT=(\d+)/)
        if (m) {
          resolved = true
          backendPort = parseInt(m[1], 10)
          clearTimeout(killTimer)
          resolve(backendPort)
        }
      }
    })
    backend.stderr.on('data', (buf) => {
      process.stderr.write(`[py-err] ${buf.toString()}`)
    })
    backend.on('exit', (code, signal) => {
      console.log(`[main] backend exited code=${code} signal=${signal}`)
      backend = null
      if (!resolved) {
        clearTimeout(killTimer)
        reject(new Error(`backend exited before reporting port (code=${code})`))
      }
    })
  })
}

function stopBackend () {
  if (backend) {
    console.log('[main] killing backend')
    try { backend.kill('SIGTERM') } catch (err) { /* noop */ }
    backend = null
  }
}

function buildMenu () {
  const template = [
    { role: 'appMenu' },
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    { role: 'windowMenu' },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Open backend API (browser)',
          click: () => {
            if (backendPort) shell.openExternal(`http://127.0.0.1:${backendPort}/docs`)
          }
        }
      ]
    }
  ]
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

async function createWindow () {
  win = new BrowserWindow({
    width: 1280,
    height: 820,
    title: 'Simtool',
    backgroundColor: '#0b1020',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  // The renderer is a plain file. It discovers the backend's port via
  // `window.SIMTOOL_API_BASE` (injected at load time below).
  const baseUrl = `http://127.0.0.1:${backendPort}`
  await win.loadFile(RENDERER_INDEX, { query: { api: baseUrl } })
  // In dev it's sometimes convenient to auto-open DevTools — keep closed by default.
}

app.on('ready', async () => {
  try {
    await startBackend()
  } catch (err) {
    console.error('[main] failed to start backend:', err.message)
    app.quit()
    return
  }
  buildMenu()
  await createWindow()
})

app.on('window-all-closed', () => {
  stopBackend()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', stopBackend)

app.on('activate', async () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    await createWindow()
  }
})
