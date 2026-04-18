// Simtool renderer — minimal vanilla JS, one view per section.
//
// The backend port is passed via the file:// URL's `api` query param
// (Electron main process sets this). We derive the base URL and use
// fetch() with absolute URLs throughout.

const API = (() => {
  try {
    const u = new URL(window.location.href)
    const api = u.searchParams.get('api')
    if (api) return api
  } catch (err) { /* noop */ }
  return 'http://127.0.0.1:8000'
})()

async function api (path, opts) {
  const resp = await fetch(`${API}${path}`, opts)
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`)
  }
  const ct = resp.headers.get('content-type') || ''
  if (ct.includes('application/json')) return resp.json()
  return resp.text()
}

// ---------- navigation ----------

const views = {}
for (const section of document.querySelectorAll('.view')) {
  views[section.id.replace('view-', '')] = section
}
for (const btn of document.querySelectorAll('.nav-item')) {
  btn.addEventListener('click', () => switchView(btn.dataset.view))
}

const viewLoaded = {}
function switchView (name) {
  for (const s of Object.values(views)) s.classList.remove('active')
  views[name].classList.add('active')
  for (const b of document.querySelectorAll('.nav-item')) {
    b.classList.toggle('active', b.dataset.view === name)
  }
  if (!viewLoaded[name]) { viewLoaded[name] = true; loaders[name]?.() }
}

// ---------- health / status ----------

async function pollHealth () {
  try {
    const h = await api('/api/health')
    const el = document.getElementById('backend-status')
    el.textContent = h.ok
      ? `✓ backend · ${h.n_reconciled_params} params`
      : '✗ backend error'
    el.style.color = h.ok ? 'var(--green)' : 'var(--red)'
  } catch (err) {
    const el = document.getElementById('backend-status')
    el.textContent = `✗ ${err.message.slice(0, 60)}`
    el.style.color = 'var(--red)'
  }
}

// ---------- metamodel view ----------

let allParams = []

async function loadMetamodel () {
  const [mm, params] = await Promise.all([
    api('/api/metamodel'),
    api('/api/metamodel/parameters')
  ])
  allParams = params
  // stats
  document.getElementById('stat-params').textContent = params.length
  const dois = new Set()
  let conflicts = 0, highQ = 0
  for (const p of params) {
    for (const d of (p.supporting_record_dois || [])) dois.add(d)
    if ((p.conflict_flags || []).length) conflicts++
    if (p.quality_rating === 'high') highQ++
  }
  document.getElementById('stat-dois').textContent = dois.size
  document.getElementById('stat-conflicts').textContent = conflicts
  document.getElementById('stat-quality').textContent = highQ
  // overview plot
  const img = document.getElementById('plot-overview')
  img.src = `${API}/api/metamodel/plot/overview`
  img.onerror = () => {
    img.replaceWith(Object.assign(document.createElement('div'), {
      className: 'muted',
      textContent: 'Overview picture not generated yet. Run the plot-generation script (see README).'
    }))
  }
  renderMetamodelTable()
}

function renderMetamodelTable () {
  const q = document.getElementById('mm-search').value.toLowerCase()
  const qf = document.getElementById('mm-quality-filter').value
  const co = document.getElementById('mm-conflict-only').checked
  const tbody = document.querySelector('#mm-table tbody')
  tbody.innerHTML = ''
  const filtered = allParams.filter(p => {
    if (qf && p.quality_rating !== qf) return false
    if (co && !(p.conflict_flags || []).length) return false
    if (!q) return true
    const hay = [
      p.parameter_id,
      p.canonical_unit,
      ...Object.entries(p.context_keys || {}).map(([k, v]) => `${k}=${v}`),
    ].join(' ').toLowerCase()
    return hay.includes(q)
  })
  for (const p of filtered) {
    const tr = document.createElement('tr')
    const ctx = p.context_keys || {}
    const valDesc = p.point_estimate != null
      ? fmtNum(p.point_estimate)
      : (p.samples ? `n=${p.samples.length} (${fmtNum(Math.min(...p.samples))}…${fmtNum(Math.max(...p.samples))})` : '—')
    tr.innerHTML = `
      <td class="mono">${escHtml(p.parameter_id)}</td>
      <td>${escHtml(ctx.species || '—')}</td>
      <td>${escHtml(ctx.substrate || '—')}</td>
      <td class="num">${escHtml(valDesc)}</td>
      <td class="mono">${escHtml(p.canonical_unit)}</td>
      <td><span class="pill ${p.quality_rating}">${escHtml(p.quality_rating)}</span></td>
      <td class="num">${(p.supporting_record_dois || []).length}</td>
      <td>${(p.conflict_flags || []).length ? '<span class="conflict-flag">⚠ ' + escHtml((p.conflict_flags || []).join('; ')) + '</span>' : '—'}</td>
    `
    tbody.appendChild(tr)
  }
  document.getElementById('mm-filter-hint').textContent = `(${filtered.length} of ${allParams.length})`
}

document.getElementById('mm-search').addEventListener('input', renderMetamodelTable)
document.getElementById('mm-quality-filter').addEventListener('change', renderMetamodelTable)
document.getElementById('mm-conflict-only').addEventListener('change', renderMetamodelTable)

// ---------- data view ----------

let selectedDoi = null
async function loadData () {
  const papers = await api('/api/data/papers')
  const ul = document.getElementById('paper-list')
  ul.innerHTML = ''
  for (const p of papers) {
    const li = document.createElement('li')
    li.dataset.doi = p.doi
    li.innerHTML = `<div class="doi">${escHtml(p.doi)}</div><div class="count">${p.extraction_count} extractions</div>`
    li.addEventListener('click', () => selectDoi(p.doi))
    ul.appendChild(li)
  }
  if (papers.length) selectDoi(papers[0].doi)
}

async function selectDoi (doi) {
  selectedDoi = doi
  for (const li of document.querySelectorAll('#paper-list li')) {
    li.classList.toggle('active', li.dataset.doi === doi)
  }
  document.getElementById('data-selected-doi').textContent = `— ${doi}`
  const rows = await api(`/api/data/extractions?doi=${encodeURIComponent(doi)}`)
  const tbody = document.querySelector('#data-table tbody')
  tbody.innerHTML = ''
  for (const r of rows) {
    const tr = document.createElement('tr')
    const ctx = r.context || {}
    tr.innerHTML = `
      <td class="mono">${escHtml(r.parameter_id)}</td>
      <td class="num">${fmtNum(r.value)}</td>
      <td class="mono">${escHtml(r.unit)}</td>
      <td>${escHtml(ctx.species || '—')}</td>
      <td>${escHtml(ctx.substrate || '—')}</td>
      <td class="mono">${escHtml(r.method)}</td>
      <td class="num">${(r.span || {}).page ?? '—'}</td>
      <td style="max-width:480px;"><span title="${escAttr((r.span||{}).text_excerpt||'')}">${escHtml(((r.span||{}).text_excerpt||'').slice(0, 140))}${((r.span||{}).text_excerpt||'').length > 140 ? '…' : ''}</span></td>
    `
    tbody.appendChild(tr)
  }
}

// ---------- search view ----------

document.getElementById('search-go').addEventListener('click', runSearch)
document.getElementById('search-query').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch() })

async function runSearch () {
  const q = document.getElementById('search-query').value.trim()
  if (!q) return
  const status = document.getElementById('search-status')
  const tbody = document.querySelector('#search-table tbody')
  status.textContent = 'Searching PMC…'
  tbody.innerHTML = ''
  try {
    const { results } = await api(`/api/papers/search?q=${encodeURIComponent(q)}&retmax=25`)
    status.textContent = `${results.length} results for "${q}"`
    for (const r of results) {
      const tr = document.createElement('tr')
      tr.innerHTML = `
        <td class="mono"><a href="https://pmc.ncbi.nlm.nih.gov/articles/${escAttr(r.pmc_id)}/" target="_blank" rel="noopener">${escHtml(r.pmc_id)}</a></td>
        <td>${escHtml(r.title)}</td>
        <td>${escHtml(r.journal || '—')}</td>
        <td class="num">${r.year || '—'}</td>
        <td>${escHtml(r.authors || '—')}</td>
      `
      tbody.appendChild(tr)
    }
  } catch (err) {
    status.textContent = `error: ${err.message}`
  }
}

// ---------- recommend view ----------

document.getElementById('recommend-go').addEventListener('click', async () => {
  const req = document.querySelectorAll('#req-phenomena input:checked')
  const exc = document.querySelectorAll('#exc-phenomena input:checked')
  const body = {
    required_phenomena: Array.from(req).map(i => i.value),
    excluded_phenomena: Array.from(exc).map(i => i.value),
    predictive_priorities: [],
    time_horizon_days: parseFloat(document.getElementById('horizon-days').value) || 5,
    compute_budget_wall_time_hours: parseFloat(document.getElementById('max-hours').value) || null,
    compute_budget_memory_gb: parseFloat(document.getElementById('max-gb').value) || null,
    note: ''
  }
  const panel = document.getElementById('recommend-output')
  panel.innerHTML = '<p class="muted">Computing…</p>'
  try {
    const res = await api('/api/workflows/recommend', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    panel.innerHTML = renderRecommendation(res, body)
  } catch (err) {
    panel.innerHTML = `<p class="conflict-flag">error: ${escHtml(err.message)}</p>`
  }
})

function renderRecommendation (res, req) {
  if (res.recommendation === null) {
    return `
      <h3>Nothing to recommend yet</h3>
      <div class="impl-block">
        <strong>Why:</strong> the meta-model has no submodel hierarchy populated — the recommender has no options to rank.
        Once submodels are seeded (e.g. 'well-mixed ODE', '1-D biofilm', '3-D agent-based'), requests like this return a
        concrete recommendation with its assumptions.
      </div>
    `
  }
  const r = res
  const assumptions = r.assumptions_introduced || []
  const unmet = r.unmet_constraints || []
  const reasoning = r.reasoning || []

  const implBlock = assumptions.length === 0 && unmet.length === 0
    ? `<div class="impl-ok"><strong>No implicit cost:</strong> the recommended submodel fits the requested constraints without simplification.</div>`
    : [
        assumptions.length ? `
          <div class="impl-block">
            <strong>Assumptions this simplification introduces:</strong>
            <ul>${assumptions.map(a => `<li>${escHtml(a)}</li>`).join('')}</ul>
          </div>` : '',
        unmet.length ? `
          <div class="impl-block">
            <strong>Constraints NOT fully met:</strong>
            <ul>${unmet.map(u => `<li>${escHtml(u)}</li>`).join('')}</ul>
          </div>` : '',
      ].join('')

  const trace = reasoning.length ? `
    <h4>Reasoning trace</h4>
    <ul>${reasoning.map(s => `<li><strong>${escHtml(s.kind)}</strong>${s.submodel_id ? ` <span class="muted">(${escHtml(s.submodel_id)})</span>` : ''} — ${escHtml(s.note)}</li>`).join('')}</ul>
  ` : ''

  return `
    <h3>Recommended submodel: <code>${escHtml(r.submodel_id || '—')}</code></h3>
    ${r.derived_via_operator_id ? `<p class="muted">Synthesized by approximation operator <code>${escHtml(r.derived_via_operator_id)}</code></p>` : ''}
    ${implBlock}
    ${trace}
  `
}

// ---------- adjust view ----------

document.getElementById('adjust-go').addEventListener('click', async () => {
  const kind = document.getElementById('adjust-kind').value
  const targetId = document.getElementById('adjust-target').value.trim()
  const userNote = document.getElementById('adjust-desc').value.trim()
  const req = document.querySelectorAll('#adjust-req input:checked')
  const spec = kind === 'change_scope'
    ? { required_phenomena: Array.from(req).map(i => i.value) }
    : (kind === 'add_process' || kind === 'add_entity')
      ? { required_parameter_ids: targetId ? [targetId] : [] }
      : {}
  const panel = document.getElementById('adjust-output')
  panel.innerHTML = '<p class="muted">Computing…</p>'
  try {
    const res = await api('/api/workflows/adjust', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, target_id: targetId, spec, user_note: userNote })
    })
    panel.innerHTML = renderAdjustment(res)
  } catch (err) {
    panel.innerHTML = `<p class="conflict-flag">error: ${escHtml(err.message)}</p>`
  }
})

function renderAdjustment (res) {
  const support = res.support_level || 'unknown'
  const pillClass = support === 'supported' ? 'succeeded'
    : support === 'speculative' || support === 'unsupported' ? 'failed'
    : 'running'
  const dependsP = res.depends_on_parameter_ids || []
  const dependsS = res.depends_on_submodel_ids || []
  const has = res.meta_model_has || []
  const missing = res.meta_model_missing || []
  const specDois = res.speculative_candidate_dois || []
  return `
    <h3>Support level: <span class="pill ${pillClass}">${escHtml(support)}</span></h3>
    ${res.reasoning ? `<p>${escHtml(res.reasoning)}</p>` : ''}
    ${has.length ? `
      <h4>The meta-model already has</h4>
      <ul>${has.map(d => `<li><code>${escHtml(d)}</code></li>`).join('')}</ul>
    ` : ''}
    ${missing.length ? `
      <div class="impl-block">
        <strong>Missing from the meta-model — this adjustment is speculative until filled:</strong>
        <ul>${missing.map(m => `<li><code>${escHtml(m)}</code></li>`).join('')}</ul>
      </div>
    ` : ''}
    ${(dependsP.length || dependsS.length) ? `
      <h4>Dependencies</h4>
      ${dependsP.length ? `<div>parameters: ${dependsP.map(x => `<code>${escHtml(x)}</code>`).join(', ')}</div>` : ''}
      ${dependsS.length ? `<div>submodels: ${dependsS.map(x => `<code>${escHtml(x)}</code>`).join(', ')}</div>` : ''}
    ` : ''}
    ${specDois.length ? `
      <div class="impl-block">
        <strong>Speculative support from candidate literature (not yet in meta-model):</strong>
        <ul>${specDois.map(d => `<li><code>${escHtml(d)}</code></li>`).join('')}</ul>
      </div>
    ` : ''}
  `
}

// ---------- simulate view ----------

let currentRun = null
let currentRunFramework = null
let pollTimer = null
let frameworks = []

document.getElementById('run-start').addEventListener('click', startRun)
document.getElementById('run-cancel').addEventListener('click', cancelRun)

async function loadFrameworks () {
  const sel = document.getElementById('framework-select')
  const status = document.getElementById('framework-status')
  try {
    const res = await api('/api/frameworks')
    frameworks = res.frameworks
    sel.innerHTML = ''
    for (const fw of frameworks) {
      const opt = document.createElement('option')
      opt.value = fw.id
      opt.textContent = fw.available
        ? `${fw.name}`
        : `${fw.name}  (not configured)`
      opt.disabled = !fw.available
      sel.appendChild(opt)
    }
    sel.dispatchEvent(new Event('change'))
  } catch (err) {
    status.textContent = `framework discovery failed: ${err.message}`
    status.style.color = 'var(--red)'
  }
}

document.getElementById('framework-select').addEventListener('change', () => {
  const id = document.getElementById('framework-select').value
  const fw = frameworks.find(f => f.id === id)
  const status = document.getElementById('framework-status')
  if (!fw) { status.textContent = ''; return }
  if (fw.available) {
    if (id === 'demo') {
      status.textContent = '✓ demo runtime — synthetic outputs in ~3s, useful for UI smoke testing'
      status.style.color = 'var(--green)'
    } else {
      const oks = (fw.checks || []).filter(c => c.ok).length
      const total = (fw.checks || []).length
      status.textContent = `✓ ${fw.name} — ${oks}/${total} runtime checks pass`
      status.style.color = 'var(--green)'
    }
  } else {
    const fail = (fw.checks || []).find(c => !c.ok)
    status.innerHTML = `<span class="conflict-flag">✗ ${escHtml(fw.name)} not available</span> — ${escHtml(fail ? fail.detail : 'see backend logs')}`
    status.style.color = ''
  }
})

async function startRun () {
  clearInterval(pollTimer)
  const framework = document.getElementById('framework-select').value
  const nSteps = parseInt(document.getElementById('run-steps').value, 10) || 12
  document.getElementById('run-start').disabled = true
  document.getElementById('run-cancel').disabled = false
  document.getElementById('progress-fill').style.width = '0%'
  document.getElementById('progress-line').textContent = 'Starting…'
  document.getElementById('live-observables').innerHTML = '—'
  document.getElementById('analysis-output').innerHTML = '<p class="muted">Running…</p>'
  document.getElementById('series-charts').innerHTML = ''
  setRunStatus('running')
  try {
    const res = await api('/api/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_steps: nSteps, framework })
    })
    currentRun = res.run_id
    currentRunFramework = res.framework_id
    document.getElementById('run-id-label').textContent = `run ${currentRun} · ${currentRunFramework}`
    pollTimer = setInterval(pollProgress, 400)
  } catch (err) {
    setRunStatus('failed')
    document.getElementById('progress-line').textContent = `error: ${err.message}`
    document.getElementById('run-start').disabled = false
    document.getElementById('run-cancel').disabled = true
  }
}

async function cancelRun () {
  if (!currentRun) return
  try { await api(`/api/runs/${currentRun}/cancel`, { method: 'POST' }) } catch { /* ignore */ }
}

async function pollProgress () {
  if (!currentRun) return
  let p
  try { p = await api(`/api/runs/${currentRun}/progress`) } catch { return }
  setRunStatus(p.status)
  const reports = p.reports || []
  if (reports.length) {
    const last = reports[reports.length - 1]
    let pct = 0
    if (last.timestep_total && last.timestep_index != null) {
      pct = 100 * last.timestep_index / last.timestep_total
    } else if (last.sim_time_horizon_s && last.sim_time_s != null) {
      pct = 100 * last.sim_time_s / last.sim_time_horizon_s
    }
    document.getElementById('progress-fill').style.width = `${Math.min(100, pct)}%`
    const stepLabel = (last.timestep_total != null && last.timestep_index != null)
      ? `step ${last.timestep_index}/${last.timestep_total}`
      : (last.sim_time_s != null ? `t=${fmtNum(last.sim_time_s)}s` : '')
    document.getElementById('progress-line').textContent =
      `${stepLabel} · ${last.message || ''}`
    const obs = last.observables || {}
    document.getElementById('live-observables').innerHTML = Object.entries(obs).map(([k, v]) => `
      <div class="obs"><div class="k">${escHtml(k)}</div><div class="v">${fmtNum(v)}</div></div>
    `).join('') || '—'
  }
  if (p.status === 'succeeded' || p.status === 'failed' || p.status === 'terminated') {
    clearInterval(pollTimer)
    pollTimer = null
    document.getElementById('run-start').disabled = false
    document.getElementById('run-cancel').disabled = true
    if (p.status === 'succeeded') loadAnalysis()
    else if (p.status === 'failed') {
      document.getElementById('analysis-output').innerHTML =
        '<p class="conflict-flag">Run failed — check the backend log for details.</p>'
    }
  }
}

async function loadAnalysis () {
  if (!currentRun) return
  try {
    const out = await api(`/api/runs/${currentRun}/output`)
    renderAnalysis(out)
  } catch (err) {
    document.getElementById('analysis-output').innerHTML = `<p class="conflict-flag">error: ${escHtml(err.message)}</p>`
  }
}

function renderAnalysis (out) {
  const findings = (out.analysis || {}).findings || []
  const panel = document.getElementById('analysis-output')
  if (findings.length === 0) {
    panel.innerHTML = '<p class="muted">Run complete — no matching observables for automated comparison.</p>'
  } else {
    panel.innerHTML = findings.map(f => {
      const ok = f.within_expected
      const block = f.expected_range
        ? (ok
            ? `<div class="impl-ok"><strong>${escHtml(f.observable)}</strong> tail-mean <code>${fmtNum(f.tail_mean)}</code> falls inside literature range <code>${fmtNum(f.expected_range.lo)}..${fmtNum(f.expected_range.hi)}</code></div>`
            : `<div class="impl-block"><strong>${escHtml(f.observable)}</strong> tail-mean <code>${fmtNum(f.tail_mean)}</code> is OUTSIDE literature range <code>${fmtNum(f.expected_range.lo)}..${fmtNum(f.expected_range.hi)}</code></div>`)
        : `<div class="impl-ok"><strong>${escHtml(f.observable)}</strong> tail-mean <code>${fmtNum(f.tail_mean)}</code> (no matching literature range)</div>`
      return block
    }).join('')
  }
  // charts
  const bundle = out.bundle || {}
  const series = bundle.scalar_time_series || {}
  const host = document.getElementById('series-charts')
  host.innerHTML = ''
  for (const [name, pts] of Object.entries(series)) {
    host.appendChild(renderChart(name, pts))
  }
}

function renderChart (name, pts) {
  const w = 360, h = 120, pad = 18
  const xs = pts.map(p => p[0])
  const ys = pts.map(p => p[1])
  const xmin = Math.min(...xs), xmax = Math.max(...xs)
  const ymin = Math.min(...ys), ymax = Math.max(...ys)
  const span = (a, b) => (b - a) === 0 ? 1 : (b - a)
  const xp = x => pad + ((x - xmin) / span(xmin, xmax)) * (w - 2 * pad)
  const yp = y => h - pad - ((y - ymin) / span(ymin, ymax)) * (h - 2 * pad)
  const path = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xp(p[0]).toFixed(1)},${yp(p[1]).toFixed(1)}`).join(' ')
  const card = document.createElement('div')
  card.className = 'series-chart'
  card.innerHTML = `
    <h4>${escHtml(name)}</h4>
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line class="axis" x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}"></line>
      <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}"></line>
      <path class="line" d="${path}"></path>
    </svg>
    <div class="muted" style="font-size:11px; margin-top:4px;">
      range: ${fmtNum(ymin)}…${fmtNum(ymax)} · end: ${fmtNum(ys[ys.length - 1])}
    </div>
  `
  return card
}

function setRunStatus (s) {
  const el = document.getElementById('run-status')
  el.textContent = s
  el.className = `pill ${s}`
}

// ---------- utilities ----------

function fmtNum (v) {
  if (v == null || Number.isNaN(v)) return '—'
  if (typeof v !== 'number') return String(v)
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1000 || a < 0.01) return v.toExponential(2)
  return Number(v.toFixed(Math.max(0, 4 - Math.floor(Math.log10(a))))).toString()
}
function escHtml (s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])) }
function escAttr (s) { return escHtml(s) }

// ---------- load order ----------

const loaders = {
  metamodel: loadMetamodel,
  data: loadData,
  simulate: loadFrameworks,
}

pollHealth()
setInterval(pollHealth, 5000)
loadMetamodel()   // preload first view
loadFrameworks()  // populate framework selector eagerly so it's ready
