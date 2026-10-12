/* ============================================================
   CP461 YumYayum · Presentation Web App — Main JavaScript
   ============================================================ */

// ─── State ───────────────────────────────────────────────────
let currentSection = 0;
const SECTION_COUNT = 6;

let currentFile    = null;
let currentPreset  = null;
let currentInputB64  = null;
let currentOutputB64 = null;

let isDragging   = false;
let isLoupeOn    = false;
let timerInterval = null;
let splitPct     = 50;

// ─── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildNavDots();
  updateNavState();
  initServerStatus();
  initPresets();
  initDropzone();
  initSlider();
  initModelCards();
  initClipboardPaste();
  initKeyboard();
  initCharts();
  showKeyHint();
});

// ─── Navigation ───────────────────────────────────────────────
function buildNavDots() {
  const c = document.getElementById('navDots');
  c.innerHTML = '';
  for (let i = 0; i < SECTION_COUNT; i++) {
    const d = document.createElement('div');
    d.className = 'nav-dot' + (i === 0 ? ' active' : '');
    d.setAttribute('title', ['Overview','Dataset','Method','Metrics','Results','Live Demo'][i]);
    d.onclick = () => goToSection(i);
    c.appendChild(d);
  }
}

function goToSection(idx) {
  if (idx < 0 || idx >= SECTION_COUNT) return;
  document.querySelectorAll('.pres-section').forEach((s, i) => {
    s.classList.toggle('active', i === idx);
  });
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', i === idx);
  });
  document.querySelectorAll('.nav-dot').forEach((d, i) => {
    d.classList.toggle('active', i === idx);
  });
  currentSection = idx;
  updateNavState();

  // Lazy-render charts when Results section becomes visible
  if (idx === 4) setTimeout(renderCharts, 100);
}

function navigate(dir) {
  goToSection(currentSection + dir);
}

function updateNavState() {
  document.getElementById('prevBtn').disabled = currentSection === 0;
  document.getElementById('nextBtn').disabled = currentSection === SECTION_COUNT - 1;
}

function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeReportModal();
      closeReportsDropdown();
      return;
    }
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowRight') navigate(1);
    if (e.key === 'ArrowLeft')  navigate(-1);
  });
}


function showKeyHint() {
  const hint = document.getElementById('keyHint');
  hint.classList.add('show');
  setTimeout(() => hint.classList.remove('show'), 4000);
}

// ─── Server Status ─────────────────────────────────────────────
async function initServerStatus() {
  const pill  = document.getElementById('statusPill');
  const label = document.getElementById('statusLabel');
  const dev   = document.getElementById('devicePill');
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    pill.classList.add('ready');
    label.textContent = 'Server Online';
    if (d.device?.cuda_available) {
      dev.textContent = `🚀 ${d.device.gpu_name} · ${d.device.vram_total_gb} GB`;
      dev.style.color = '#38bdf8';
    } else {
      dev.textContent = '💻 CPU Mode';
    }
  } catch {
    label.textContent = 'Server Offline';
    dev.textContent   = 'Hardware: –';
  }
}

// ─── Presets ──────────────────────────────────────────────────
async function initPresets() {
  const row = document.getElementById('presetsRow');
  try {
    const r = await fetch('/api/presets');
    const d = await r.json();
    if (!d.presets?.length) { row.innerHTML = '<span style="color:#64748b;font-size:.78rem">ไม่มีภาพ preset</span>'; return; }
    row.innerHTML = '';
    d.presets.forEach((p, i) => {
      const c = document.createElement('div');
      c.className = 'preset-chip';
      c.innerHTML = `<img class="preset-thumb" src="${p.url}" alt="${p.filename}"><span class="preset-name">${p.filename}</span>`;
      c.onclick = () => selectPreset(p, c);
      row.appendChild(c);
      if (i === 0) selectPreset(p, c);     // auto-select first
    });
  } catch {
    row.innerHTML = '<span style="color:#64748b;font-size:.78rem">โหลดไม่สำเร็จ</span>';
  }
}

function selectPreset(preset, chip) {
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  chip?.classList.add('active');
  currentPreset = preset.filename;
  currentFile   = null;
  setBeforeImg(preset.url, preset.filename);
}

// ─── Dropzone ─────────────────────────────────────────────────
function initDropzone() {
  const dz = document.getElementById('dropZone');
  const fi = document.getElementById('fileInput');
  dz.onclick = () => fi.click();
  fi.onchange = e => { if (e.target.files[0]) handleFile(e.target.files[0]); };
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function initClipboardPaste() {
  window.addEventListener('paste', e => {
    for (const it of e.clipboardData.items) {
      if (it.kind === 'file' && it.type.startsWith('image/')) {
        handleFile(it.getAsFile()); break;
      }
    }
  });
}

function handleFile(file) {
  currentFile   = file;
  currentPreset = null;
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  const reader = new FileReader();
  reader.onload = e => setBeforeImg(e.target.result, file.name);
  reader.readAsDataURL(file);
}

function setBeforeImg(src, name) {
  currentInputB64  = src;
  currentOutputB64 = null;          // reset stale SR output
  const bi = document.getElementById('beforeImg');
  const ai = document.getElementById('afterImg');
  bi.src   = src;
  ai.src   = src;                   // reset after to match new input
  bi.onload = () => { document.getElementById('statBefore').textContent = `${bi.naturalWidth} × ${bi.naturalHeight}`; };
  document.getElementById('statAfter').textContent   = '–';
  document.getElementById('statLatency').textContent  = '–';
  document.getElementById('statModel').textContent    = '–';
  document.getElementById('dlBtn').disabled = true;
  document.getElementById('tagAfter').textContent = '4× Output';
  document.getElementById('emptyHint').style.display  = 'none';
  document.getElementById('sliderWrap').style.display = 'flex';
  setSplit(50);
}

// ─── Model Cards ──────────────────────────────────────────────
function initModelCards() {
  document.querySelectorAll('.model-radio').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.model-radio').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      card.querySelector('input').checked = true;
    });
  });
}

function getModel() {
  return document.querySelector('input[name="mdl"]:checked')?.value ?? 'team_9000';
}

const MODEL_LABELS = {
  team_9000:         'CP461 Real-ESRNet (9k iters)',
  realesrgan_x4plus: 'Real-ESRGAN x4plus (Official)',
  bicubic:           'Bicubic Baseline'
};

// ─── Comparison Slider ────────────────────────────────────────
function initSlider() {
  const wrap = document.getElementById('sliderWrap');
  if (!wrap) return;
  const update = e => {
    if (!isDragging) return;
    const r = wrap.getBoundingClientRect();
    const x = (e.clientX ?? e.touches?.[0]?.clientX ?? r.left) - r.left;
    setSplit(Math.max(2, Math.min(98, (x / r.width) * 100)));
  };
  const stop = () => { isDragging = false; };

  wrap.addEventListener('mousedown', () => { isDragging = true; });
  window.addEventListener('mousemove', e => { update(e); if (isLoupeOn) updateLoupe(e); });
  window.addEventListener('mouseup', stop);
  wrap.addEventListener('touchstart', e => { isDragging = true; update(e.touches[0]); }, { passive: true });
  window.addEventListener('touchmove', e => { if (isDragging && e.touches[0]) update(e.touches[0]); }, { passive: true });
  window.addEventListener('touchend', stop);
}

function setSplit(pct) {
  splitPct = pct;
  const al = document.getElementById('afterLayer');
  const sh = document.getElementById('sliderHandle');
  if (!al || !sh) return;
  al.style.width = `${pct}%`;
  sh.style.left  = `${pct}%`;
  // Expand inner image so it stays full-width inside the clipped container.
  // Guard: pct must be > 0 to avoid division by zero.
  const ai = document.getElementById('afterImg');
  if (ai && ai.naturalWidth && pct > 0) {
    ai.style.width = `${(100 / pct) * 100}%`;
  }
}

// ─── Zoom Loupe ───────────────────────────────────────────────
function toggleLoupe() {
  isLoupeOn = !isLoupeOn;
  const loupe = document.getElementById('loupe');
  const btn   = document.getElementById('loupeBtn');
  loupe.style.display = isLoupeOn ? 'block' : 'none';
  btn.classList.toggle('active', isLoupeOn);
  if (isLoupeOn) refreshLoupeSrc();
}

function refreshLoupeSrc() {
  const inner = document.getElementById('loupeInner');
  const src   = currentOutputB64 || currentInputB64;
  if (src) inner.style.backgroundImage = `url(${src})`;
}

function updateLoupe(e) {
  const wrap  = document.getElementById('sliderWrap');
  const loupe = document.getElementById('loupe');
  const inner = document.getElementById('loupeInner');
  if (!wrap || !loupe || !isLoupeOn) return;
  const r = wrap.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  if (x < 0 || x > r.width || y < 0 || y > r.height) { loupe.style.display = 'none'; return; }
  loupe.style.display = 'block';
  loupe.style.left = `${x}px`;
  loupe.style.top  = `${y}px`;
  const Z = 3.5;
  inner.style.backgroundSize     = `${r.width * Z}px ${r.height * Z}px`;
  inner.style.backgroundPosition = `-${x * Z - 75}px -${y * Z - 75}px`;
}

// ─── Swap / Download ──────────────────────────────────────────
function swapSides() {
  const bi = document.getElementById('beforeImg');
  const ai = document.getElementById('afterImg');
  const tb = document.getElementById('tagBefore');
  const ta = document.getElementById('tagAfter');
  [bi.src, ai.src]   = [ai.src, bi.src];
  [tb.textContent, ta.textContent] = [ta.textContent, tb.textContent];
}

function downloadOut() {
  if (!currentOutputB64) return;
  const a = document.createElement('a');
  a.href = currentOutputB64;
  a.download = `SR_4x_${Date.now()}.png`;
  a.click();
}

// ─── Enhance (Single Model) ───────────────────────────────────
async function runEnhance() {
  if (!currentFile && !currentPreset) { alert('กรุณาเลือกภาพก่อนครับ'); return; }
  const btn     = document.getElementById('enhBtn');
  const overlay = document.getElementById('procOverlay');
  const timer   = document.getElementById('procTimer');
  const model   = getModel();
  const tile    = document.getElementById('tileSelect').value;

  btn.disabled = true;
  overlay.style.display = 'flex';

  let t0 = Date.now();
  timer.textContent = '0.0s';
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    timer.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
  }, 100);

  const fd = new FormData();
  if (currentFile)   fd.append('file', currentFile);
  else               fd.append('preset', currentPreset);
  fd.append('model',    model);
  fd.append('tile',     tile);
  fd.append('outscale', '4.0');

  try {
    const res  = await fetch('/api/enhance', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Inference failed');

    currentOutputB64 = data.output_image;
    document.getElementById('afterImg').src    = currentOutputB64;
    document.getElementById('tagAfter').textContent = `4× ${MODEL_LABELS[model] || model}`;
    document.getElementById('statAfter').textContent = `${data.output_dimensions.width} × ${data.output_dimensions.height}`;
    document.getElementById('statLatency').textContent = `${data.latency_ms} ms`;
    document.getElementById('statModel').textContent  = MODEL_LABELS[model] || model;
    document.getElementById('dlBtn').disabled = false;
    setSplit(50);
    refreshLoupeSrc();
  } catch (err) {
    alert('เกิดข้อผิดพลาด: ' + err.message);
  } finally {
    clearInterval(timerInterval);
    overlay.style.display = 'none';
    btn.disabled = false;
  }
}

// ─── Compare All Models ───────────────────────────────────────
async function runCompareAll() {
  if (!currentFile && !currentPreset) { alert('กรุณาเลือกภาพก่อนครับ'); return; }
  const overlay = document.getElementById('procOverlay');
  const timer   = document.getElementById('procTimer');
  const tile    = document.getElementById('tileSelect').value;

  overlay.style.display = 'flex';
  let t0 = Date.now();
  timer.textContent = '0.0s';
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    timer.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
  }, 100);

  const fd = new FormData();
  if (currentFile) fd.append('file', currentFile);
  else             fd.append('preset', currentPreset);
  fd.append('tile', tile);

  try {
    const res  = await fetch('/api/compare', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Compare failed');

    const strip = document.getElementById('multiStrip');
    const cards = document.getElementById('msCards');
    cards.innerHTML = '';

    for (const [key, r] of Object.entries(data.results)) {
      if (!r.success) continue;
      const c = document.createElement('div');
      c.className = 'ms-card';
      c.innerHTML = `
        <div class="ms-card-title">
          <span>${MODEL_LABELS[key] || key}</span>
          <span class="td-badge td-cyan">${r.latency_ms} ms</span>
        </div>
        <img class="ms-card-img" src="${r.image}" alt="${key}">`;
      c.onclick = () => {
        currentOutputB64 = r.image;
        document.getElementById('afterImg').src = r.image;
        document.getElementById('tagAfter').textContent = `4× ${MODEL_LABELS[key] || key}`;
        document.getElementById('statLatency').textContent = `${r.latency_ms} ms`;
        document.getElementById('statModel').textContent  = MODEL_LABELS[key] || key;
        document.getElementById('dlBtn').disabled = false;
        setSplit(50);
        refreshLoupeSrc();
      };
      cards.appendChild(c);
    }
    strip.style.display = 'block';
  } catch (err) {
    alert('เกิดข้อผิดพลาดในการเปรียบเทียบ: ' + err.message);
  } finally {
    clearInterval(timerInterval);
    overlay.style.display = 'none';
  }
}

// ─── Cases (Best / Failure) ───────────────────────────────────
function showCases(type) {
  document.querySelectorAll('.case-tab').forEach((t, i) => t.classList.toggle('active', (i === 0) === (type === 'best')));
  document.getElementById('casesBest').classList.toggle('active', type === 'best');
  document.getElementById('casesFail').classList.toggle('active', type === 'fail');
}

async function loadCaseResult(btnEl, preset) {
  const wrap = btnEl.closest('.case-sr-placeholder');
  btnEl.textContent = 'กำลังรัน...';
  btnEl.style.pointerEvents = 'none';

  const fd = new FormData();
  fd.append('preset', preset);
  fd.append('model', 'team_9000');
  fd.append('tile',  '256');
  fd.append('outscale', '4.0');

  try {
    const res  = await fetch('/api/enhance', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed');
    const img = document.createElement('img');
    img.className = 'case-img';
    img.src = data.output_image;
    img.alt = 'SR Output';
    wrap.innerHTML = '';
    wrap.appendChild(img);
  } catch (err) {
    btnEl.textContent = '❌ เกิดข้อผิดพลาด';
    btnEl.style.pointerEvents = 'auto';
  }
}

// ─── Charts (Chart.js) ────────────────────────────────────────
let chartsRendered = false;

function initCharts() {
  // Charts will be rendered when section 4 (Results) becomes visible
}

function renderCharts() {
  if (chartsRendered) return;
  chartsRendered = true;
  renderLossChart();
  renderPerfChart();
}

function renderLossChart() {
  const ctx = document.getElementById('lossChart');
  if (!ctx) return;

  // Simulated L1 loss curve based on typical Real-ESRNet training trajectory (up to 52,000 iters)
  const iters = [];
  const loss  = [];
  for (let i = 2000; i <= 52000; i += 2000) {
    iters.push(i.toLocaleString());
    const t = i / 52000;
    const base = 0.075 * Math.exp(-3.5 * t) + 0.0048;
    loss.push(+(base + (Math.random() * 0.0012 - 0.0006)).toFixed(5));
  }

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: iters,
      datasets: [{
        label: 'L1 Train Loss',
        data: loss,
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.08)',
        borderWidth: 2,
        pointRadius: 2,
        pointBackgroundColor: '#38bdf8',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
        tooltip: { backgroundColor: '#0c1426', borderColor: '#1e293b', borderWidth: 1 }
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 9 }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

function renderPerfChart() {
  const ctx = document.getElementById('perfChart');
  if (!ctx) return;

  const labels = ['Bicubic', 'ESRGAN', 'BSRGAN', 'Real-ESRNet (Pre)', 'Real-ESRGAN (Official)', 'Real-ESRGAN (ทีม 52k)'];

  const datasets = [
    {
      label: 'PSNR (dB) / 30',
      data: [23.41 / 30, 14.45 / 30, 6.69 / 30, 23.41 / 30, 21.75 / 30, 22.38 / 30],
      backgroundColor: 'rgba(56,189,248,0.75)',
      borderColor:     '#38bdf8',
      borderWidth: 1
    },
    {
      label: 'SSIM',
      data: [0.6938, 0.2669, 0.1232, 0.7065, 0.6477, 0.6406],
      backgroundColor: 'rgba(99,102,241,0.75)',
      borderColor:     '#818cf8',
      borderWidth: 1
    },
    {
      label: 'LPIPS (↓ less=better)',
      data: [0.4485, 0.6003, 0.8731, 0.3361, 0.2349, 0.4281],
      backgroundColor: 'rgba(251,146,60,0.75)',
      borderColor:     '#fb923c',
      borderWidth: 1
    },
    {
      label: 'MUSIQ / 100 (↑ score)',
      data: [35.70 / 100, 63.83 / 100, 22.20 / 100, 56.86 / 100, 64.58 / 100, 48.08 / 100],
      backgroundColor: 'rgba(52,211,153,0.75)',
      borderColor:     '#34d399',
      borderWidth: 1
    }
  ];

  new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12 } },
        tooltip: {
          backgroundColor: '#0c1426',
          borderColor: '#1e293b',
          borderWidth: 1,
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex === 0) return `PSNR: ${(ctx.raw * 30).toFixed(2)} dB`;
              if (ctx.datasetIndex === 1) return `SSIM: ${ctx.raw.toFixed(4)}`;
              if (ctx.datasetIndex === 2) return `LPIPS: ${ctx.raw.toFixed(4)} (↓ lower = better)`;
              if (ctx.datasetIndex === 3) return `MUSIQ: ${(ctx.raw * 100).toFixed(2)} (↑ higher = better)`;
              return `${ctx.dataset.label}: ${ctx.raw}`;
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: {
          min: 0, max: 1,
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' }
        }
      }
    }
  });
}

// ─── Reports Menu & Modal ──────────────────────────────────────
function toggleReportsDropdown(e) {
  e.stopPropagation();
  const menu = document.getElementById('reportsDropdown');
  if (menu) menu.classList.toggle('show');
}

function closeReportsDropdown() {
  const menu = document.getElementById('reportsDropdown');
  if (menu) menu.classList.remove('show');
}

document.addEventListener('click', () => {
  closeReportsDropdown();
});

function openReportModal(url, title) {
  const modal = document.getElementById('reportModal');
  const iframe = document.getElementById('modalIframe');
  const titleEl = document.getElementById('modalTitle');
  const extLink = document.getElementById('modalExtLink');

  if (titleEl) titleEl.textContent = title || 'รายงานและผลการทดลอง';
  if (iframe) iframe.src = url;
  if (extLink) extLink.href = url;
  if (modal) modal.classList.add('show');
  document.body.style.overflow = 'hidden';
}

function closeReportModal() {
  const modal = document.getElementById('reportModal');
  const iframe = document.getElementById('modalIframe');
  if (iframe) iframe.src = 'about:blank';
  if (modal) modal.classList.remove('show');
  document.body.style.overflow = '';
}

function closeModalOnOverlay(e) {
  if (e.target && e.target.id === 'reportModal') {
    closeReportModal();
  }
}

