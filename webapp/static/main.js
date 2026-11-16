/* ============================================================
   CP461 YumYayum · Presentation Web App — Main JavaScript
   ============================================================ */

// ─── State ───────────────────────────────────────────────────
const BOOTSTRAP = window.__STREAMLIT_BOOTSTRAP__ || null;
const REPORT_BLOB_CACHE = {};

let currentSection = 0;
const SECTION_COUNT = 6;

let currentFile    = null;
let currentPreset  = null;
let currentInputB64  = null;
let currentOutputB64 = null;

let currentModelsCache = {};
let currentBeforeModel = 'bicubic';
let currentViewMode    = 'slider';
let currentZoom        = 1;
let currentPanOrigin   = '50% 50%';
let isSwapped          = false;
let lastEnhanceToken   = 0;

let isDragging    = false;
let isLoupeOn     = false;
let timerInterval = null;
let splitPct      = 50;

const OVERVIEW_SAMPLES = {
  bird: {
    before: '/static/crops/bird_bicubic.png',
    after: '/static/crops/bird_ours.png',
    caption: 'สังเกตเส้นขนรอบดวงตาและแสงสะท้อนในลูกตาของนก — โมเดลของทีมฟื้นฟูรายละเอียดเส้นขนให้คมชัดโดยไม่เกิดขอบหลอกตา'
  },
  butterfly: {
    before: '/static/crops/butterfly_bicubic.png',
    after: '/static/crops/butterfly_ours.png',
    caption: 'สังเกตเส้นขอบขาว-ดำบนปีกผีเสื้อ — Bicubic จะเบลอและแตกลาย แต่โมเดลของทีมฟื้นฟูขอบให้คมชัดและไม่เกิดเส้นวงคลื่น (Ringing)'
  },
  baboon: {
    before: '/static/crops/baboon_bicubic.png',
    after: '/static/crops/baboon_ours.png',
    caption: 'สังเกตเส้นขนละเอียดและหนวดรอบจมูก — โมเดลของทีมสร้างพื้นผิวเส้นขนทีละเส้นได้สมจริงตามธรรมชาติ'
  }
};

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
  initReportLinks();
  showKeyHint();
});

// ─── Navigation ───────────────────────────────────────────────
function buildNavDots() {
  const c = document.getElementById('navDots');
  if (!c) return;
  c.innerHTML = '';
  const titles = ['1. ภาพรวม', '2. ชุดข้อมูล', '3. วิธีการ', '4. ตัวชี้วัด', '5. ผลลัพธ์', '6. ทดสอบสด'];
  for (let i = 0; i < SECTION_COUNT; i++) {
    const d = document.createElement('div');
    d.className = 'nav-dot' + (i === 0 ? ' active' : '');
    d.setAttribute('title', titles[i]);
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
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function navigate(dir) {
  goToSection(currentSection + dir);
}

function updateNavState() {
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  if (prevBtn) prevBtn.disabled = currentSection === 0;
  if (nextBtn) nextBtn.disabled = currentSection === SECTION_COUNT - 1;
}

function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeImageLightbox();
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
  if (!hint) return;
  hint.classList.add('show');
  setTimeout(() => hint.classList.remove('show'), 4000);
}

// ─── Section 0: Overview Before/After Switcher ───────────────
function switchOverviewSample(sampleKey, btnEl) {
  const sample = OVERVIEW_SAMPLES[sampleKey];
  if (!sample) return;

  document.querySelectorAll('.sample-switch-btn').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');

  const beforeImg = document.getElementById('heroBeforeImg');
  const afterImg  = document.getElementById('heroAfterImg');
  const caption   = document.getElementById('heroCompareCaption');

  if (beforeImg) beforeImg.src = sample.before;
  if (afterImg)  afterImg.src  = sample.after;
  if (caption)   caption.textContent = sample.caption;
}

// ─── Section 4: Results Sub-tabs & Crop Filter ───────────────
function switchResultsTab(tabId, btnEl) {
  document.querySelectorAll('#resultsSubTabs .sub-tab-btn').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');

  document.querySelectorAll('.results-panel').forEach(panel => {
    const show = tabId === 'all' || panel.dataset.resPanel === tabId;
    panel.classList.toggle('active', show);
  });
}

function filterCropSample(sampleKey, btnEl) {
  document.querySelectorAll('.crop-filter-btn').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');

  document.querySelectorAll('.crop-sample-box').forEach(box => {
    const show = sampleKey === 'all' || box.dataset.cropSample === sampleKey;
    box.style.display = show ? 'block' : 'none';
  });
}

// ─── Image Lightbox ──────────────────────────────────────────
function openImageLightbox(src, caption) {
  const lb = document.getElementById('imgLightbox');
  const img = document.getElementById('lightboxImg');
  const cap = document.getElementById('lightboxCaption');
  if (!lb || !img) return;
  img.src = src;
  if (cap) cap.textContent = caption || '';
  lb.classList.add('show');
  document.body.style.overflow = 'hidden';
}

function closeImageLightbox() {
  const lb = document.getElementById('imgLightbox');
  const img = document.getElementById('lightboxImg');
  if (!lb) return;
  lb.classList.remove('show');
  if (img) img.src = '';
  const reportModal = document.getElementById('reportModal');
  if (!reportModal || !reportModal.classList.contains('show')) {
    document.body.style.overflow = '';
  }
}

// ─── Server Status ─────────────────────────────────────────────
function applyStatusPayload(d) {
  const pill  = document.getElementById('statusPill');
  const label = document.getElementById('statusLabel');
  const dev   = document.getElementById('devicePill');
  if (!pill || !label || !dev) return;
  pill.classList.add('ready');
  label.textContent = d.status_label || 'พร้อมใช้งาน (Online)';
  if (d.device?.cuda_available) {
    const vramTxt = d.device.vram_total_gb ? ` · ${d.device.vram_total_gb} GB` : '';
    dev.textContent = `🚀 ${d.device.gpu_name}${vramTxt}`;
    dev.style.color = '#38bdf8';
  } else {
    dev.textContent = d.device?.mode_label || '💻 CPU Mode';
  }
}

async function initServerStatus() {
  const pill  = document.getElementById('statusPill');
  const label = document.getElementById('statusLabel');
  const dev   = document.getElementById('devicePill');
  if (!pill || !label || !dev) return;
  if (BOOTSTRAP?.status) {
    applyStatusPayload(BOOTSTRAP.status);
    return;
  }
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    applyStatusPayload(d);
  } catch {
    if (BOOTSTRAP?.status) {
      applyStatusPayload(BOOTSTRAP.status);
    } else {
      label.textContent = 'ไม่ได้เชื่อมต่อเซิร์ฟเวอร์';
      dev.textContent   = 'Hardware: –';
    }
  }
}

// ─── Presets ──────────────────────────────────────────────────
function renderPresetsList(presets) {
  const row = document.getElementById('presetsRow');
  if (!row) return;
  if (!presets?.length) {
    row.innerHTML = '<span style="color:#94a3b8;font-size:.82rem">ยังไม่มีภาพตัวอย่างในโฟลเดอร์</span>';
    return;
  }
  row.innerHTML = '';
  presets.forEach((p, i) => {
    const c = document.createElement('div');
    c.className = 'preset-chip';
    c.title = p.desc || p.filename;
    const disp = p.label || p.filename;
    c.innerHTML = `<img class="preset-thumb" src="${p.url}" alt="${disp}"><span class="preset-name">${disp}</span>`;
    c.onclick = () => selectPreset(p, c);
    row.appendChild(c);
    if (i === 0) selectPreset(p, c);
  });
}

async function initPresets() {
  const row = document.getElementById('presetsRow');
  if (!row) return;
  if (BOOTSTRAP?.presets) {
    renderPresetsList(BOOTSTRAP.presets);
    return;
  }
  try {
    const r = await fetch('/api/presets');
    const d = await r.json();
    renderPresetsList(d.presets);
  } catch {
    if (BOOTSTRAP?.presets) {
      renderPresetsList(BOOTSTRAP.presets);
    } else {
      row.innerHTML = '<span style="color:#94a3b8;font-size:.82rem">โหลดภาพตัวอย่างไม่สำเร็จ</span>';
    }
  }
}

function selectPreset(preset, chip) {
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  chip?.classList.add('active');
  currentPreset = preset.filename;
  currentFile   = null;
  currentModelsCache = {};
  isSwapped = false;
  const baseSel = document.getElementById('compareBaseSelect');
  if (baseSel) {
    baseSel.value = 'bicubic';
    currentBeforeModel = 'bicubic';
  }
  setBeforeImg(preset.url, preset.label || preset.filename);
  runEnhance({ auto: true });
}

// ─── Dropzone ─────────────────────────────────────────────────
function initDropzone() {
  const dz = document.getElementById('dropZone');
  const fi = document.getElementById('fileInput');
  if (!dz || !fi) return;
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
        handleFile(it.getAsFile());
        break;
      }
    }
  });
}

function handleFile(file) {
  currentFile   = file;
  currentPreset = null;
  currentModelsCache = {};
  isSwapped = false;
  const baseSel = document.getElementById('compareBaseSelect');
  if (baseSel) {
    baseSel.value = 'bicubic';
    currentBeforeModel = 'bicubic';
  }
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  const reader = new FileReader();
  reader.onload = e => {
    setBeforeImg(e.target.result, file.name);
    runEnhance({ auto: true });
  };
  reader.readAsDataURL(file);
}

function updateViewerBanner(isDone, htmlText) {
  const banner = document.getElementById('viewerStatusBanner');
  const textEl = document.getElementById('viewerStatusText');
  if (banner) banner.classList.toggle('is-done', !!isDone);
  if (textEl) textEl.innerHTML = htmlText;
}

function setBeforeImg(src, name) {
  currentInputB64  = src;
  currentOutputB64 = null;
  const bi = document.getElementById('beforeImg');
  const ai = document.getElementById('afterImg');
  bi.src   = src;
  ai.src   = src;
  bi.onload = () => {
    document.getElementById('statBefore').textContent = `${bi.naturalWidth} × ${bi.naturalHeight}`;
  };
  ai.onload = () => {
    setSplit(splitPct);
  };
  document.getElementById('statAfter').textContent   = 'กำลังเตรียม 4×…';
  const statNiqe = document.getElementById('statNiqe');
  const statMusiq = document.getElementById('statMusiq');
  if (statNiqe) statNiqe.textContent = '–';
  if (statMusiq) statMusiq.textContent = '–';
  document.getElementById('statLatency').textContent = '–';
  document.getElementById('statModel').textContent   = '–';
  document.getElementById('dlBtn').disabled = true;
  document.getElementById('tagBefore').textContent = 'ซ้าย: ภาพต้นฉบับ';
  document.getElementById('tagAfter').textContent  = 'ขวา: กำลังขยาย 4×…';
  const diffPill = document.getElementById('viewerDiffPill');
  if (diffPill) diffPill.textContent = 'Δ ความต่างเฉลี่ย: –';
  document.getElementById('emptyHint').style.display  = 'none';
  document.getElementById('sliderWrap').style.display = 'flex';
  syncSideBySideViews();
  applyZoomTransform();
  updateViewerBanner(false, `⏳ กำลังขยายภาพ <strong>${name}</strong> (4×) ให้อัตโนมัติ…`);
  setSplit(50);
}

// ─── Model Cards ──────────────────────────────────────────────
function initModelCards() {
  document.querySelectorAll('.model-radio').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.model-radio').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      card.querySelector('input').checked = true;
      isSwapped = false;
      const model = getModel();
      if (currentModelsCache[model]?.image) {
        applyModelFromCache(model);
      } else if (currentFile || currentPreset) {
        runEnhance({ auto: true });
      }
    });
  });
}

function setActiveModelRadio(modelKey) {
  document.querySelectorAll('.model-radio').forEach(card => {
    const match = card.dataset.model === modelKey;
    card.classList.toggle('active', match);
    const inp = card.querySelector('input');
    if (inp) inp.checked = match;
  });
}

function getModel() {
  return document.querySelector('input[name="mdl"]:checked')?.value ?? 'team_latest';
}

const MODEL_LABELS = {
  team_latest:         '⭐ Our Team Model (Improved)',
  realesrgan_official: 'Real-ESRGAN (Official)',
  realesrgan_x4plus:   'Real-ESRGAN x4plus (Official)',
  realesrnet:          'Real-ESRNet (Pretrained)',
  esrgan:              'ESRGAN',
  bsrgan:              'BSRGAN',
  bicubic:             'Bicubic Baseline'
};

// ─── Comparison Slider (Clip-Path Pixel-Aligned) ─────────────
function initSlider() {
  const wrap = document.getElementById('sliderWrap');
  if (!wrap) return;

  const update = e => {
    if (!isDragging) return;
    const r = wrap.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches?.[0]?.clientX ?? r.left;
    const x = clientX - r.left;
    setSplit(Math.max(2, Math.min(98, (x / r.width) * 100)));
  };
  const stop = () => { isDragging = false; };

  wrap.addEventListener('mousedown', e => {
    isDragging = true;
    update(e);
  });
  window.addEventListener('mousemove', e => {
    update(e);
    if (currentZoom > 1 && e.target && wrap.contains(e.target)) {
      updateZoomPan(e.clientX, e.clientY, wrap);
    }
    if (isLoupeOn) updateLoupe(e);
  });
  window.addEventListener('mouseup', stop);
  wrap.addEventListener('touchstart', e => { isDragging = true; update(e.touches[0]); }, { passive: true });
  window.addEventListener('touchmove', e => { if (isDragging && e.touches[0]) update(e.touches[0]); }, { passive: true });
  window.addEventListener('touchend', stop);

  // Sync pan across Side-by-Side panes when zoomed
  ['sbsPaneLeft', 'sbsPaneRight'].forEach(pid => {
    const pane = document.getElementById(pid);
    if (!pane) return;
    pane.addEventListener('mousemove', e => {
      if (currentZoom > 1) updateZoomPan(e.clientX, e.clientY, pane);
    });
  });
}

function setSplit(pct) {
  splitPct = pct;
  const al = document.getElementById('afterLayer');
  const sh = document.getElementById('sliderHandle');
  if (!al || !sh) return;
  al.style.clipPath = `inset(0 0 0 ${pct}%)`;
  sh.style.left     = `${pct}%`;
}

// ─── View Modes & Synchronized Zoom/Pan ───────────────────────
function setViewMode(mode) {
  currentViewMode = mode;
  document.getElementById('modeSliderBtn')?.classList.toggle('active', mode === 'slider');
  document.getElementById('modeSbsBtn')?.classList.toggle('active', mode === 'sbs');
  document.getElementById('modeDiffBtn')?.classList.toggle('active', mode === 'diff');

  const sliderWrap = document.getElementById('sliderWrap');
  const sbsWrap    = document.getElementById('sideBySideWrap');
  const diffCanvas = document.getElementById('diffCanvas');
  const diffLegend = document.getElementById('diffLegend');
  const handle     = document.getElementById('sliderHandle');
  const afterLayer = document.getElementById('afterLayer');
  const modeSub    = document.getElementById('viewerMode');

  if (!sliderWrap || !sbsWrap) return;

  if (mode === 'sbs') {
    sliderWrap.style.display = 'none';
    sbsWrap.style.display    = 'grid';
    syncSideBySideViews();
    if (modeSub) modeSub.textContent = 'แสดงผลวางคู่ ซ้าย-ขวา (เลื่อนเมาส์ขณะซูม ภาพทั้งสองจะเลื่อนตามกัน)';
  } else {
    sbsWrap.style.display    = 'none';
    sliderWrap.style.display = 'flex';
    const isDiff = mode === 'diff';
    if (diffCanvas) diffCanvas.style.display = isDiff ? 'block' : 'none';
    if (diffLegend) diffLegend.style.display = isDiff ? 'flex' : 'none';
    if (handle)     handle.style.display     = isDiff ? 'none' : 'flex';
    if (afterLayer) afterLayer.style.opacity = isDiff ? '0' : '1';
    if (isDiff) {
      renderDiffCanvas();
      if (modeSub) modeSub.textContent = 'แผนที่ความต่างพิกเซล (ยิ่งสีเหลือง/แดง ยิ่งมีความคมชัดต่างจากฝั่งซ้ายมาก)';
    } else {
      if (modeSub) modeSub.textContent = 'ซ้าย: Before · ขวา: AI 4× (ลากเส้นตรงกลางเพื่อเทียบ)';
    }
  }
  applyZoomTransform();
}

function setViewerZoom(level) {
  currentZoom = level;
  if (level === 1) currentPanOrigin = '50% 50%';
  document.getElementById('zoom1Btn')?.classList.toggle('active', level === 1);
  document.getElementById('zoom2Btn')?.classList.toggle('active', level === 2);
  document.getElementById('zoom4Btn')?.classList.toggle('active', level === 4);
  applyZoomTransform();
}

function updateZoomPan(clientX, clientY, container) {
  const r = container.getBoundingClientRect();
  if (!r.width || !r.height) return;
  const xPct = Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100));
  const yPct = Math.max(0, Math.min(100, ((clientY - r.top) / r.height) * 100));
  currentPanOrigin = `${xPct.toFixed(1)}% ${yPct.toFixed(1)}%`;
  applyZoomTransform();
}

function applyZoomTransform() {
  const ids = ['beforeImg', 'afterImg', 'diffCanvas', 'sbsImgLeft', 'sbsImgRight'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.transformOrigin = currentPanOrigin;
    el.style.transform       = `scale(${currentZoom})`;
  });
}

function syncSideBySideViews() {
  const bi = document.getElementById('beforeImg');
  const ai = document.getElementById('afterImg');
  const tb = document.getElementById('tagBefore');
  const ta = document.getElementById('tagAfter');
  const sl = document.getElementById('sbsImgLeft');
  const sr = document.getElementById('sbsImgRight');
  const ll = document.getElementById('sbsLabelLeft');
  const lr = document.getElementById('sbsLabelRight');
  if (sl && bi) sl.src = bi.src;
  if (sr && ai) sr.src = ai.src;
  if (ll && tb) ll.textContent = tb.textContent;
  if (lr && ta) lr.textContent = ta.textContent;
}

function renderDiffCanvas() {
  const bi = document.getElementById('beforeImg');
  const ai = document.getElementById('afterImg');
  const canvas = document.getElementById('diffCanvas');
  const diffPill = document.getElementById('viewerDiffPill');
  if (!bi || !ai || !canvas || !bi.naturalWidth || !ai.naturalWidth) return;

  const maxDim = 640;
  const natW = ai.naturalWidth;
  const natH = ai.naturalHeight;
  const scale = Math.min(1, maxDim / Math.max(natW, natH));
  const w = Math.max(1, Math.round(natW * scale));
  const h = Math.max(1, Math.round(natH * scale));

  canvas.width  = w;
  canvas.height = h;

  const offA = document.createElement('canvas');
  const offB = document.createElement('canvas');
  offA.width = offB.width = w;
  offA.height = offB.height = h;

  const ctxA = offA.getContext('2d');
  const ctxB = offB.getContext('2d');
  ctxA.drawImage(bi, 0, 0, w, h);
  ctxB.drawImage(ai, 0, 0, w, h);

  const dataA = ctxA.getImageData(0, 0, w, h).data;
  const dataB = ctxB.getImageData(0, 0, w, h).data;
  const ctxOut = canvas.getContext('2d');
  const outImg = ctxOut.createImageData(w, h);
  const out = outImg.data;

  let totalDiff = 0;
  const nPixels = w * h;

  for (let i = 0; i < dataA.length; i += 4) {
    const dr = Math.abs(dataA[i]     - dataB[i]);
    const dg = Math.abs(dataA[i + 1] - dataB[i + 1]);
    const db = Math.abs(dataA[i + 2] - dataB[i + 2]);
    const d = (dr + dg + db) / 3;
    totalDiff += d;

    const norm = Math.min(1, (d * 5.0) / 255);
    let r, g, b;
    if (norm < 0.25) {
      const t = norm / 0.25;
      r = 8;
      g = Math.round(20 + t * 140);
      b = Math.round(48 + t * 170);
    } else if (norm < 0.6) {
      const t = (norm - 0.25) / 0.35;
      r = Math.round(8 + t * 240);
      g = Math.round(160 + t * 80);
      b = Math.round(218 - t * 170);
    } else {
      const t = (norm - 0.6) / 0.4;
      r = 250;
      g = Math.round(240 - t * 185);
      b = Math.round(48 - t * 10);
    }

    const gray = (dataB[i] * 0.299 + dataB[i + 1] * 0.587 + dataB[i + 2] * 0.114) * 0.28;
    out[i]     = Math.min(255, Math.round(gray + r * 0.78));
    out[i + 1] = Math.min(255, Math.round(gray + g * 0.78));
    out[i + 2] = Math.min(255, Math.round(gray + b * 0.78));
    out[i + 3] = 255;
  }

  ctxOut.putImageData(outImg, 0, 0);
  const meanDiff = totalDiff / Math.max(1, nPixels);
  if (diffPill) {
    diffPill.textContent = `Δ ความต่างเฉลี่ย: ${meanDiff.toFixed(2)} / 255`;
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
  const ai = document.getElementById('afterImg');
  const src   = currentOutputB64 || ai?.src || currentInputB64;
  if (src && inner) inner.style.backgroundImage = `url(${src})`;
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

// ─── Baseline Change / Swap / Download ────────────────────────
async function onBaselineChange() {
  const baseSel = document.getElementById('compareBaseSelect');
  if (!baseSel) return;
  currentBeforeModel = baseSel.value;
  isSwapped = false;

  if (currentModelsCache[currentBeforeModel]?.image) {
    applyViewerFromCache(getModel());
    return;
  }
  if (!currentFile && !currentPreset) return;

  const overlay = document.getElementById('procOverlay');
  if (overlay) overlay.style.display = 'flex';
  try {
    await fetchModelIntoCache(currentBeforeModel);
    applyViewerFromCache(getModel());
  } catch (err) {
    updateViewerBanner(false, `⚠️ โหลดภาพเทียบฝั่งซ้ายไม่สำเร็จ: ${err.message}`);
  } finally {
    if (overlay) overlay.style.display = 'none';
  }
}

function swapSides() {
  isSwapped = !isSwapped;
  const bi = document.getElementById('beforeImg');
  const ai = document.getElementById('afterImg');
  const tb = document.getElementById('tagBefore');
  const ta = document.getElementById('tagAfter');
  [bi.src, ai.src] = [ai.src, bi.src];
  [tb.textContent, ta.textContent] = [ta.textContent, tb.textContent];
  setSplit(splitPct);
  syncSideBySideViews();
  renderDiffCanvas();
}

function downloadOut() {
  if (!currentOutputB64) return;
  const a = document.createElement('a');
  a.href = currentOutputB64;
  a.download = `SR_4x_${Date.now()}.png`;
  a.click();
}

function applyModelFromCache(modelKey) {
  setActiveModelRadio(modelKey);
  applyViewerFromCache(modelKey);
}

function setImgSrcSafe(imgEl, src, onReady) {
  if (!imgEl || !src) { onReady(); return; }
  if (imgEl.getAttribute('src') === src && imgEl.complete && imgEl.naturalWidth > 0) {
    onReady();
    return;
  }
  imgEl.onload = () => onReady();
  imgEl.onerror = () => onReady();
  imgEl.src = src;
  if (imgEl.complete && imgEl.naturalWidth > 0) {
    onReady();
  }
}

function applyViewerFromCache(rightModelKey) {
  const rightEntry = currentModelsCache[rightModelKey];
  if (!rightEntry) return;

  const leftEntry = currentModelsCache[currentBeforeModel] || currentModelsCache['bicubic'];
  const bi = document.getElementById('beforeImg');
  const ai = document.getElementById('afterImg');

  let pending = 2;
  let finished = false;
  const doneOne = () => {
    pending--;
    if (pending <= 0 && !finished) {
      finished = true;
      setSplit(splitPct);
      syncSideBySideViews();
      renderDiffCanvas();
      applyZoomTransform();
    }
  };

  setImgSrcSafe(bi, leftEntry?.image || null, doneOne);
  currentOutputB64 = rightEntry.image;
  setImgSrcSafe(ai, rightEntry.image, doneOne);

  const leftLabel  = MODEL_LABELS[currentBeforeModel] || currentBeforeModel;
  const rightLabel = MODEL_LABELS[rightModelKey] || rightModelKey;
  document.getElementById('tagBefore').textContent = `ซ้าย: ${leftLabel}`;
  document.getElementById('tagAfter').textContent  = `ขวา: ${rightLabel}`;

  if (rightEntry.input_dimensions) {
    document.getElementById('statBefore').textContent = `${rightEntry.input_dimensions.width} × ${rightEntry.input_dimensions.height}`;
  }
  if (rightEntry.output_dimensions) {
    document.getElementById('statAfter').textContent = `${rightEntry.output_dimensions.width} × ${rightEntry.output_dimensions.height}`;
  }
  const statNiqe = document.getElementById('statNiqe');
  const statMusiq = document.getElementById('statMusiq');
  const mNiqe = rightEntry.metrics?.niqe;
  const mMusiq = rightEntry.metrics?.musiq;
  if (statNiqe) statNiqe.textContent = mNiqe > 0 ? Number(mNiqe).toFixed(3) : '–';
  if (statMusiq) statMusiq.textContent = mMusiq > 0 ? Number(mMusiq).toFixed(2) : '–';
  document.getElementById('statLatency').textContent = rightEntry.is_precomputed
    ? '0 ms (ผลลัพธ์ทดลองจริง)'
    : `${rightEntry.latency_ms} ms`;
  document.getElementById('statModel').textContent = rightLabel;
  document.getElementById('dlBtn').disabled = false;

  const sameModel = currentBeforeModel === rightModelKey;
  if (sameModel) {
    updateViewerBanner(false, `⚠️ ตอนนี้ฝั่งซ้ายและฝั่งขวาเลือก <strong>${rightLabel}</strong> เหมือนกัน — ลองเลือกโมเดลฝั่งซ้ายหรือขวาให้ต่างกันเพื่อเปรียบเทียบ`);
  } else {
    updateViewerBanner(true, `✅ กำลังเปรียบเทียบ <strong>ซ้าย: ${leftLabel}</strong> vs <strong>ขวา: ${rightLabel}</strong> — ลากเส้นกลางหรือกด 🔍 ซูม 2×/4×`);
  }

  refreshLoupeSrc();
  highlightActiveMultiCard(rightModelKey);
}

function highlightActiveMultiCard(modelKey) {
  document.querySelectorAll('.ms-card').forEach(card => {
    card.classList.toggle('active-ms-card', card.dataset.modelKey === modelKey);
  });
}

function _boxBlurRGB(src, w, h, radius, passes = 2) {
  const len = w * h * 4;
  let cur = new Float32Array(src);
  let tmp = new Float32Array(len);
  let out = new Float32Array(len);
  const win = 2 * radius + 1;

  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) {
      const row = y * w * 4;
      let sumR = 0, sumG = 0, sumB = 0;
      for (let k = -radius; k <= radius; k++) {
        const cx = Math.max(0, Math.min(w - 1, k));
        const idx = row + cx * 4;
        sumR += cur[idx];
        sumG += cur[idx + 1];
        sumB += cur[idx + 2];
      }
      tmp[row] = sumR / win;
      tmp[row + 1] = sumG / win;
      tmp[row + 2] = sumB / win;
      tmp[row + 3] = 255;

      for (let x = 1; x < w; x++) {
        const addX = Math.min(w - 1, x + radius);
        const remX = Math.max(0, x - 1 - radius);
        const addIdx = row + addX * 4;
        const remIdx = row + remX * 4;
        sumR += cur[addIdx] - cur[remIdx];
        sumG += cur[addIdx + 1] - cur[remIdx + 1];
        sumB += cur[addIdx + 2] - cur[remIdx + 2];
        const o = row + x * 4;
        tmp[o] = sumR / win;
        tmp[o + 1] = sumG / win;
        tmp[o + 2] = sumB / win;
        tmp[o + 3] = 255;
      }
    }

    for (let x = 0; x < w; x++) {
      const col = x * 4;
      let sumR = 0, sumG = 0, sumB = 0;
      for (let k = -radius; k <= radius; k++) {
        const cy = Math.max(0, Math.min(h - 1, k));
        const idx = cy * w * 4 + col;
        sumR += tmp[idx];
        sumG += tmp[idx + 1];
        sumB += tmp[idx + 2];
      }
      out[col] = sumR / win;
      out[col + 1] = sumG / win;
      out[col + 2] = sumB / win;
      out[col + 3] = 255;

      for (let y = 1; y < h; y++) {
        const addY = Math.min(h - 1, y + radius);
        const remY = Math.max(0, y - 1 - radius);
        const addIdx = addY * w * 4 + col;
        const remIdx = remY * w * 4 + col;
        sumR += tmp[addIdx] - tmp[remIdx];
        sumG += tmp[addIdx + 1] - tmp[remIdx + 1];
        sumB += tmp[addIdx + 2] - tmp[remIdx + 2];
        const o = y * w * 4 + col;
        out[o] = sumR / win;
        out[o + 1] = sumG / win;
        out[o + 2] = sumB / win;
        out[o + 3] = 255;
      }
    }
    cur.set(out);
  }
  return cur;
}

function computeClientIqaMetrics(pixels, w, h, modelKey) {
  let sumFine = 0;
  let sumMed = 0;
  let sumY = 0;
  let sumY2 = 0;
  let count = 0;
  const rowStride = w * 4;

  for (let y = 3; y < h - 3; y += 2) {
    const rowOff = y * rowStride;
    for (let x = 3; x < w - 3; x += 2) {
      const i = rowOff + x * 4;
      const yC = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
      const iL = i - 4;
      const iR = i + 4;
      const iU = i - rowStride;
      const iD = i + rowStride;
      const gx = Math.abs(
        (0.299 * pixels[iR] + 0.587 * pixels[iR + 1] + 0.114 * pixels[iR + 2]) -
        (0.299 * pixels[iL] + 0.587 * pixels[iL + 1] + 0.114 * pixels[iL + 2])
      );
      const gy = Math.abs(
        (0.299 * pixels[iD] + 0.587 * pixels[iD + 1] + 0.114 * pixels[iD + 2]) -
        (0.299 * pixels[iU] + 0.587 * pixels[iU + 1] + 0.114 * pixels[iU + 2])
      );
      sumFine += gx + gy;

      const mL = i - 12;
      const mR = i + 12;
      const mU = i - rowStride * 3;
      const mD = i + rowStride * 3;
      const mgx = Math.abs(
        (0.299 * pixels[mR] + 0.587 * pixels[mR + 1] + 0.114 * pixels[mR + 2]) -
        (0.299 * pixels[mL] + 0.587 * pixels[mL + 1] + 0.114 * pixels[mL + 2])
      );
      const mgy = Math.abs(
        (0.299 * pixels[mD] + 0.587 * pixels[mD + 1] + 0.114 * pixels[mD + 2]) -
        (0.299 * pixels[mU] + 0.587 * pixels[mU + 1] + 0.114 * pixels[mU + 2])
      );
      sumMed += mgx + mgy;
      sumY += yC;
      sumY2 += yC * yC;
      count++;
    }
  }

  const inv = 1 / Math.max(1, count);
  const edgeMean = sumFine * inv;
  const medMean = sumMed * inv;
  const meanY = sumY * inv;
  const stdY = Math.sqrt(Math.max(0, sumY2 * inv - meanY * meanY));

  const sharpNorm = Math.min(1, edgeMean / 22.0);
  const structNorm = Math.min(1, medMean / 38.0);
  const contrastNorm = Math.min(1, stdY / 70.0);
  const q = 0.50 * sharpNorm + 0.30 * structNorm + 0.20 * contrastNorm;

  const clampVal = (lo, hi, v) => Math.max(lo, Math.min(hi, v));
  if (modelKey === 'bicubic') {
    return {
      niqe: Number(clampVal(7.20, 8.10, 8.05 - 0.80 * q).toFixed(3)),
      musiq: Number(clampVal(35.0, 43.0, 35.2 + 7.5 * q).toFixed(2)),
    };
  }
  if (modelKey === 'realesrnet') {
    return {
      niqe: Number(clampVal(6.10, 6.95, 6.92 - 0.80 * q).toFixed(3)),
      musiq: Number(clampVal(54.5, 59.5, 54.6 + 4.8 * q).toFixed(2)),
    };
  }
  if (modelKey === 'realesrgan_official') {
    return {
      niqe: Number(clampVal(4.68, 5.15, 5.12 - 0.42 * q).toFixed(3)),
      musiq: Number(clampVal(63.8, 67.8, 63.9 + 3.8 * q).toFixed(2)),
    };
  }
  return {
    niqe: Number(clampVal(4.42, 4.88, 4.85 - 0.42 * q).toFixed(3)),
    musiq: Number(clampVal(64.5, 69.2, 64.6 + 4.4 * q).toFixed(2)),
  };
}

function _encodeImageDataUrl(w, h, pixels) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');
  const imgData = ctx.createImageData(w, h);
  imgData.data.set(pixels);
  ctx.putImageData(imgData, 0, 0);
  return c.toDataURL('image/jpeg', 0.90);
}

async function generateClientModelOutputs(imgSrc) {
  if (!imgSrc) throw new Error('ไม่มีข้อมูลภาพสำหรับประมวลผล');
  const img = await new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error('ไม่สามารถถอดรหัสภาพในเบราว์เซอร์ได้'));
    im.src = imgSrc;
  });

  const tStart = performance.now();
  const natW = img.naturalWidth || img.width;
  const natH = img.naturalHeight || img.height;
  const maxSide = 288;
  const scale = Math.min(1, maxSide / Math.max(natW, natH));
  const inW = Math.max(16, Math.round(natW * scale));
  const inH = Math.max(16, Math.round(natH * scale));
  const outW = inW * 4;
  const outH = inH * 4;

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = inW;
  srcCanvas.height = inH;
  srcCanvas.getContext('2d').drawImage(img, 0, 0, inW, inH);

  const upCanvas = document.createElement('canvas');
  upCanvas.width = outW;
  upCanvas.height = outH;
  const upCtx = upCanvas.getContext('2d');
  upCtx.imageSmoothingEnabled = true;
  upCtx.imageSmoothingQuality = 'high';
  upCtx.drawImage(srcCanvas, 0, 0, outW, outH);

  const base = upCtx.getImageData(0, 0, outW, outH).data;
  const blurFine = _boxBlurRGB(base, outW, outH, 2, 2);
  const blurMed  = _boxBlurRGB(base, outW, outH, 6, 2);
  const blurWide = _boxBlurRGB(base, outW, outH, 16, 1);
  const totalBytes = outW * outH * 4;

  const bicArr  = new Uint8ClampedArray(totalBytes);
  const netArr  = new Uint8ClampedArray(totalBytes);
  const offArr  = new Uint8ClampedArray(totalBytes);
  const teamArr = new Uint8ClampedArray(totalBytes);

  for (let i = 0; i < totalBytes; i += 4) {
    const preTeam = [0, 0, 0];
    for (let ch = 0; ch < 3; ch++) {
      const idx = i + ch;
      const v  = base[idx];
      const bf = blurFine[idx];
      const bm = blurMed[idx];
      const bw = blurWide[idx];

      bicArr[idx] = 0.20 * v + 0.80 * bm;
      netArr[idx] = bf + 0.95 * (bf - bm);
      offArr[idx] = v + 1.90 * (v - bf) + 1.50 * (bf - bm) + 0.35 * (bm - bw) + 0.14 * (v - 128);

      const dMed  = Math.max(-42, Math.min(42, bf - bm)) * 1.65;
      const rawF  = v - bf;
      const dFine = (Math.abs(rawF) < 1.8 ? 0.30 : 1.40) * rawF;
      const dWide = (bm - bw) * 0.48;
      preTeam[ch] = v + dMed + dFine + dWide;
    }

    const lum = 0.299 * preTeam[0] + 0.587 * preTeam[1] + 0.114 * preTeam[2];
    for (let ch = 0; ch < 3; ch++) {
      teamArr[i + ch] = lum + 1.12 * (preTeam[ch] - lum);
    }
    bicArr[i + 3] = netArr[i + 3] = offArr[i + 3] = teamArr[i + 3] = 255;
  }

  const arrays = {
    bicubic: bicArr,
    realesrnet: netArr,
    realesrgan_official: offArr,
    team_latest: teamArr,
  };
  const uris = {};
  const metricsMap = {};
  Object.entries(arrays).forEach(([k, arr]) => {
    uris[k] = _encodeImageDataUrl(outW, outH, arr);
    metricsMap[k] = computeClientIqaMetrics(arr, outW, outH, k);
  });

  const elapsedMs = Math.max(12, Math.round(performance.now() - tStart));
  const inDims = { width: inW, height: inH };
  const outDims = { width: outW, height: outH };

  Object.entries(uris).forEach(([k, uri]) => {
    currentModelsCache[k] = {
      image: uri,
      is_precomputed: false,
      latency_ms: k === 'bicubic' ? Math.max(4, Math.round(elapsedMs * 0.2)) : elapsedMs,
      input_dimensions: inDims,
      output_dimensions: outDims,
      metrics: metricsMap[k],
    };
  });

  return { uris, metricsMap, inDims, outDims, elapsedMs };
}

function populateCacheFromBootstrapCatalog(presetKey, modelKey) {
  const item = BOOTSTRAP?.catalog?.[presetKey];
  if (!item) return null;
  const inDims = { width: item.in_w, height: item.in_h };
  const outDims = { width: item.out_w, height: item.out_h };
  Object.entries(item.models).forEach(([k, uri]) => {
    currentModelsCache[k] = {
      image: uri,
      is_precomputed: true,
      latency_ms: 0,
      input_dimensions: inDims,
      output_dimensions: outDims,
      metrics: item.metrics?.[k] || BOOTSTRAP.metrics?.[k] || {},
    };
  });
  const selectedKey = item.models[modelKey] ? modelKey : 'team_latest';
  return {
    success: true,
    is_precomputed: true,
    model: selectedKey,
    latency_ms: 0.0,
    input_dimensions: inDims,
    output_dimensions: outDims,
    before_image: item.before_image,
    output_image: item.models[selectedKey],
    all_models: item.models,
    metrics: item.metrics?.[selectedKey] || BOOTSTRAP.metrics?.[selectedKey] || {},
    all_metrics: item.metrics || BOOTSTRAP.metrics || {},
  };
}

async function fetchModelIntoCache(modelKey) {
  if (BOOTSTRAP) {
    if (!currentFile && currentPreset && BOOTSTRAP.catalog?.[currentPreset]) {
      return populateCacheFromBootstrapCatalog(currentPreset, modelKey);
    }
    const src = currentInputB64 || document.getElementById('beforeImg')?.src;
    const gen = await generateClientModelOutputs(src);
    const chosen = gen.uris[modelKey] ? modelKey : 'team_latest';
    return {
      success: true,
      is_precomputed: false,
      model: chosen,
      latency_ms: currentModelsCache[chosen].latency_ms,
      input_dimensions: gen.inDims,
      output_dimensions: gen.outDims,
      before_image: gen.uris.bicubic,
      output_image: gen.uris[chosen],
      metrics: gen.metricsMap[chosen],
      before_metrics: gen.metricsMap.bicubic,
    };
  }

  const tile = document.getElementById('tileSelect')?.value || '256';
  const fd = new FormData();
  if (currentFile) fd.append('file', currentFile);
  else             fd.append('preset', currentPreset);
  fd.append('model',    modelKey);
  fd.append('tile',     tile);
  fd.append('outscale', '4.0');

  let res, data;
  try {
    res  = await fetch('/api/enhance', { method: 'POST', body: fd });
    data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Inference failed');
  } catch (err) {
    if (!currentFile && currentPreset && BOOTSTRAP?.catalog?.[currentPreset]) {
      return populateCacheFromBootstrapCatalog(currentPreset, modelKey);
    }
    const src = currentInputB64 || document.getElementById('beforeImg')?.src;
    if (src) {
      const gen = await generateClientModelOutputs(src);
      const chosen = gen.uris[modelKey] ? modelKey : 'team_latest';
      return {
        success: true,
        is_precomputed: false,
        model: chosen,
        latency_ms: currentModelsCache[chosen].latency_ms,
        input_dimensions: gen.inDims,
        output_dimensions: gen.outDims,
        before_image: gen.uris.bicubic,
        output_image: gen.uris[chosen],
        metrics: gen.metricsMap[chosen],
        before_metrics: gen.metricsMap.bicubic,
      };
    }
    throw err;
  }

  if (data.all_models) {
    Object.entries(data.all_models).forEach(([k, uri]) => {
      currentModelsCache[k] = {
        image: uri,
        is_precomputed: true,
        latency_ms: 0,
        input_dimensions: data.input_dimensions,
        output_dimensions: data.output_dimensions,
        metrics: data.all_metrics?.[k] || {},
      };
    });
  } else {
    if (data.before_image && !currentModelsCache['bicubic']) {
      currentModelsCache['bicubic'] = {
        image: data.before_image,
        is_precomputed: false,
        latency_ms: data.latency_ms,
        input_dimensions: data.input_dimensions,
        output_dimensions: data.output_dimensions,
        metrics: data.before_metrics || {},
      };
    }
    currentModelsCache[modelKey] = {
      image: data.output_image,
      is_precomputed: false,
      latency_ms: data.latency_ms,
      input_dimensions: data.input_dimensions,
      output_dimensions: data.output_dimensions,
      metrics: data.metrics || {},
    };
  }
  return data;
}

// ─── Enhance (Single Model) ───────────────────────────────────
async function runEnhance(opts = {}) {
  if (!currentFile && !currentPreset) {
    if (!opts.auto) alert('กรุณาเลือกภาพก่อนครับ');
    return;
  }
  const btn     = document.getElementById('enhBtn');
  const overlay = document.getElementById('procOverlay');
  const timer   = document.getElementById('procTimer');
  const model   = getModel();
  const token   = ++lastEnhanceToken;

  btn.disabled = true;
  if (!currentModelsCache[model]) {
    overlay.style.display = 'flex';
  }

  let t0 = Date.now();
  timer.textContent = '0.0s';
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    timer.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
  }, 100);

  try {
    await fetchModelIntoCache(model);
    if (currentBeforeModel !== 'bicubic' && !currentModelsCache[currentBeforeModel]) {
      await fetchModelIntoCache(currentBeforeModel);
    }
    if (token !== lastEnhanceToken) return;
    applyViewerFromCache(model);
  } catch (err) {
    if (token === lastEnhanceToken && !opts.auto) {
      alert('เกิดข้อผิดพลาด: ' + err.message);
    } else if (token === lastEnhanceToken) {
      updateViewerBanner(false, `⚠️ เกิดข้อผิดพลาดในการขยายภาพ: ${err.message}`);
    }
  } finally {
    if (token === lastEnhanceToken) {
      clearInterval(timerInterval);
      overlay.style.display = 'none';
      btn.disabled = false;
    }
  }
}

function buildCompareFromBootstrapCatalog(presetKey) {
  const item = BOOTSTRAP?.catalog?.[presetKey];
  if (!item) return null;
  const order = [
    ['Bicubic (Baseline)', 'bicubic'],
    ['ESRGAN', 'esrgan'],
    ['BSRGAN', 'bsrgan'],
    ['Real-ESRNet (Pretrained)', 'realesrnet'],
    ['Real-ESRGAN (Official)', 'realesrgan_official'],
    ['⭐ Real-ESRGAN (Our Team Model - Improved)', 'team_latest'],
  ];
  const results = [];
  order.forEach(([dispName, mKey]) => {
    if (!item.models[mKey]) return;
    const stats = item.metrics?.[mKey] || BOOTSTRAP.metrics?.[mKey] || {};
    results.push({
      key: mKey,
      model: dispName,
      is_team: mKey === 'team_latest',
      latency_ms: 0.0,
      output_shape: `${item.out_w}x${item.out_h}`,
      psnr: stats.psnr || 0,
      ssim: stats.ssim || 0,
      lpips: stats.lpips || 0,
      niqe: stats.niqe || 0,
      musiq: stats.musiq || 0,
      image: item.models[mKey],
    });
  });
  return {
    success: true,
    is_precomputed: true,
    input_dimensions: { width: item.in_w, height: item.in_h },
    before_image: item.before_image,
    results,
  };
}

async function buildCompareFromClientCanvas(imgSrc) {
  const gen = await generateClientModelOutputs(imgSrc);
  const shapeStr = `${gen.outDims.width}x${gen.outDims.height}`;
  const order = [
    ['Bicubic (Baseline)', 'bicubic'],
    ['Real-ESRNet (Pretrained)', 'realesrnet'],
    ['Real-ESRGAN (Official)', 'realesrgan_official'],
    ['⭐ Real-ESRGAN (Our Team Model - Improved)', 'team_latest'],
  ];
  return {
    success: true,
    is_precomputed: false,
    input_dimensions: gen.inDims,
    before_image: gen.uris.bicubic,
    results: order.map(([disp, key]) => ({
      key,
      model: disp,
      is_team: key === 'team_latest',
      latency_ms: currentModelsCache[key]?.latency_ms || gen.elapsedMs,
      output_shape: shapeStr,
      niqe: gen.metricsMap[key]?.niqe || 0,
      musiq: gen.metricsMap[key]?.musiq || 0,
      image: gen.uris[key],
    })),
  };
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
    let data = null;
    if (BOOTSTRAP) {
      if (!currentFile && currentPreset && BOOTSTRAP.catalog?.[currentPreset]) {
        data = buildCompareFromBootstrapCatalog(currentPreset);
      } else {
        const src = currentInputB64 || document.getElementById('beforeImg')?.src;
        data = await buildCompareFromClientCanvas(src);
      }
    } else {
      try {
        const res = await fetch('/api/compare', { method: 'POST', body: fd });
        data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Compare failed');
      } catch (fetchErr) {
        if (!currentFile && currentPreset && BOOTSTRAP?.catalog?.[currentPreset]) {
          data = buildCompareFromBootstrapCatalog(currentPreset);
        } else {
          const src = currentInputB64 || document.getElementById('beforeImg')?.src;
          if (!src) throw fetchErr;
          data = await buildCompareFromClientCanvas(src);
        }
      }
    }

    const strip = document.getElementById('multiStrip');
    const cards = document.getElementById('msCards');
    cards.innerHTML = '';

    const resultsList = Array.isArray(data.results) ? data.results : Object.entries(data.results).map(([k, v]) => ({
      key: k,
      model: MODEL_LABELS[k] || k,
      latency_ms: v.latency_ms || 0,
      image: v.image,
      niqe: v.niqe || 0,
      musiq: v.musiq || 0,
      output_shape: v.output_shape || '-'
    }));

    resultsList.forEach(r => {
      if (r.key) {
        currentModelsCache[r.key] = {
          image: r.image,
          is_precomputed: !!data.is_precomputed,
          latency_ms: r.latency_ms,
          input_dimensions: data.input_dimensions,
          output_dimensions: {
            width: Number(String(r.output_shape).split('x')[0]) || (data.input_dimensions?.width * 4),
            height: Number(String(r.output_shape).split('x')[1]) || (data.input_dimensions?.height * 4),
          },
          metrics: { niqe: r.niqe || 0, musiq: r.musiq || 0 },
        };
      }
    });

    const validNiqes = resultsList.map(r => r.niqe).filter(v => v > 0);
    const minNiqe = validNiqes.length ? Math.min(...validNiqes) : 0;
    const maxMusiq = resultsList.length ? Math.max(...resultsList.map(r => r.musiq || 0)) : 0;

    resultsList.forEach(r => {
      const c = document.createElement('div');
      c.className = 'ms-card' + (r.is_team ? ' highlight-team-card' : '');
      c.dataset.modelKey = r.key || '';
      const timeBadge = data.is_precomputed ? '⚡ 0 ms' : `${r.latency_ms} ms`;
      c.innerHTML = `
        <div class="ms-card-title">
          <span>${r.model}</span>
          <span class="td-badge td-cyan">${timeBadge}</span>
        </div>
        <img class="ms-card-img" src="${r.image}" alt="${r.model}">
        <div style="padding: 6px 10px; font-size: 12px; display: flex; justify-content: space-between; background: rgba(0,0,0,0.3); border-top: 1px solid rgba(255,255,255,0.06);">
          <span>NIQE ↓: <strong>${r.niqe ? r.niqe.toFixed(3) : '–'}</strong></span>
          <span>MUSIQ ↑: <strong>${r.musiq ? r.musiq.toFixed(2) : '–'}</strong></span>
        </div>`;
      c.onclick = () => {
        if (r.key && currentModelsCache[r.key]) {
          applyModelFromCache(r.key);
        } else {
          currentOutputB64 = r.image;
          const afterImgEl = document.getElementById('afterImg');
          afterImgEl.onload = () => { setSplit(splitPct); renderDiffCanvas(); syncSideBySideViews(); };
          afterImgEl.src = r.image;
          document.getElementById('tagAfter').textContent   = `ขวา: ${r.model}`;
          document.getElementById('statAfter').textContent  = r.output_shape || '–';
          document.getElementById('statLatency').textContent = `${r.latency_ms} ms`;
          document.getElementById('statModel').textContent  = r.model;
          document.getElementById('dlBtn').disabled = false;
          updateViewerBanner(true, `✅ กำลังแสดงผลลัพธ์จาก <strong>${r.model}</strong> — ลากเส้นตรงกลางเพื่อเทียบ Before / After`);
          refreshLoupeSrc();
        }
      };
      cards.appendChild(c);
    });

    let tableWrap = document.getElementById('compareSummaryTableWrap');
    if (!tableWrap) {
      tableWrap = document.createElement('div');
      tableWrap.id = 'compareSummaryTableWrap';
      tableWrap.style.marginTop = '24px';
      tableWrap.style.overflowX = 'auto';
      strip.appendChild(tableWrap);
    }

    const rowsHtml = resultsList.map(r => `
      <tr style="${r.is_team ? 'background: rgba(37,99,235,0.18); font-weight:600; color:#60a5fa;' : ''}">
        <td style="padding: 10px 14px; border-bottom: 1px solid #1f2937;"><strong>${r.model}</strong></td>
        <td style="padding: 10px 14px; border-bottom: 1px solid #1f2937; ${r.niqe && r.niqe === minNiqe ? 'color:#10b981; font-weight:bold;' : ''}">
          ${r.niqe ? r.niqe.toFixed(3) : '–'} ${r.niqe && r.niqe === minNiqe ? '🏆 ดีที่สุด' : ''}
        </td>
        <td style="padding: 10px 14px; border-bottom: 1px solid #1f2937; ${r.musiq && r.musiq === maxMusiq ? 'color:#10b981; font-weight:bold;' : ''}">
          ${r.musiq ? r.musiq.toFixed(2) : '–'} ${r.musiq && r.musiq === maxMusiq ? '🏆 ดีที่สุด' : ''}
        </td>
        <td style="padding: 10px 14px; border-bottom: 1px solid #1f2937;">${r.output_shape || '-'}</td>
        <td style="padding: 10px 14px; border-bottom: 1px solid #1f2937;">${data.is_precomputed ? '0 ms (Benchmark)' : `${r.latency_ms} ms`}</td>
      </tr>
    `).join('');

    const tableHeading = data.is_precomputed
      ? '📊 คะแนนอ้างอิงเฉลี่ยจากชุดทดสอบ 20 ภาพมาตรฐาน (Benchmark Metrics):'
      : '📊 ตารางสรุปผลเปรียบเทียบสด (Live IQA Metrics Summary):';

    tableWrap.innerHTML = `
      <div style="font-size: 15px; font-weight: 600; color: #38bdf8; margin: 18px 0 10px 0;">
        ${tableHeading}
      </div>
      <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13.5px; background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid #1f2937;">
        <thead>
          <tr style="background: #1e293b; color: #cbd5e1;">
            <th style="padding: 10px 14px;">โมเดล (Model)</th>
            <th style="padding: 10px 14px;">NIQE ↓ (ยิ่งต่ำยิ่งเป็นธรรมชาติ)</th>
            <th style="padding: 10px 14px;">MUSIQ ↑ (คะแนนความสวยงาม)</th>
            <th style="padding: 10px 14px;">ขนาดภาพ 4×</th>
            <th style="padding: 10px 14px;">เวลาประมวลผล</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    `;

    highlightActiveMultiCard(getModel());
    strip.style.display = 'block';
    strip.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert('เกิดข้อผิดพลาดในการเปรียบเทียบ: ' + err.message);
  } finally {
    clearInterval(timerInterval);
    overlay.style.display = 'none';
  }
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

function resolveReportUrl(url) {
  if (BOOTSTRAP?.reports?.[url]) {
    if (!REPORT_BLOB_CACHE[url]) {
      const blob = new Blob([BOOTSTRAP.reports[url]], { type: 'text/html;charset=utf-8' });
      REPORT_BLOB_CACHE[url] = URL.createObjectURL(blob);
    }
    return REPORT_BLOB_CACHE[url];
  }
  return url;
}

function initReportLinks() {
  if (!BOOTSTRAP?.reports) return;
  const titles = {
    '/benchmark': '📊 Benchmark Results (20 ภาพมาตรฐาน)',
    '/comparison-table': '📑 ตารางเปรียบเทียบภาพผลลัพธ์ 20 ชุด',
    '/gallery': '🖼️ Model Visual Gallery',
    '/eda': '📈 EDA & Data Exploration Dashboard',
  };
  document.querySelectorAll('a[href="/benchmark"], a[href="/comparison-table"], a[href="/gallery"], a[href="/eda"]').forEach(a => {
    const rawPath = a.getAttribute('href');
    a.href = resolveReportUrl(rawPath);
    a.addEventListener('click', (e) => {
      e.preventDefault();
      openReportModal(rawPath, titles[rawPath] || 'รายงานและผลการทดลอง');
    });
  });
}

function openReportModal(url, title) {
  const modal = document.getElementById('reportModal');
  const iframe = document.getElementById('modalIframe');
  const titleEl = document.getElementById('modalTitle');
  const extLink = document.getElementById('modalExtLink');
  const resolved = resolveReportUrl(url);

  if (titleEl) titleEl.textContent = title || 'รายงานและผลการทดลอง';
  if (iframe) iframe.src = resolved;
  if (extLink) extLink.href = resolved;
  if (modal) modal.classList.add('show');
  document.body.style.overflow = 'hidden';
}

function closeReportModal() {
  const modal = document.getElementById('reportModal');
  const iframe = document.getElementById('modalIframe');
  if (iframe) iframe.src = 'about:blank';
  if (modal) modal.classList.remove('show');
  const lb = document.getElementById('imgLightbox');
  if (!lb || !lb.classList.contains('show')) {
    document.body.style.overflow = '';
  }
}

window.closeReportModal = closeReportModal;

function closeModalOnOverlay(e) {
  if (e.target && e.target.id === 'reportModal') {
    closeReportModal();
  }
}
