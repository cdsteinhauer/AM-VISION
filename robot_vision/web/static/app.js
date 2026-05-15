const state = {
  view: "rgb",
  mode: "live",
  recipe: null,
  activePanel: "recipePanel",
  calibrationRoi: [0.2, 0.2, 0.6, 0.6],
  lastCalibrationDetection: null,
  lastSnapshot: null,
  lastResultTools: [],
  busy: false,
  previewTimer: null,
  trainingTimer: null,
  lastLiveDebugAt: 0,
  cameraSettings: null,
  dragHandle: null,
  panDrag: null,
  zoom: 1,
  panX: 0,
  panY: 0,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${text}`);
  }
  return response.json();
}

function setStatus(id, text) {
  $(id).textContent = text;
}

function updateZoom() {
  document.documentElement.style.setProperty("--viewer-zoom", state.zoom);
  document.documentElement.style.setProperty("--viewer-pan-x", `${state.panX}px`);
  document.documentElement.style.setProperty("--viewer-pan-y", `${state.panY}px`);
  $("zoomValue").textContent = `${Math.round(state.zoom * 100)}%`;
  drawOverlay(state.lastResultTools || []);
}

function changeZoom(delta, anchor = null) {
  const previous = state.zoom;
  state.zoom = Math.max(0.5, Math.min(6, Number((state.zoom + delta).toFixed(2))));
  if (anchor && previous !== state.zoom) {
    const ratio = state.zoom / previous;
    state.panX = anchor.x - (anchor.x - state.panX) * ratio;
    state.panY = anchor.y - (anchor.y - state.panY) * ratio;
  }
  updateZoom();
}

function resetZoom() {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  updateZoom();
}

function previewIntervalMs() {
  if (state.mode === "live") return 25;
  if (state.mode === "capture") return 1000;
  return 0;
}

function startPreviewLoop(immediate = false) {
  if (state.previewTimer) {
    clearTimeout(state.previewTimer);
    state.previewTimer = null;
  }
  const delay = immediate ? 0 : previewIntervalMs();
  if (!delay && !immediate) return;
  state.previewTimer = setTimeout(runPreviewLoop, delay);
}

async function runPreviewLoop() {
  state.previewTimer = null;
  if (state.mode !== "live" && state.mode !== "capture") return;
  await preview().catch((error) => setStatus("resultStatus", `Camera: ${error.message}`));
  startPreviewLoop(false);
}

async function refreshHealth() {
  try {
    const health = await api("/api/health");
    setStatus("healthStatus", `Service: OK / ROS ${health.ros_domain_id}`);
    const camera = await api("/api/camera/status");
    setStatus("cameraStatus", `Camera: ${camera.provider}${camera.error ? " error" : ""}`);
  } catch (error) {
    setStatus("healthStatus", `Service: ${error.message}`);
  }
}

async function loadCameraSettings() {
  const container = $("cameraSettings");
  try {
    const data = await api("/api/camera/settings");
    state.cameraSettings = data;
    renderCameraSettings(data.controls || []);
  } catch (error) {
    container.textContent = `Camera settings unavailable: ${error.message}`;
  }
}

function renderCameraSettings(controls) {
  const container = $("cameraSettings");
  container.innerHTML = "";
  if (!controls.length) {
    container.textContent = "No adjustable camera settings found.";
    return;
  }
  controls.forEach((control) => {
    const row = document.createElement("div");
    row.className = "setting-row";
    const label = document.createElement("span");
    label.textContent = control.label || control.name;
    row.appendChild(label);

    const input = buildSettingInput(control);
    row.appendChild(input);

    const value = document.createElement("span");
    value.dataset.settingValue = control.name;
    value.textContent = displaySettingValue(control, input);
    row.appendChild(value);

    input.addEventListener("input", () => {
      value.textContent = displaySettingValue(control, input);
    });
    container.appendChild(row);
  });
}

function buildSettingInput(control) {
  if (control.kind === "bool") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.setting = control.name;
    input.checked = Number(control.value) === 1;
    return input;
  }
  const options = control.options || {};
  if (Object.keys(options).length) {
    const select = document.createElement("select");
    select.dataset.setting = control.name;
    Object.entries(options).forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    });
    select.value = String(control.value ?? control.default ?? "");
    return select;
  }
  const input = document.createElement("input");
  input.type = "range";
  input.dataset.setting = control.name;
  input.min = control.min ?? 0;
  input.max = control.max ?? 100;
  input.step = control.step || 1;
  input.value = control.value ?? control.default ?? input.min;
  return input;
}

function displaySettingValue(control, input) {
  if (control.kind === "bool") return input.checked ? "On" : "Off";
  const options = control.options || {};
  if (Object.keys(options).length) return options[input.value] || input.value;
  return input.value;
}

function readCameraSettingsFromUi() {
  const updates = {};
  document.querySelectorAll("[data-setting]").forEach((input) => {
    if (input.type === "checkbox") {
      updates[input.dataset.setting] = input.checked ? 1 : 0;
    } else {
      updates[input.dataset.setting] = Number(input.value);
    }
  });
  return updates;
}

async function applyCameraSettings() {
  const data = await api("/api/camera/settings", {
    method: "POST",
    body: JSON.stringify({ settings: readCameraSettingsFromUi() }),
  });
  state.cameraSettings = data;
  renderCameraSettings(data.controls || []);
  const errorCount = Object.keys(data.errors || {}).length;
  setStatus("cameraStatus", errorCount ? `Camera settings: ${errorCount} rejected` : "Camera settings: applied");
}

async function loadRecipes(selectedName = null) {
  const data = await api("/api/recipes");
  const select = $("recipeSelect");
  const previous = selectedName || select.value || state.recipe?.name || "default";
  select.innerHTML = "";
  data.recipes.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });
  select.value = data.recipes.includes(previous) ? previous : (data.recipes[0] || "default");
  await loadRecipe(select.value || "default");
}

async function loadRecipe(name) {
  const data = await api(`/api/recipes/${encodeURIComponent(name)}`);
  state.recipe = data.recipe;
  $("recipeName").value = state.recipe.name;
  $("recipeDescription").value = state.recipe.description || "";
  renderTools();
}

function renderTools() {
  const list = $("toolList");
  list.innerHTML = "";
  (state.recipe.tools || []).forEach((tool, index) => {
    const card = document.createElement("div");
    card.className = "tool-card";
    card.innerHTML = `
      <div class="tool-card-header">
        <button class="tool-collapse" data-toggle-tool="${index}" type="button">▾</button>
        <strong>${tool.name}</strong>
        <button class="danger small" data-delete-tool="${index}">Delete</button>
      </div>
      <div class="tool-card-body">
      <label>Name <input data-tool="${index}" data-field="name" value="${tool.name}"></label>
      <label>Type <select data-tool="${index}" data-field="type">
        <option value="rectangle">rectangle</option>
        <option value="edge_1">edge check 1 edge</option>
        <option value="edge_2">edge check 2 edge</option>
        <option value="ai_classifier">AI classifier</option>
      </select></label>
      <label class="inline-toggle"><input type="checkbox" data-tool="${index}" data-field="enabled" ${tool.enabled === false ? "" : "checked"}> Tool enabled</label>
      <label class="inline-toggle"><input type="checkbox" data-tool="${index}" data-field="debug" ${tool.debug ? "checked" : ""}> Debug lines</label>
      <p class="help-text">Search area: the app detects the rectangle inside this box, then measures the detected rectangle using calibration.</p>
      <div class="tool-grid">
        <label>Search X <input type="number" step="0.01" data-tool="${index}" data-roi="0" value="${tool.roi[0]}"></label>
        <label>Search Y <input type="number" step="0.01" data-tool="${index}" data-roi="1" value="${tool.roi[1]}"></label>
        <label>Search W <input type="number" step="0.01" data-tool="${index}" data-roi="2" value="${tool.roi[2]}"></label>
        <label>Search H <input type="number" step="0.01" data-tool="${index}" data-roi="3" value="${tool.roi[3]}"></label>
      </div>
      <div class="tool-grid">
        <label>Min W <input type="number" data-tool="${index}" data-field="min_width_mm" value="${tool.min_width_mm ?? ""}"></label>
        <label>Max W <input type="number" data-tool="${index}" data-field="max_width_mm" value="${tool.max_width_mm ?? ""}"></label>
        <label>Min H <input type="number" data-tool="${index}" data-field="min_height_mm" value="${tool.min_height_mm ?? ""}"></label>
        <label>Max H <input type="number" data-tool="${index}" data-field="max_height_mm" value="${tool.max_height_mm ?? ""}"></label>
      </div>
      <div class="tool-grid">
        <label>Min Line <input type="number" data-tool="${index}" data-field="min_length_mm" value="${tool.min_length_mm ?? ""}"></label>
        <label>Max Line <input type="number" data-tool="${index}" data-field="max_length_mm" value="${tool.max_length_mm ?? ""}"></label>
        <label>Line Dir <select data-tool="${index}" data-field="line_orientation">
          <option value="auto">auto</option>
          <option value="horizontal">horizontal</option>
          <option value="vertical">vertical</option>
        </select></label>
        <label>Line Score <input type="number" data-tool="${index}" data-field="min_edge_score" value="${tool.min_edge_score ?? 25}"></label>
      </div>
      <div class="tool-grid">
        <label>Min Line Ratio <input type="number" step="0.01" data-tool="${index}" data-field="min_line_length_ratio" value="${tool.min_line_length_ratio ?? 0.15}"></label>
      </div>
      <div class="tool-grid">
        <label>Model Dir <input data-tool="${index}" data-field="model_dir" value="${tool.model_dir ?? "data/models/pass_fail_classifier"}"></label>
        <label>Min AI Conf % <input type="number" step="1" data-tool="${index}" data-field="min_confidence" value="${confidenceToPercent(tool.min_confidence ?? 0.8)}"></label>
      </div>
      </div>
    `;
    card.querySelector("select").value = tool.type;
    const orientationSelect = card.querySelector('[data-field="line_orientation"]');
    if (orientationSelect) orientationSelect.value = tool.line_orientation || "auto";
    list.appendChild(card);
  });
  document.querySelectorAll("[data-delete-tool]").forEach((button) => {
    button.addEventListener("click", () => deleteTool(Number(button.dataset.deleteTool)));
  });
  document.querySelectorAll("[data-toggle-tool]").forEach((button) => {
    button.addEventListener("click", () => toggleToolCard(Number(button.dataset.toggleTool)));
  });
  document.querySelectorAll("[data-tool]").forEach((input) => {
    input.addEventListener("input", () => {
      readRecipeFromUi();
      state.lastResultTools = activeResultTools();
      drawOverlay(state.lastResultTools);
    });
  });
  drawOverlay(state.lastResultTools || []);
}

function toggleToolCard(index) {
  const card = document.querySelectorAll(".tool-card")[index];
  if (!card) return;
  card.classList.toggle("collapsed");
  const button = card.querySelector(".tool-collapse");
  if (button) button.textContent = card.classList.contains("collapsed") ? "▸" : "▾";
}

function confidenceToPercent(value) {
  const confidence = Number(value);
  if (confidence <= 1) return Math.round(confidence * 100);
  return Math.round(confidence);
}

function readRecipeFromUi() {
  state.recipe.name = $("recipeName").value || "default";
  state.recipe.description = $("recipeDescription").value || "";
  document.querySelectorAll("[data-tool]").forEach((input) => {
    const tool = state.recipe.tools[Number(input.dataset.tool)];
    if (input.dataset.roi !== undefined) {
      tool.roi[Number(input.dataset.roi)] = Number(input.value);
      return;
    }
    const field = input.dataset.field;
    if (field === "name" || field === "type" || field === "model_dir") {
      tool[field] = input.value;
    } else if (field === "enabled") {
      tool.enabled = input.checked;
    } else if (field === "debug") {
      tool.debug = input.checked;
    } else if (field) {
      tool[field] = input.value === "" ? null : Number(input.value);
    }
  });
}

function activeResultTools() {
  if (!state.recipe) return [];
  const enabledIds = new Set((state.recipe.tools || []).filter((tool) => tool.enabled !== false).map((tool) => tool.id));
  return (state.lastResultTools || []).filter((tool) => enabledIds.has(tool.tool_id));
}

async function snap() {
  if (state.busy) return;
  state.busy = true;
  try {
    const data = await api("/api/camera/snapshot", { method: "POST", body: "{}" });
    state.lastSnapshot = data;
    $("rgbImage").src = `data:image/png;base64,${data.rgb_png}`;
    if (data.depth_png) $("depthImage").src = `data:image/png;base64,${data.depth_png}`;
    state.lastResultTools = [];
    drawOverlay([]);
    if ($("autoSnap").checked && data.auto_trigger.fired) {
      state.busy = false;
      await inspect();
      return;
    }
  } finally {
    state.busy = false;
  }
}

async function preview() {
  if (state.busy) return;
  state.busy = true;
  try {
    const processTrigger = state.mode === "capture";
    const data = await api(`/api/camera/preview?view=${state.view}&process_trigger=${processTrigger}`);
    $("rgbImage").src = `data:image/jpeg;base64,${data.rgb_jpg}`;
    if (data.depth_jpg) $("depthImage").src = `data:image/jpeg;base64,${data.depth_jpg}`;
    if (!state.lastResultTools.length) drawOverlay([]);
    if (state.mode === "capture" && $("autoSnap").checked && data.auto_trigger.fired) {
      state.busy = false;
      await inspect();
    } else if ($("liveDebug").checked && state.mode !== "snap" && Date.now() - state.lastLiveDebugAt > 1000) {
      state.lastLiveDebugAt = Date.now();
      state.busy = false;
      await debugDetect();
    }
  } finally {
    state.busy = false;
  }
}

async function inspect() {
  if (state.busy) return;
  state.busy = true;
  try {
  readRecipeFromUi();
  const recipeName = state.recipe.name;
  await saveRecipe(false, false);
  const data = await api("/api/inspect", {
    method: "POST",
    body: JSON.stringify({ recipe_name: recipeName, calibration_name: "default", save_report: true }),
  });
  $("rgbImage").src = `data:image/png;base64,${data.rgb_png}`;
  if (data.depth_png) $("depthImage").src = `data:image/png;base64,${data.depth_png}`;
  $("resultBox").textContent = JSON.stringify(data.result, null, 2);
  setStatus("resultStatus", `Result: ${data.result.passed ? "PASS" : "FAIL"}`);
  state.lastResultTools = data.result.tools || [];
  drawOverlay(data.result.tools || []);
  await loadReports();
  } finally {
    state.busy = false;
  }
}

async function debugDetect() {
  if (state.busy) return;
  state.busy = true;
  try {
    readRecipeFromUi();
    const recipeName = state.recipe.name;
    await saveRecipe(false, false);
    const data = await api("/api/inspect", {
      method: "POST",
      body: JSON.stringify({ recipe_name: recipeName, calibration_name: "default", save_report: false }),
    });
    $("rgbImage").src = `data:image/png;base64,${data.rgb_png}`;
    if (data.depth_png) $("depthImage").src = `data:image/png;base64,${data.depth_png}`;
    $("resultBox").textContent = JSON.stringify(data.result, null, 2);
    setStatus("resultStatus", `Debug: ${data.result.passed ? "PASS" : "FAIL"}`);
    state.lastResultTools = data.result.tools || [];
    drawOverlay(data.result.tools || []);
  } finally {
    state.busy = false;
  }
}

function drawOverlay(tools) {
  const img = state.view === "depth" ? $("depthImage") : $("rgbImage");
  const canvas = $("overlayCanvas");
  if (!img.complete || !img.naturalWidth) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawRecipeRois(ctx, canvas.width, canvas.height);
  drawCalibrationOverlay(ctx, canvas.width, canvas.height);
  ctx.lineWidth = 5;
  ctx.font = "22px Segoe UI";
  activeResultToolsFrom(tools).forEach((tool) => {
    const debugLines = tool.measurements?.debug_lines || [];
    if (debugLines.length) {
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(33, 118, 255, 0.8)";
      debugLines.forEach((candidate) => {
        const line = candidate.line;
        ctx.beginPath();
        ctx.moveTo(line[0], line[1]);
        ctx.lineTo(line[2], line[3]);
        ctx.stroke();
      });
      ctx.restore();
    }
    const lineA = tool.measurements?.line_a;
    const lineB = tool.measurements?.line_b;
    if (lineA) {
      ctx.strokeStyle = tool.passed ? "#159650" : "#cf342b";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.moveTo(lineA[0], lineA[1]);
      ctx.lineTo(lineA[2], lineA[3]);
      if (lineB) {
        ctx.moveTo(lineB[0], lineB[1]);
        ctx.lineTo(lineB[2], lineB[3]);
      }
      ctx.stroke();
      ctx.fillText(tool.name, lineA[0] + 8, Math.max(24, lineA[1] - 10));
      return;
    }
    if (!tool.bbox_px) return;
    const [x0, y0, x1, y1] = tool.bbox_px;
    ctx.strokeStyle = tool.passed ? "#159650" : "#cf342b";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillText(tool.name, x0 + 8, Math.max(24, y0 - 10));
  });
}

function activeResultToolsFrom(tools) {
  if (!state.recipe) return tools || [];
  const enabledIds = new Set((state.recipe.tools || []).filter((tool) => tool.enabled !== false).map((tool) => tool.id));
  return (tools || []).filter((tool) => enabledIds.has(tool.tool_id));
}

function drawCalibrationOverlay(ctx, width, height) {
  if (state.activePanel !== "calibrationPanel") return;
  ctx.save();
  const box = roiToBox(readCalibrationRoiFromUi(), width, height);
  ctx.lineWidth = 3;
  ctx.font = "18px Segoe UI";
  ctx.strokeStyle = "#2176ff";
  ctx.fillStyle = "rgba(33, 118, 255, 0.12)";
  ctx.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
  ctx.fillRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
  ctx.fillStyle = "#2176ff";
  ctx.fillText("Calibration search area", box.x0 + 8, box.y0 + 22);
  cornerPoints(box).forEach((point) => {
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#2176ff";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.rect(point.x - 7, point.y - 7, 14, 14);
    ctx.fill();
    ctx.stroke();
  });
  if (state.lastCalibrationDetection?.bbox_px) {
    const [x0, y0, x1, y1] = state.lastCalibrationDetection.bbox_px;
    ctx.strokeStyle = "#00a8a8";
    ctx.fillStyle = "#00a8a8";
    ctx.lineWidth = 5;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillText("Detected calibration rectangle", x0 + 8, Math.max(24, y0 - 10));
  }
  ctx.restore();
}

function drawRecipeRois(ctx, width, height) {
  if (!state.recipe) return;
  ctx.save();
  ctx.lineWidth = 3;
  ctx.font = "18px Segoe UI";
  (state.recipe.tools || []).forEach((tool, index) => {
    if (tool.enabled === false) return;
    const box = roiToBox(tool.roi, width, height);
    ctx.strokeStyle = "#ff7a1a";
    ctx.fillStyle = "rgba(255, 122, 26, 0.14)";
    ctx.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
    ctx.fillRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
    ctx.fillStyle = "#ff7a1a";
    ctx.fillText(`${tool.name || `Tool ${index + 1}`} search area`, box.x0 + 8, box.y0 + 22);
    cornerPoints(box).forEach((point) => {
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "#ff7a1a";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.rect(point.x - 7, point.y - 7, 14, 14);
      ctx.fill();
      ctx.stroke();
    });
  });
  ctx.restore();
}

function roiToBox(roi, width, height) {
  const x0 = roi[0] * width;
  const y0 = roi[1] * height;
  return {
    x0,
    y0,
    x1: x0 + roi[2] * width,
    y1: y0 + roi[3] * height,
  };
}

function boxToRoi(box, width, height) {
  const x0 = Math.max(0, Math.min(width - 1, Math.min(box.x0, box.x1)));
  const y0 = Math.max(0, Math.min(height - 1, Math.min(box.y0, box.y1)));
  const x1 = Math.max(x0 + 1, Math.min(width, Math.max(box.x0, box.x1)));
  const y1 = Math.max(y0 + 1, Math.min(height, Math.max(box.y0, box.y1)));
  return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height].map((value) => Number(value.toFixed(4)));
}

function cornerPoints(box) {
  return [
    { corner: "nw", x: box.x0, y: box.y0 },
    { corner: "ne", x: box.x1, y: box.y0 },
    { corner: "sw", x: box.x0, y: box.y1 },
    { corner: "se", x: box.x1, y: box.y1 },
  ];
}

function canvasPoint(event) {
  const canvas = $("overlayCanvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function readCalibrationRoiFromUi() {
  state.calibrationRoi = [
    Number($("calRoiX").value),
    Number($("calRoiY").value),
    Number($("calRoiW").value),
    Number($("calRoiH").value),
  ];
  return state.calibrationRoi;
}

function updateCalibrationRoiInputs() {
  $("calRoiX").value = state.calibrationRoi[0];
  $("calRoiY").value = state.calibrationRoi[1];
  $("calRoiW").value = state.calibrationRoi[2];
  $("calRoiH").value = state.calibrationRoi[3];
}

function findDragHandle(point) {
  const canvas = $("overlayCanvas");
  if (state.activePanel === "calibrationPanel" && canvas.width && canvas.height) {
    const calibrationBox = roiToBox(readCalibrationRoiFromUi(), canvas.width, canvas.height);
    for (const corner of cornerPoints(calibrationBox)) {
      if (Math.abs(point.x - corner.x) <= 18 && Math.abs(point.y - corner.y) <= 18) {
        return { calibration: true, corner: corner.corner };
      }
    }
    if (pointInBox(point, calibrationBox)) {
      return { calibration: true, move: true, startPoint: point, startRoi: [...state.calibrationRoi] };
    }
  }
  if (!state.recipe || !canvas.width || !canvas.height) return null;
  const hitRadius = 18;
  for (let toolIndex = 0; toolIndex < state.recipe.tools.length; toolIndex += 1) {
    const tool = state.recipe.tools[toolIndex];
    if (tool.enabled === false) continue;
    const box = roiToBox(tool.roi, canvas.width, canvas.height);
    for (const corner of cornerPoints(box)) {
      if (Math.abs(point.x - corner.x) <= hitRadius && Math.abs(point.y - corner.y) <= hitRadius) {
        return { toolIndex, corner: corner.corner };
      }
    }
    if (pointInBox(point, box)) {
      return { toolIndex, move: true, startPoint: point, startRoi: [...tool.roi] };
    }
  }
  return null;
}

function pointInBox(point, box) {
  return point.x >= box.x0 && point.x <= box.x1 && point.y >= box.y0 && point.y <= box.y1;
}

function updateToolRoiInputs(toolIndex) {
  const tool = state.recipe.tools[toolIndex];
  document.querySelectorAll(`[data-tool="${toolIndex}"][data-roi]`).forEach((input) => {
    input.value = tool.roi[Number(input.dataset.roi)];
  });
}

function dragRectangleCorner(handle, point) {
  const canvas = $("overlayCanvas");
  if (handle.calibration) {
    if (handle.move) {
      state.calibrationRoi = moveRoi(handle.startRoi, handle.startPoint, point, canvas.width, canvas.height);
      state.lastCalibrationDetection = null;
      updateCalibrationRoiInputs();
      drawOverlay(state.lastResultTools || []);
      return;
    }
    const box = roiToBox(readCalibrationRoiFromUi(), canvas.width, canvas.height);
    if (handle.corner.includes("n")) box.y0 = point.y;
    if (handle.corner.includes("s")) box.y1 = point.y;
    if (handle.corner.includes("w")) box.x0 = point.x;
    if (handle.corner.includes("e")) box.x1 = point.x;
    state.calibrationRoi = boxToRoi(box, canvas.width, canvas.height);
    state.lastCalibrationDetection = null;
    updateCalibrationRoiInputs();
    drawOverlay(state.lastResultTools || []);
    return;
  }
  const tool = state.recipe.tools[handle.toolIndex];
  if (handle.move) {
    tool.roi = moveRoi(handle.startRoi, handle.startPoint, point, canvas.width, canvas.height);
    updateToolRoiInputs(handle.toolIndex);
    state.lastResultTools = [];
    drawOverlay([]);
    return;
  }
  const box = roiToBox(tool.roi, canvas.width, canvas.height);
  if (handle.corner.includes("n")) box.y0 = point.y;
  if (handle.corner.includes("s")) box.y1 = point.y;
  if (handle.corner.includes("w")) box.x0 = point.x;
  if (handle.corner.includes("e")) box.x1 = point.x;
  tool.roi = boxToRoi(box, canvas.width, canvas.height);
  updateToolRoiInputs(handle.toolIndex);
  state.lastResultTools = [];
  drawOverlay([]);
}

function moveRoi(startRoi, startPoint, point, canvasWidth, canvasHeight) {
  const dx = (point.x - startPoint.x) / canvasWidth;
  const dy = (point.y - startPoint.y) / canvasHeight;
  const width = startRoi[2];
  const height = startRoi[3];
  const x = Math.max(0, Math.min(1 - width, startRoi[0] + dx));
  const y = Math.max(0, Math.min(1 - height, startRoi[1] + dy));
  return [x, y, width, height].map((value) => Number(value.toFixed(4)));
}

async function saveRecipe(showResult = true, reloadList = true) {
  readRecipeFromUi();
  const recipeName = state.recipe.name;
  const data = await api("/api/recipes", {
    method: "POST",
    body: JSON.stringify({ recipe: state.recipe }),
  });
  state.recipe = data.recipe;
  if (showResult) $("resultBox").textContent = `Saved recipe ${state.recipe.name}`;
  if (reloadList) await loadRecipes(recipeName);
}

async function deleteRecipe() {
  const name = state.recipe?.name || $("recipeSelect").value;
  if (!name) return;
  if (!window.confirm(`Delete recipe "${name}"? This cannot be undone.`)) return;
  const data = await api(`/api/recipes/${encodeURIComponent(name)}`, { method: "DELETE" });
  $("resultBox").textContent = `Deleted recipe ${name}`;
  state.lastResultTools = [];
  const nextRecipe = data.recipes.includes("default") ? "default" : data.recipes[0];
  await loadRecipes(nextRecipe);
  drawOverlay([]);
}

async function calibrate() {
  const data = await api("/api/calibration/run", {
    method: "POST",
    body: JSON.stringify({
      name: "default",
      pixel_width: Number($("pixelWidth").value),
      pixel_height: Number($("pixelHeight").value),
      real_width_mm: Number($("realWidth").value),
      real_height_mm: Number($("realHeight").value),
    }),
  });
  $("resultBox").textContent = JSON.stringify(data.calibration, null, 2);
  $("calibrationStatus").textContent = `Manual calibration saved: ${data.calibration.pixels_per_mm_x.toFixed(3)} px/mm X, ${data.calibration.pixels_per_mm_y.toFixed(3)} px/mm Y`;
}

async function loadCalibration(name = "default") {
  try {
    const data = await api(`/api/calibration/${encodeURIComponent(name)}`);
    const profile = data.calibration;
    if (profile.pixel_width) $("pixelWidth").value = profile.pixel_width;
    if (profile.pixel_height) $("pixelHeight").value = profile.pixel_height;
    if (profile.real_width_mm) $("realWidth").value = profile.real_width_mm;
    if (profile.real_height_mm) $("realHeight").value = profile.real_height_mm;
    $("calibrationStatus").textContent = `Loaded calibration: ${Number(profile.pixels_per_mm_x).toFixed(3)} px/mm X, ${Number(profile.pixels_per_mm_y).toFixed(3)} px/mm Y.`;
    return profile;
  } catch (error) {
    $("calibrationStatus").textContent = `Calibration not loaded: ${error.message}`;
    return null;
  }
}

async function detectCalibration() {
  if (state.busy) return;
  state.busy = true;
  try {
    const data = await api("/api/calibration/detect", {
      method: "POST",
      body: JSON.stringify({
        name: "default",
        roi: readCalibrationRoiFromUi(),
        real_width_mm: Number($("realWidth").value),
        real_height_mm: Number($("realHeight").value),
      }),
    });
    state.lastCalibrationDetection = data.detection;
    $("pixelWidth").value = data.detection.width_px;
    $("pixelHeight").value = data.detection.height_px;
    $("rgbImage").src = `data:image/png;base64,${data.rgb_png}`;
    if (data.depth_png) $("depthImage").src = `data:image/png;base64,${data.depth_png}`;
    $("resultBox").textContent = JSON.stringify({
      calibration: data.calibration,
      detection: data.detection,
    }, null, 2);
    $("calibrationStatus").textContent = `Detected ${data.detection.width_px} x ${data.detection.height_px} px. Saved ${data.calibration.pixels_per_mm_x.toFixed(3)} px/mm X, ${data.calibration.pixels_per_mm_y.toFixed(3)} px/mm Y.`;
    drawOverlay(state.lastResultTools || []);
  } finally {
    state.busy = false;
  }
}

async function loadReports() {
  const data = await api("/api/reports");
  $("reportList").innerHTML = data.reports.slice(0, 8).map((report) => `
    <div>
      <span class="${report.passed ? "pass" : "fail"}">${report.passed ? "PASS" : "FAIL"}</span>
      ${report.id} ${report.recipe}
    </div>
  `).join("") || "No reports yet.";
}

async function refreshTrainingSamples() {
  const recipe = $("trainRecipe").value.trim();
  const source = $("trainSource").value;
  const dataset = $("trainDataset").value.trim();
  const params = new URLSearchParams({ source });
  if (recipe) params.set("recipe", recipe);
  if (dataset) params.set("dataset", dataset);
  const query = `?${params.toString()}`;
  const data = await api(`/api/training/samples${query}`);
  $("trainTotal").textContent = data.total;
  $("trainPass").textContent = data.counts.PASS || 0;
  $("trainFail").textContent = data.counts.FAIL || 0;
  $("trainingStatus").textContent = data.ready
    ? "Training data ready."
    : "Training needs at least one PASS and one FAIL report.";
  return data;
}

async function captureTrainingSample(label) {
  const dataset = $("trainDataset").value.trim() || "default_part";
  const data = await api("/api/training/capture", {
    method: "POST",
    body: JSON.stringify({ dataset, label }),
  });
  $("rgbImage").src = `data:image/png;base64,${data.rgb_png}`;
  if (data.depth_png) $("depthImage").src = `data:image/png;base64,${data.depth_png}`;
  $("captureStatus").textContent = `Captured ${label}: PASS ${data.sample.counts.PASS}, FAIL ${data.sample.counts.FAIL}`;
  $("trainSource").value = "captures";
  await refreshTrainingSamples();
}

async function refreshTrainingDatasets() {
  const data = await api("/api/training/datasets");
  $("captureStatus").textContent = `Datasets: ${data.datasets.map((item) => `${item.name} (${item.total})`).join(", ") || "none"}`;
  await refreshTrainingSamples();
}

async function refreshTrainingStatus() {
  const data = await api("/api/training/status");
  renderTrainingStatus(data);
  if (data.running && !state.trainingTimer) {
    state.trainingTimer = setInterval(() => {
      refreshTrainingStatus().catch((error) => {
        $("trainingStatus").textContent = `Training status error: ${error.message}`;
      });
    }, 2500);
  }
  if (!data.running && state.trainingTimer) {
    clearInterval(state.trainingTimer);
    state.trainingTimer = null;
  }
  return data;
}

async function refreshTrainingDependencies() {
  const data = await api("/api/training/dependencies");
  if (data.ok) {
    $("trainingDepsStatus").textContent = "Training dependencies: installed.";
  } else {
    $("trainingDepsStatus").textContent = `Training dependencies missing: ${data.missing.join(", ")}. Run: ${data.install_command}`;
  }
  return data;
}

function renderTrainingStatus(data) {
  const suffix = data.error ? ` ${data.error}` : "";
  $("trainingStatus").textContent = `Training: ${data.state}. ${data.message || ""}${suffix}`;
  $("trainingBox").textContent = JSON.stringify(data.manifest || data, null, 2);
}

async function startTraining() {
  const sampleData = await refreshTrainingSamples();
  if (!sampleData.ready) {
    $("trainingBox").textContent = "Training requires at least one PASS and one FAIL report.";
    return;
  }
  const data = await api("/api/training/start", {
    method: "POST",
    body: JSON.stringify({
      recipe: $("trainRecipe").value.trim() || null,
      dataset: $("trainDataset").value.trim() || null,
      source: $("trainSource").value,
      model: $("trainModel").value.trim() || "microsoft/resnet-18",
      output_dir: $("trainOutput").value.trim() || "data/models/pass_fail_classifier",
      validation_fraction: Number($("trainValFraction").value),
      epochs: Number($("trainEpochs").value),
      batch_size: Number($("trainBatch").value),
      learning_rate: Number($("trainLearningRate").value),
    }),
  });
  renderTrainingStatus(data);
  await refreshTrainingStatus();
}

async function deleteAllSamples() {
  if (!window.confirm("Delete all saved reports/training samples? This cannot be undone.")) return;
  const data = await api("/api/reports?confirm=DELETE", { method: "DELETE" });
  $("trainingStatus").textContent = `Deleted ${data.deleted} samples.`;
  $("trainingBox").textContent = JSON.stringify(data, null, 2);
  await loadReports();
  await refreshTrainingSamples();
}

function addTool(type) {
  readRecipeFromUi();
  const id = `${type}_${Date.now()}`;
  state.recipe.tools.push({
    id,
    name: toolDefaultName(type),
    type,
    roi: [0.2, 0.2, 0.6, 0.6],
    enabled: true,
    min_width_mm: type === "rectangle" ? 20 : null,
    max_width_mm: type === "rectangle" ? 1000 : null,
    min_height_mm: type === "rectangle" ? 20 : null,
    max_height_mm: type === "rectangle" ? 1000 : null,
    min_edge_score: 25,
    min_length_mm: type === "edge_1" || type === "edge_2" ? 20 : null,
    max_length_mm: type === "edge_1" || type === "edge_2" ? 1000 : null,
    line_orientation: "auto",
    debug: false,
    min_line_length_ratio: 0.15,
    model_dir: "data/models/pass_fail_classifier",
    min_confidence: 0.8,
  });
  renderTools();
}

function toolDefaultName(type) {
  if (type === "edge_1") return "Edge check 1 edge";
  if (type === "edge_2") return "Edge check 2 edge";
  if (type === "ai_classifier") return "AI classifier";
  return "Rectangle check";
}

function deleteTool(index) {
  if (!state.recipe || !state.recipe.tools[index]) return;
  state.recipe.tools.splice(index, 1);
  state.lastResultTools = [];
  renderTools();
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab === button));
    $("rgbImage").hidden = state.view !== "rgb";
    $("depthImage").hidden = state.view !== "depth";
  });
});

document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll(".mode-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
    if (state.mode === "snap") {
      setStatus("resultStatus", "Mode: Snap Capture");
      startPreviewLoop(false);
    } else if (state.mode === "capture") {
      setStatus("resultStatus", "Mode: Live Capture");
      startPreviewLoop(true);
    } else {
      setStatus("resultStatus", "Mode: Live View");
      startPreviewLoop(true);
    }
  });
});

document.querySelectorAll(".panel-tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.activePanel = button.dataset.panel;
    document.querySelectorAll(".panel-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
    document.querySelectorAll(".panel-page").forEach((page) => {
      page.classList.toggle("active", page.id === button.dataset.panel);
    });
    drawOverlay(state.lastResultTools || []);
  });
});

$("overlayCanvas").addEventListener("pointerdown", (event) => {
  const handle = findDragHandle(canvasPoint(event));
  if (handle) {
    state.dragHandle = handle;
  } else {
    state.panDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      panX: state.panX,
      panY: state.panY,
    };
  }
  $("overlayCanvas").setPointerCapture(event.pointerId);
  event.preventDefault();
});

$("overlayCanvas").addEventListener("pointermove", (event) => {
  if (state.panDrag) {
    state.panX = state.panDrag.panX + event.clientX - state.panDrag.startX;
    state.panY = state.panDrag.panY + event.clientY - state.panDrag.startY;
    updateZoom();
    event.preventDefault();
    return;
  }
  if (!state.dragHandle) {
    $("overlayCanvas").style.cursor = findDragHandle(canvasPoint(event)) ? "nwse-resize" : "grab";
    return;
  }
  dragRectangleCorner(state.dragHandle, canvasPoint(event));
  event.preventDefault();
});

$("overlayCanvas").addEventListener("pointerup", (event) => {
  state.dragHandle = null;
  state.panDrag = null;
  $("overlayCanvas").releasePointerCapture(event.pointerId);
});

$("overlayCanvas").addEventListener("pointercancel", () => {
  state.dragHandle = null;
  state.panDrag = null;
});

$("overlayCanvas").addEventListener("wheel", (event) => {
  const rect = $("overlayCanvas").getBoundingClientRect();
  const anchor = {
    x: event.clientX - rect.left - rect.width / 2,
    y: event.clientY - rect.top - rect.height / 2,
  };
  changeZoom(event.deltaY < 0 ? 0.25 : -0.25, anchor);
  event.preventDefault();
});

["rgbImage", "depthImage"].forEach((id) => {
  $(id).addEventListener("load", () => drawOverlay(state.lastResultTools || []));
});

$("snapButton").addEventListener("click", snap);
$("inspectButton").addEventListener("click", inspect);
$("debugDetectButton").addEventListener("click", debugDetect);
$("zoomOutButton").addEventListener("click", () => changeZoom(-0.25));
$("zoomInButton").addEventListener("click", () => changeZoom(0.25));
$("zoomResetButton").addEventListener("click", resetZoom);
$("saveRecipeButton").addEventListener("click", () => saveRecipe(true));
$("deleteRecipeButton").addEventListener("click", () => deleteRecipe().catch((error) => {
  $("resultBox").textContent = `Delete recipe failed: ${error.message}`;
}));
$("calibrateButton").addEventListener("click", calibrate);
$("detectCalibrationButton").addEventListener("click", () => detectCalibration().catch((error) => {
  if (error.message.startsWith("404 ")) {
    $("calibrationStatus").textContent = "Calibration detect failed: backend is still running old code. Restart Robot Vision, then reload this page.";
  } else {
    $("calibrationStatus").textContent = `Calibration detect failed: ${error.message}`;
  }
  setStatus("resultStatus", "Calibration: failed");
}));
$("refreshCameraSettingsButton").addEventListener("click", loadCameraSettings);
$("applyCameraSettingsButton").addEventListener("click", () => applyCameraSettings().catch((error) => setStatus("cameraStatus", `Settings: ${error.message}`)));
$("refreshTrainingButton").addEventListener("click", () => refreshTrainingSamples().catch((error) => {
  $("trainingStatus").textContent = `Training samples failed: ${error.message}`;
}));
$("capturePassButton").addEventListener("click", () => captureTrainingSample("PASS").catch((error) => {
  $("captureStatus").textContent = `Capture PASS failed: ${error.message}`;
}));
$("captureFailButton").addEventListener("click", () => captureTrainingSample("FAIL").catch((error) => {
  $("captureStatus").textContent = `Capture FAIL failed: ${error.message}`;
}));
$("refreshDatasetsButton").addEventListener("click", () => refreshTrainingDatasets().catch((error) => {
  $("captureStatus").textContent = `Refresh datasets failed: ${error.message}`;
}));
$("startTrainingButton").addEventListener("click", () => startTraining().catch((error) => {
  $("trainingStatus").textContent = `Training start failed: ${error.message}`;
  $("trainingBox").textContent = error.message;
}));
$("deleteSamplesButton").addEventListener("click", () => deleteAllSamples().catch((error) => {
  $("trainingStatus").textContent = `Delete samples failed: ${error.message}`;
}));
$("recipeSelect").addEventListener("change", (event) => loadRecipe(event.target.value));
$("addRectangleTool").addEventListener("click", () => addTool("rectangle"));
$("addEdgeOneTool").addEventListener("click", () => addTool("edge_1"));
$("addEdgeTwoTool").addEventListener("click", () => addTool("edge_2"));
$("addAiTool").addEventListener("click", () => addTool("ai_classifier"));
["calRoiX", "calRoiY", "calRoiW", "calRoiH"].forEach((id) => {
  $(id).addEventListener("input", () => {
    readCalibrationRoiFromUi();
    state.lastCalibrationDetection = null;
    drawOverlay(state.lastResultTools || []);
  });
});

refreshHealth();
updateZoom();
loadRecipes()
  .then(loadReports)
  .then(loadCalibration)
  .then(loadCameraSettings)
  .then(refreshTrainingDependencies)
  .then(refreshTrainingDatasets)
  .then(refreshTrainingSamples)
  .then(refreshTrainingStatus)
  .then(() => startPreviewLoop(true))
  .catch((error) => {
  $("resultBox").textContent = error.message;
});
