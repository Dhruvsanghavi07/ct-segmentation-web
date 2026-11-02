// In-browser inference: load the ONNX U-Net, run it on a held-out CT slice, overlay the result.
"use strict";

const $ = (id) => document.getElementById(id);
const view = $("view"), ctx = view.getContext("2d");
let session, manifest, size = 128, current = null;

function setStatus(msg) { $("status").textContent = msg; }

async function loadImage(src) {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
}

// Image -> Float32Array of grayscale values in [0, 1], resampled to size x size.
function toGray(img) {
  const c = new OffscreenCanvas(size, size), g = c.getContext("2d");
  g.drawImage(img, 0, 0, size, size);
  const px = g.getImageData(0, 0, size, size).data, out = new Float32Array(size * size);
  for (let i = 0; i < out.length; i++) out[i] = (px[4 * i] + px[4 * i + 1] + px[4 * i + 2]) / (3 * 255);
  return out;
}

async function segment(gray) {
  const t0 = performance.now();
  const res = await session.run({ image: new ort.Tensor("float32", gray, [1, 1, size, size]) });
  const ms = performance.now() - t0, logits = res.logits.data, n = size * size, mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = logits[n + i] > logits[i] ? 1 : 0; // argmax over 2 classes
  return { mask, ms };
}

function diceScore(a, b) {
  let inter = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) { inter += a[i] & b[i]; sa += a[i]; sb += b[i]; }
  return sa + sb === 0 ? null : (2 * inter) / (sa + sb);
}

function draw(gray, pred, gt) {
  const base = new OffscreenCanvas(size, size), bg = base.getContext("2d"), im = bg.createImageData(size, size);
  const over = new OffscreenCanvas(size, size), og = over.getContext("2d"), om = og.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = Math.round(gray[i] * 255);
    im.data.set([v, v, v, 255], 4 * i);
    if (pred && $("showPred").checked && pred[i]) om.data.set([230, 57, 70, 120], 4 * i);
    if (gt && $("showGT").checked && gt[i]) {
      const x = i % size, y = (i / size) | 0;
      const edge = x === 0 || y === 0 || x === size - 1 || y === size - 1 ||
        !gt[i - 1] || !gt[i + 1] || !gt[i - size] || !gt[i + size];
      if (edge) om.data.set([46, 196, 182, 255], 4 * i);
    }
  }
  bg.putImageData(im, 0, 0);
  og.putImageData(om, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(base, 0, 0, view.width, view.height);
  ctx.imageSmoothingEnabled = false; // keep mask pixels crisp
  ctx.drawImage(over, 0, 0, view.width, view.height);
}

function redraw() { if (current) draw(current.gray, current.pred, current.gt); }

async function showSample() {
  const c = manifest.cases[$("case").value], s = c.slices[$("slice").value];
  const [ctImg, gtImg] = await Promise.all([loadImage(`samples/${s.ct}`), loadImage(`samples/${s.gt}`)]);
  const gray = toGray(ctImg), gt = Uint8Array.from(toGray(gtImg), (v) => (v > 0.5 ? 1 : 0));
  const { mask, ms } = await segment(gray);
  current = { gray, pred: mask, gt };
  redraw();
  const d = diceScore(mask, gt);
  $("sliceDice").textContent = d === null ? "n/a (no spleen)" : d.toFixed(3);
  $("latency").textContent = `${ms.toFixed(1)} ms`;
  setStatus(`Patient ${c.case}, slice ${Number($("slice").value) + 1} of ${c.slices.length}`);
}

function selectCase() {
  const c = manifest.cases[$("case").value];
  $("slice").max = c.slices.length - 1;
  $("slice").value = Math.floor(c.slices.length / 2);
  return showSample();
}

function kpi(value, label) {
  const d = document.createElement("div");
  d.className = "kpi";
  const b = document.createElement("b"), s = document.createElement("span");
  b.textContent = value;
  s.textContent = label;
  d.append(b, s);
  return d;
}

async function loadMetrics() {
  try {
    const m = await (await fetch("metrics.json")).json();
    $("kpis").replaceChildren(
      kpi(m.test.dice.mean.toFixed(3), `mean 3D Dice, ${m.test.patients} held-out patients`),
      kpi(`${m.test.hd95_mm.median.toFixed(1)} mm`, "median HD95 (boundary error)"),
      kpi(`${m.export.onnx_mb.toFixed(1)} MB`, "model download size"),
      kpi(`${m.train.minutes} min`, "CPU training time in CI"),
    );
  } catch {
    $("kpis").hidden = true;
  }
}

async function main() {
  try {
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    [session, manifest] = await Promise.all([
      ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] }),
      fetch("samples/manifest.json").then((r) => r.json()),
    ]);
    size = manifest.size;
    $("backend").textContent = "WebAssembly (CPU)";
    manifest.cases.forEach((c, i) => $("case").add(new Option(`Patient ${c.case.replace("spleen_", "#")}`, i)));
    $("case").addEventListener("change", selectCase);
    $("slice").addEventListener("input", showSample);
    $("showPred").addEventListener("change", redraw);
    $("showGT").addEventListener("change", redraw);
    $("upload").addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const img = await loadImage(URL.createObjectURL(file));
      const gray = toGray(img), { mask, ms } = await segment(gray);
      current = { gray, pred: mask, gt: null };
      redraw();
      $("sliceDice").textContent = "n/a (no label)";
      $("latency").textContent = `${ms.toFixed(1)} ms`;
      setStatus(`Your image (${file.name}), resized to ${size}×${size}`);
    });
    loadMetrics();
    await selectCase();
  } catch (err) {
    setStatus(`Could not start the demo: ${err.message}`);
    console.error(err);
  }
}

main();
