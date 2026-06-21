// ============================================================
// Cat vs Dog — The Ultimate Debate | Frontend Logic
// ============================================================

const uploadZone   = document.getElementById("upload-zone");
const uploadContent= document.getElementById("upload-content");
const previewWrap  = document.getElementById("preview-wrapper");
const previewImg   = document.getElementById("preview-img");
const fileInput    = document.getElementById("file-input");
const classifyBtn  = document.getElementById("classify-btn");
const changeBtn    = document.getElementById("change-btn");
const resultCard   = document.getElementById("result-card");
const resultLabel  = document.getElementById("result-label");
const confBar      = document.getElementById("confidence-bar");
const confValue    = document.getElementById("confidence-value");
const tryAgainBtn  = document.getElementById("try-again-btn");
const memeBox      = document.getElementById("meme-box");
const memeLine     = document.getElementById("meme-line");
const cinemaImg    = document.getElementById("cinema-img");
const dogFrame     = document.getElementById("dog-frame");
const catFrame     = document.getElementById("cat-frame");
const mainHeader   = document.getElementById("main-header");
const stage        = document.getElementById("stage");
const soundToggle  = document.getElementById("sound-toggle");

let selectedFile = null;
let memeInterval = null;
let duckResumeBound = false;
let autoplayToastShown = false;
let isMuted = localStorage.getItem("catdog_sound_muted") === "1";

// ── Audio setup ─────────────────────────────────────────────
const duckAudio = new Audio("/media/sounds/Fluffing-a-Duck.mp3");
duckAudio.loop = true;
duckAudio.preload = "auto";

const cheerAudio = new Audio("/media/sounds/dragon-studio-cheering-crowd.mp3");
cheerAudio.preload = "auto";

duckAudio.muted = isMuted;
cheerAudio.muted = isMuted;
updateSoundButton();

soundToggle.addEventListener("click", () => {
  isMuted = !isMuted;
  localStorage.setItem("catdog_sound_muted", isMuted ? "1" : "0");

  duckAudio.muted = isMuted;
  cheerAudio.muted = isMuted;

  if (isMuted) {
    duckAudio.pause();
    duckAudio.currentTime = 0;
    cheerAudio.pause();
    cheerAudio.currentTime = 0;
  } else {
    startAmbientAudio();
  }

  updateSoundButton();
});

// Try to start ambient duck audio as soon as UI loads.
initializeAmbientAudio();

// ── Meme texts (cycled during loading) ──────────────────────
const memeTexts = [
  "one eternity later...",
  "teaching the AI what fur is... 🐾",
  "consulting 22,400 animal witnesses...",
  "calculating fluffiness coefficient...",
  "the AI is having a snack break 🍕",
  "asking very nicely... please hold 🙏",
  "measuring the ear-to-fluff ratio...",
  "checking the whisker database 📊",
  "this is taking longer than expected...",
  "running advanced paw analysis...",
];

// ── Cinema image paths ───────────────────────────────────────
const cinemaImages = {
  Cat: "/media/absolute%20cinima%20cat.png",
  Dog: "/media/absolute%20cinima%20dog.png",
};

// ── Drag & Drop ──────────────────────────────────────────────
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

// ── Click zone ───────────────────────────────────────────────
uploadZone.addEventListener("click", (e) => {
  if (e.target.closest(".change-btn")) return;
  if (e.target.closest("label[for='file-input']")) return;
  if (previewWrap.style.display !== "none") return;
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// ── Handle selected file ─────────────────────────────────────
function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showToast("Please upload an image file.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadContent.style.display = "none";
    previewWrap.style.display   = "flex";
    classifyBtn.disabled = false;
    resultCard.style.display = "none";
  };
  reader.readAsDataURL(file);
}

// ── Remove image ─────────────────────────────────────────────
changeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  resetAll();
});

function resetAll() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.style.display   = "none";
  uploadContent.style.display = "flex";
  classifyBtn.disabled = true;
  resultCard.style.display = "none";
  resultCard.classList.remove("fullscreen");
  memeBox.style.display = "none";
  stopMeme();
  stopPredictionAudio();
  cheerAudio.pause();
  cheerAudio.currentTime = 0;
  startAmbientAudio();
  wakeAnimals();

  // Restore top section
  [mainHeader, stage].forEach(el => {
    el.classList.remove("hidden-collapsed");
  });
}

// ── Classify ─────────────────────────────────────────────────
classifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  classifyBtn.disabled = true;
  classifyBtn.classList.add("loading");
  resultCard.style.display = "none";

  // Morph animals to sleepy + show meme
  sleepAnimals();
  startMeme();
  startAmbientAudio();

  const formData = new FormData();
  formData.append("file", selectedFile);

  // Wait for BOTH API response AND minimum 4s delay
  const [result] = await Promise.all([
    fetch("/predict", { method: "POST", body: formData })
      .then(r => r.json())
      .catch(() => ({ error: "Connection error. Is the server running?" })),
    new Promise(resolve => setTimeout(resolve, 4000))
  ]);

  stopMeme();
  memeBox.style.display = "none";
  classifyBtn.classList.remove("loading");
  classifyBtn.disabled = false;

  if (result.error) {
    showToast(result.error);
    wakeAnimals();
    return;
  }

  showResult(result);
});

// ── Show result ───────────────────────────────────────────────
function showResult({ label, confidence }) {
  const isCat = label === "Cat";

  // Instantly collapse the header + stage (no delay — the cinema reveal is the drama)
  mainHeader.classList.add("hidden-collapsed");
  stage.classList.add("hidden-collapsed");

  // Set cinema image
  cinemaImg.src = cinemaImages[label];

  // Result text
  resultLabel.textContent = isCat ? "🐱  It's a Cat!" : "🐶  It's a Dog!";
  resultLabel.className   = "result-verdict " + label.toLowerCase();

  // Confidence bar
  confBar.className   = "conf-bar " + label.toLowerCase();
  confBar.style.width = "0%";
  confValue.textContent = confidence + "%";

  // Show result card
  resultCard.style.display = "flex";
  resultCard.classList.add("fullscreen");
  stopPredictionAudio();
  playResultAudio();
  triggerCornerConfetti();

  window.scrollTo({ top: 0, behavior: "instant" });

  requestAnimationFrame(() => {
    setTimeout(() => { confBar.style.width = confidence + "%"; }, 150);
  });
}

// ── Try again ─────────────────────────────────────────────────
tryAgainBtn.addEventListener("click", () => {
  resetAll();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// ── Animal crossfade helpers ──────────────────────────────────
function sleepAnimals() {
  dogFrame && dogFrame.classList.add("sleeping");
  catFrame && catFrame.classList.add("sleeping");
}
function wakeAnimals() {
  dogFrame && dogFrame.classList.remove("sleeping");
  catFrame && catFrame.classList.remove("sleeping");
}

// ── Meme text rotation ────────────────────────────────────────
function startMeme() {
  memeBox.style.display = "flex";
  let idx = 0;
  memeLine.textContent = memeTexts[0];
  memeLine.style.opacity = "1";
  memeInterval = setInterval(() => {
    memeLine.style.opacity = "0";
    setTimeout(() => {
      idx = (idx + 1) % memeTexts.length;
      memeLine.textContent = memeTexts[idx];
      memeLine.style.opacity = "1";
    }, 300);
  }, 1400);
}
function stopMeme() {
  clearInterval(memeInterval);
  memeInterval = null;
}

function initializeAmbientAudio() {
  // Some browsers block autoplay with sound. If blocked, arm a one-time user-gesture retry.
  startAmbientAudio();
  bindDuckResumeOnInteraction();
}

function bindDuckResumeOnInteraction() {
  if (isMuted) return;
  if (duckResumeBound) return;
  duckResumeBound = true;

  const tryResume = () => {
    startAmbientAudio();
    if (!duckAudio.paused) {
      window.removeEventListener("pointerdown", tryResume);
      window.removeEventListener("keydown", tryResume);
      duckResumeBound = false;
    }
  };

  // Keep listeners active until playback actually starts.
  window.addEventListener("pointerdown", tryResume);
  window.addEventListener("keydown", tryResume);
}

function startAmbientAudio() {
  if (isMuted) return;
  // Stop cheering if still active and keep the duck loop running under UI flow.
  cheerAudio.pause();
  cheerAudio.currentTime = 0;

  if (!duckAudio.paused) return;
  duckAudio.play().catch(() => {
    if (!autoplayToastShown) {
      autoplayToastShown = true;
      showToast("Browser blocked autoplay. Click anywhere once to enable sound.");
    }
    bindDuckResumeOnInteraction();
  });
}

function stopPredictionAudio() {
  duckAudio.pause();
  duckAudio.currentTime = 0;
}

function playResultAudio() {
  if (isMuted) return;
  cheerAudio.currentTime = 0;
  cheerAudio.play().catch(() => {
    // Ignore autoplay-policy rejections.
  });
}

function updateSoundButton() {
  soundToggle.classList.toggle("muted", isMuted);
  soundToggle.textContent = isMuted ? "🔇 Sound Off" : "🔊 Sound On";
  soundToggle.setAttribute("aria-label", isMuted ? "Unmute sound" : "Mute sound");
  soundToggle.setAttribute("title", isMuted ? "Unmute sound" : "Mute sound");
}

function triggerCornerConfetti() {
  const colors = [
    "#ff4d6d", "#ff7a00", "#ffbe0b", "#8ac926", "#2ec4b6", "#00bbf9",
    "#3a86ff", "#8338ec", "#ff66c4", "#f15bb5", "#06d6a0", "#ffd166"
  ];
  const piecesPerSide = 170;

  for (let i = 0; i < piecesPerSide; i += 1) {
    spawnConfettiPiece("left", colors, i * 8);
    spawnConfettiPiece("right", colors, i * 8);
  }
}

function spawnConfettiPiece(side, colors, delayMs) {
  const piece = document.createElement("span");
  piece.className = "confetti-piece";

  const offset = Math.floor(Math.random() * 280);
  const drift = Math.floor(140 + Math.random() * 460);
  const rotate = Math.floor(240 + Math.random() * 420);
  const size = (3 + Math.random() * 5.5).toFixed(1);
  const duration = Math.floor(2000 + Math.random() * 1300);
  const rise = (74 + Math.random() * 26).toFixed(1);

  piece.style.width = `${size}px`;
  piece.style.height = `${(Number(size) * (1.1 + Math.random() * 1.3)).toFixed(1)}px`;
  piece.style.background = colors[Math.floor(Math.random() * colors.length)];
  piece.style.opacity = (0.7 + Math.random() * 0.3).toFixed(2);
  piece.style.animationDuration = `${duration}ms`;
  piece.style.animationDelay = `${delayMs}ms`;
  piece.style.setProperty("--confetti-x", `${side === "left" ? drift : -drift}px`);
  piece.style.setProperty("--confetti-y", `-${rise}vh`);
  piece.style.setProperty("--confetti-r", `${side === "left" ? rotate : -rotate}deg`);

  if (side === "left") {
    piece.style.left = `${6 + offset}px`;
  } else {
    piece.style.right = `${6 + offset}px`;
  }

  piece.addEventListener("animationend", () => {
    piece.remove();
  });

  document.body.appendChild(piece);
}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg) {
  const old = document.querySelector(".toast");
  if (old) old.remove();
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  t.style.cssText = `
    position:fixed;bottom:2rem;left:50%;transform:translateX(-50%);
    background:#ef4444;color:#fff;padding:0.75rem 1.5rem;
    border-radius:50px;font-size:0.88rem;font-weight:500;
    box-shadow:0 8px 30px rgba(239,68,68,0.3);z-index:999;
  `;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
