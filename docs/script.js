// ============================================================
// Config
// - Local dev: keep "http://localhost:8000" (run api.py locally).
// - Production (GitHub Pages): use your deployed backend's HTTPS URL.
//   Must be HTTPS -- browsers block plain-HTTP fetches from HTTPS pages.
// ============================================================
const API_BASE_URL = "https://simpson-realized-royalty-texas.trycloudflare.com";
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

// Display-only mirror of scam_detector/config.py -- keep in sync.
const THRESHOLD = 0.4;
const OVERRIDE = 0.95;
const MODELS = [
  {
    key: "proba_word_tfidf_logreg",
    name: "Word TF-IDF + Logistic Regression",
    weight: 0.04,
    desc: "Đặc trưng theo từ và cụm 2 từ. Không đóng góp nhiều vào điểm tổng hợp, nhưng vẫn tham gia luật override.",
  },
  {
    key: "proba_char_tfidf_svm",
    name: "Char TF-IDF + Linear SVM",
    weight: 0.62,
    desc: "N-gram ký tự 3–5, bền với lỗi chính tả và cách viết biến thể.",
  },
  {
    key: "proba_phobert",
    name: "PhoBERT-base",
    weight: 0.34,
    desc: "Mô hình ngôn ngữ tiếng Việt được fine-tune, hiểu ngữ cảnh của cả câu.",
  },
];

// Risk colours for a percentage: >= HIGH red, >= MID amber, else green.
const HIGH_RISK = 70;
const MID_RISK = 40;

const SAMPLE = "Chúng tôi đã phát hiện giao dịch bất thường trong tài khoản ngân hàng ACB của bạn, vui lòng truy cập acbbbank.com để xác minh.";

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const show = (node) => node.classList.remove("hidden");
const hide = (node) => node.classList.add("hidden");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function colorForPercent(percent) {
  if (percent >= HIGH_RISK) return "#f87171"; // red -- likely scam
  if (percent >= MID_RISK) return "#fbbf24"; // amber -- uncertain
  return "#4ade80"; // green -- likely safe
}

const pct = (x) => `${Math.round(x * 100)}%`;

// ---------- elements ----------
const messageInput = $("message-input");
const checkBtn = $("check-btn");
const errorBox = $("error-box");
const loading = $("loading");
const results = $("results");
const modelResults = $("model-results");
const overallBar = $("overall-bar");
const overallPercent = $("overall-percent");
const overallLabel = $("overall-label");
const overrideNote = $("override-note");

// ---------- static content ----------
$("meta").textContent = `MODELS ${MODELS.length} · THRESHOLD ${pct(THRESHOLD)} · OVERRIDE ${pct(OVERRIDE)}`;

MODELS.forEach((m, i) => {
  const card = el("article", "card model-card");
  card.append(
    el("p", "num", String(i + 1).padStart(2, "0")),
    el("h3", "", m.name),
    el("p", "desc", m.desc),
    el("p", "weight", `TRỌNG SỐ ${pct(m.weight)}`)
  );
  $("model-cards").appendChild(card);
});

$("rules-note").textContent =
  `Điểm tổng hợp là trung bình có trọng số của các xác suất; từ ${pct(THRESHOLD)} trở lên là scam. ` +
  `Luật override: nếu bất kỳ mô hình nào đạt từ ${pct(OVERRIDE)}, tin nhắn được gắn nhãn scam bất kể điểm trung bình.`;

// ---------- events ----------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", String(t === tab));
    });
    document.querySelectorAll(".panel").forEach((p) => {
      p.classList.toggle("hidden", p.id !== `panel-${tab.dataset.tab}`);
    });
  });
});

checkBtn.addEventListener("click", handleCheck);
$("sample-btn").addEventListener("click", () => {
  messageInput.value = SAMPLE;
  messageInput.focus();
});
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleCheck();
});

// ---------- main flow ----------
async function handleCheck() {
  const message = messageInput.value.trim();

  hide(errorBox);
  hide(results);

  if (!message) {
    showError("Vui lòng nhập nội dung tin nhắn.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch(PREDICT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!response.ok) throw new Error(`Server trả về lỗi ${response.status}`);
    renderResults(await response.json());
  } catch (err) {
    showError(
      `Không thể kết nối tới backend (${err.message}). ` +
        `Kiểm tra API_BASE_URL trong script.js và đảm bảo server đang chạy.`
    );
  } finally {
    setLoading(false);
  }
}

function renderResults(data) {
  modelResults.innerHTML = "";

  // One row per "proba_*" key in the response, so a new model on the
  // backend shows up automatically (add it to MODELS for a nice name).
  Object.keys(data)
    .filter((key) => key.startsWith("proba_"))
    .forEach((key) => {
      const model = MODELS.find((m) => m.key === key);
      modelResults.appendChild(
        buildRow(model ? model.name : key, data[key] * 100, model ? model.weight : undefined)
      );
    });

  const overall = data.confidence_scam * 100;
  overallBar.style.width = `${overall}%`;
  if(!data.override_triggered) overallBar.style.background = colorForPercent(overall);
  else overallBar.style.background = "#f87171";
  overallPercent.textContent = `${overall.toFixed(1)}%`;
  if(!data.override_triggered) overallPercent.style.color = colorForPercent(overall);
  else overallPercent.style.color = "#f87171";


  const isScam = data.label === "scam";
  overallLabel.textContent = isScam ? "Nghi ngờ scam" : "An toàn";
  overallLabel.className = `label-badge ${isScam ? "scam" : "ham"}`;

  // Backend sets this when one model was >= OVERRIDE and forced "scam"
  // even though the weighted average alone would have said otherwise.
  if (data.override_triggered) {
    overrideNote.textContent =
      "⚠️ Một mô hình rất tự tin đây là scam nên hệ thống ưu tiên cảnh báo, dù điểm trung bình có trọng số thấp hơn ngưỡng.";
    show(overrideNote);
  } else {
    hide(overrideNote);
  }

  show(results);
}

function buildRow(label, percent, weight) {
  const row = el("div", "model-row");
  const top = el("div", "row-top");

  const name = el("span", "row-name", label);
  if (typeof weight === "number") name.appendChild(el("span", "tag", `${pct(weight)} trọng số`));

  const value = el("span", "row-value", `${percent.toFixed(1)}%`);
  value.style.color = colorForPercent(percent);
  top.append(name, value);

  const bar = el("div", "bar");
  const fill = el("div", "bar-fill");
  fill.style.width = `${percent}%`;
  fill.style.background = colorForPercent(percent);
  bar.appendChild(fill);

  row.append(top, bar);
  return row;
}

function setLoading(isLoading) {
  checkBtn.disabled = isLoading;
  if (isLoading) show(loading);
  else hide(loading);
}

function showError(msg) {
  errorBox.textContent = msg;
  show(errorBox);
}