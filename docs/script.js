const API_BASE_URL = "http://localhost:8000";
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;


const MODEL_LABELS = {
  proba_word_tfidf_logreg: "Word TF-IDF + Logistic Regression",
  proba_char_tfidf_svm: "Char TF-IDF + Linear SVM",
  proba_phobert: "PhoBERT-base",
};

const messageInput = document.getElementById("message-input");
const checkBtn = document.getElementById("check-btn");
const errorBox = document.getElementById("error-box");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const modelResults = document.getElementById("model-results");
const overallBar = document.getElementById("overall-bar");
const overallPercent = document.getElementById("overall-percent");
const overallLabel = document.getElementById("overall-label");

checkBtn.addEventListener("click", handleCheck);
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    handleCheck();
  }
});

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

    if (!response.ok) {
      throw new Error(`Server trả về lỗi ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
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

  Object.keys(data)
    .filter((key) => key.startsWith("proba_"))
    .forEach((key) => {
      const percent = data[key] * 100;
      const label = MODEL_LABELS[key] || key;
      modelResults.appendChild(buildModelRow(label, percent));
    });

  const overall = data.confidence_scam * 100;
  overallBar.style.width = `${overall}%`;
  overallBar.style.background = colorForPercent(overall);
  overallPercent.textContent = `${overall.toFixed(1)}%`;
  overallPercent.style.color = colorForPercent(overall);

  overallLabel.textContent = data.label === "scam" ? "Nghi ngờ scam" : "An toàn";
  overallLabel.className = `label-badge ${data.label === "scam" ? "scam" : "ham"}`;

  show(results);
}

function buildModelRow(label, percent) {
  const row = document.createElement("div");
  row.className = "model-row";

  const top = document.createElement("div");
  top.className = "model-row-top";

  const name = document.createElement("span");
  name.className = "model-name";
  name.textContent = label;

  const value = document.createElement("span");
  value.className = "model-percent";
  value.textContent = `${percent.toFixed(1)}%`;
  value.style.color = colorForPercent(percent);

  top.appendChild(name);
  top.appendChild(value);

  const barWrapper = document.createElement("div");
  barWrapper.className = "bar-wrapper";

  const barFill = document.createElement("div");
  barFill.className = "bar-fill";
  barFill.style.width = `${percent}%`;
  barFill.style.background = colorForPercent(percent);

  barWrapper.appendChild(barFill);

  row.appendChild(top);
  row.appendChild(barWrapper);
  return row;
}

function colorForPercent(percent) {
  if (percent >= 90) return "#f87171"; // đỏ — khả năng scam cao
  if (percent >= 83) return "#fbbf24"; // vàng — chưa chắc chắn
  return "#4ade80"; // xanh — khả năng an toàn
}

function setLoading(isLoading) {
  checkBtn.disabled = isLoading;
  isLoading ? show(loading) : hide(loading);
}

function showError(msg) {
  errorBox.textContent = msg;
  show(errorBox);
}

function show(el) {
  el.classList.remove("hidden");
}

function hide(el) {
  el.classList.add("hidden");
}
