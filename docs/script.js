// ============================================================
// Config
// API_BASE_URL is chosen from where this page is served:
// - Production (GitHub Pages): https://api.duyphong.info
// - Local dev (docs/ on localhost / 127.0.0.1, e.g. `python3 -m http.server 5500`):
//   the backend on port 8000 of the same host (`uvicorn api:app --port 8000`).
// ============================================================
const PRODUCTION_API_BASE_URL = "https://api.duyphong.info";
const LOCAL_API_PORT = 8000;
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const API_BASE_URL = IS_LOCAL
  ? `http://${window.location.hostname}:${LOCAL_API_PORT}`
  : PRODUCTION_API_BASE_URL;
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

// Display-only mirror of scam_detector/config.py -- keep in sync.
// THRESHOLD = sigmoid(ENSEMBLE_THRESHOLD); weight = phần của |coef| trong ENSEMBLE_COEF.
const THRESHOLD = 0.635;
const MODELS = [
  {
    key: "proba_word_tfidf_logreg",
    name: "Word TF-IDF + Logistic Regression",
    weight: 0.54,
    desc: "Đặc trưng theo từ và cụm 2 từ.",
  },
  {
    key: "proba_char_tfidf_svm",
    name: "Char TF-IDF + Linear SVM",
    weight: 0.08,
    desc: "N-gram ký tự 3–5, bền với lỗi chính tả và cách viết biến thể.",
  },
  {
    key: "proba_visobert",
    name: "ViSoBERT",
    weight: 0.07,
    desc: "Mô hình ngôn ngữ pretrain trên văn bản mạng xã hội, bền với tin không dấu và teencode.",
  },
  {
    key: "proba_phobert",
    name: "PhoBERT",
    weight: 0.31,
    desc: "Mô hình ngôn ngữ tiếng Việt chuẩn (văn bản đã tách từ), hiểu ngữ cảnh của cả câu.",
  },
];

// Risk colours for a percentage: >= HIGH red, >= MID amber, else green.
// HIGH_RISK phải lớn hơn ngưỡng (63.5%): vùng vàng = vừa qua ngưỡng, chưa chắc chắn.
const HIGH_RISK = 80;
const MID_RISK = THRESHOLD*100;

// Tin mẫu cho nút "Dùng tin mẫu": trộn cả scam và hợp lệ. Số điện thoại / STK / link là giả.
const SAMPLES = [
  // scam
  "Ngan hang MB BANK thong bao: Tai khoan cua ban vua phat sinh giao dich 8.500.000d. Neu khong phai ban, vui long truy cap rnbbank.support.com de kiem tra va huy giao dich.",
  "Cuc CSGT: Phuong tien 51G-847.29 vi pham loi vuot den do qua camera. Nop phat 2.500.000d vao STK 1903847291 Kho Bac truoc 17h ngay mai de tranh bi tam giu phuong tien.",
  "Tong cuc Thue: Ban du dieu kien nhan lai 3.250.000d thue TNCN dong thua. Vui long xac nhan thong tin tai thue-hoantien.com trong 24h.",
  "TUYỂN CTV chốt đơn TikTok Shop tại nhà, lương 300-500k/ngày, không cần cọc. Kết bạn Telegram 0912345678 để được hướng dẫn.",
  "Chúc mừng quý khách đã trúng thưởng iPhone 15 Pro Max từ chương trình tri ân. Truy cập qua-tang-tri-an.vn và đóng phí hồ sơ 350.000đ để nhận quà.",
  "[MB Bank] Nhân viên CSKH sẽ gọi xác minh trong ít phút. Vui lòng đọc mã OTP 6 số vừa gửi để hoàn tất nâng cấp bảo mật Smart OTP.",
  "Chúng tôi đã phát hiện giao dịch bất thường trong tài khoản ngân hàng ACB của bạn, vui lòng truy cập acbbbank.com để xác minh.",
  "CONG AN TP HA NOI THONG BAO: Anh/Chi dang la doi tuong tinh nghi cho mot vu an rua tien nghiem trong. Yeu cau anh/chi hop tac, cung cap thong tin ca nhan cho co quan chuc nang qua duong link b0c0ngan.com",
  "Anh oi em la Hung day, em moi hong dth, anh chuyen giup em 2tr vao stk nay em sua dth voi, ti em tra sau.",
  "Ban oi minh la nv cham soc kh cua dien luc, thang nay nha ban dc ho tro giam 50% tien dien nhung phai dong phi kich hoat 150k trc. Ck vao stk 0912312312 ten Tran Thi Hoai roi gui bien lai cho minh nhe, muon thi lam nhanh ko het slot",
  "CHUC MUNG! Ban da trung thuong 50 trieu VND tu chuong trinh khach hang may man. Goi ngay 0912341234 de nhan thuong.",
  "Chào anh, em bên bộ phận đối soát. Bên em thấy tài khoản của anh có một khoản hoàn tiền chưa nhận. Anh cho em xin thời gian thuận tiện để em gọi xác minh.",

  // hợp lệ
  "Ma OTP cua ban la 482913. Ma co hieu luc trong 5 phut. Tuyet doi KHONG chia se ma nay voi bat ky ai, ke ca nhan vien ngan hang.",
  "c ơi e vừa ck 2tr5 tiền hàng tháng này vào tk vietcombank của c r ạ, c check giúp e vs nha",
  "Đơn hàng #SPX928471 của bạn đang được giao. Shipper sẽ gọi trước khi đến, vui lòng giữ điện thoại.",
  "Mẹ ơi con về muộn chút, đừng chờ cơm con nha. Con ghé siêu thị mua ít đồ rồi về luôn.",
];
let lastSample = -1;

function randomSample() {
  // không lặp lại tin vừa hiện
  let i;
  do {
    i = Math.floor(Math.random() * SAMPLES.length);
  } while (SAMPLES.length > 1 && i === lastSample);
  lastSample = i;
  return SAMPLES[i];
}

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

// ---------- static content ----------
$("meta").textContent = `MODELS ${MODELS.length} · THRESHOLD ${pct(THRESHOLD)}`;

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
  `Điểm tổng hợp do một mô hình logistic regression (stacking) tính từ điểm chuẩn hoá của ${MODELS.length} mô hình; ` +
  `từ ${pct(THRESHOLD)} trở lên là scam.`;

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
  messageInput.value = randomSample();
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
  overallBar.style.background = colorForPercent(overall);
  overallPercent.textContent = `${overall.toFixed(1)}%`;
  overallPercent.style.color = colorForPercent(overall);

  const isScam = data.label === "scam";
  overallLabel.textContent = isScam ? "Nghi ngờ scam" : "An toàn";
  overallLabel.className = `label-badge ${isScam ? "scam" : "ham"}`;

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