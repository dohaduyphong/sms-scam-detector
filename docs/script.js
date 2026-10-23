// API_BASE_URL phụ thuộc vào nơi trang được mở:
// - Bản chính thức (GitHub Pages): https://api.duyphong.info
// - Chạy thử trên máy (docs/ mở qua localhost hoặc 127.0.0.1, ví dụ `python3 -m http.server 5500`):
//   backend chạy ở cổng 8000 trên cùng máy (`uvicorn api:app --port 8000`).
const PRODUCTION_API_BASE_URL = "https://api.duyphong.info";
const LOCAL_API_PORT = 8000;
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const API_BASE_URL = IS_LOCAL
  ? `http://${window.location.hostname}:${LOCAL_API_PORT}`
  : PRODUCTION_API_BASE_URL;
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

// Bản sao để hiển thị của scam_detector/config.py; sửa config thì phải sửa cả ở đây.
// THRESHOLD = sigmoid(ENSEMBLE_THRESHOLD).
const THRESHOLD = 0.635;
const MODELS = [
  {
    key: "proba_word_tfidf_logreg",
    name: "Word TF-IDF + Logistic Regression",
    desc: "Nhận ra từ và cụm từ hay gặp trong tin lừa đảo.",
  },
  {
    key: "proba_char_tfidf_svm",
    name: "Char TF-IDF + Linear SVM",
    desc: "Đọc theo từng mảnh ký tự, vẫn nhận ra khi chữ bị viết sai hoặc biến tấu.",
  },
  {
    key: "proba_visobert",
    name: "ViSoBERT",
    desc: "Hiểu cách viết trên mạng xã hội, kể cả tin không dấu và viết tắt.",
  },
  {
    key: "proba_phobert",
    name: "PhoBERT",
    desc: "Hiểu ngữ cảnh của cả câu tiếng Việt.",
  },
];

// Mức theo phần trăm: từ HIGH_RISK trở lên là cao (đỏ), từ ngưỡng trở lên là vừa (vàng), còn lại là thấp (xanh).
// HIGH_RISK phải lớn hơn ngưỡng (63.5%): mức vừa là tin vừa qua ngưỡng, chưa chắc chắn.
const HIGH_RISK = 80;
const MID_RISK = THRESHOLD * 100;

// Tin mẫu cho nút "Thử với tin mẫu", gồm cả tin scam và tin hợp lệ. Số điện thoại, số tài khoản và link đều là giả.
const SAMPLES = [
  // Tin scam
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

  // Tin hợp lệ
  "Ma OTP cua ban la 482913. Ma co hieu luc trong 5 phut. Tuyet doi KHONG chia se ma nay voi bat ky ai, ke ca nhan vien ngan hang.",
  "c ơi e vừa ck 2tr5 tiền hàng tháng này vào tk vietcombank của c r ạ, c check giúp e vs nha",
  "Đơn hàng #SPX928471 của bạn đang được giao. Shipper sẽ gọi trước khi đến, vui lòng giữ điện thoại.",
  "Mẹ ơi con về muộn chút, đừng chờ cơm con nha. Con ghé siêu thị mua ít đồ rồi về luôn.",
];
let lastSample = -1;

function randomSample() {
  // Không lặp lại tin vừa hiện
  let i;
  do {
    i = Math.floor(Math.random() * SAMPLES.length);
  } while (SAMPLES.length > 1 && i === lastSample);
  lastSample = i;
  return SAMPLES[i];
}

// Hàm tiện ích
const $ = (id) => document.getElementById(id);
const show = (node) => node.classList.remove("hidden");
const hide = (node) => node.classList.add("hidden");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function levelForPercent(percent) {
  if (percent >= HIGH_RISK) return "high";
  if (percent >= MID_RISK) return "mid";
  return "low";
}

const pct = (x) => `${Math.round(x * 100)}%`;
const pct1 = (percent) => `${percent.toLocaleString("vi-VN", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;

// Phần tử trên trang
const messageInput = $("message-input");
const checkBtn = $("check-btn");
const errorBox = $("error-box");
const results = $("results");
const verdict = $("verdict");
const modelResults = $("model-results");

$("rules-note").textContent =
  `Kết luận cuối cùng do một mô hình kết hợp điểm của cả ${MODELS.length} mô hình trên; ` +
  `từ ${pct(THRESHOLD)} trở lên được coi là có dấu hiệu lừa đảo.`;

// Sự kiện
checkBtn.addEventListener("click", handleCheck);
$("sample-btn").addEventListener("click", () => {
  messageInput.value = randomSample();
  hide(errorBox);
  messageInput.focus();
});
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleCheck();
});

// Luồng chính
async function handleCheck() {
  const message = messageInput.value.trim();

  hide(errorBox);
  hide(results);

  if (!message) {
    showError("Hãy dán nội dung tin nhắn vào ô trên trước khi kiểm tra.");
    messageInput.focus();
    return;
  }

  setLoading(true);
  try {
    let response;
    try {
      response = await fetch(PREDICT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
    } catch {
      showError("Không kết nối được tới máy chủ kiểm tra. Hãy kiểm tra mạng rồi thử lại sau ít phút.");
      return;
    }
    if (!response.ok) {
      showError(`Máy chủ kiểm tra đang gặp lỗi (mã ${response.status}). Hãy thử lại sau ít phút.`);
      return;
    }
    renderResults(await response.json());
  } finally {
    setLoading(false);
  }
}

function renderResults(data) {
  const overall = data.confidence_scam * 100;
  const level = data.label === "scam" ? levelForPercent(overall) : "low";

  verdict.className = `verdict ${level}`;
  $("verdict-title").textContent = {
    high: "Nhiều khả năng là tin lừa đảo",
    mid: "Có dấu hiệu lừa đảo",
    low: "Chưa thấy dấu hiệu lừa đảo",
  }[level];
  $("verdict-score").textContent = `Mức nghi ngờ: ${pct1(overall)}`;
  $("verdict-advice").textContent =
    level === "low"
      ? "Kết quả chỉ để tham khảo. Nếu tin yêu cầu chuyển tiền hoặc cung cấp mã OTP, hãy xác minh lại với người gửi qua kênh khác."
      : "Đừng bấm vào link, gọi vào số trong tin, chuyển tiền hay cung cấp mã OTP. Hãy xác minh qua kênh chính thức của tổ chức được nhắc tới.";

  // Mỗi khóa "proba_*" trong phản hồi là một dòng, nên model mới ở backend tự hiện ra
  // (thêm vào MODELS nếu muốn có tên và mô tả dễ đọc).
  modelResults.innerHTML = "";
  Object.keys(data)
    .filter((key) => key.startsWith("proba_"))
    .forEach((key) => {
      const model = MODELS.find((m) => m.key === key);
      modelResults.appendChild(buildRow(model ? model.name : key, model ? model.desc : "", data[key] * 100));
    });

  show(results);
  results.scrollIntoView({ block: "start" });
}

function buildRow(label, desc, percent) {
  const level = levelForPercent(percent);
  const row = el("li", "model-row");

  const top = el("div", "row-top");
  top.append(el("span", "row-name", label), el("span", `row-value ${level}`, pct1(percent)));
  row.appendChild(top);
  if (desc) row.appendChild(el("p", "row-desc", desc));

  const bar = el("div", "bar");
  const fill = el("div", `bar-fill ${level}`);
  fill.style.width = `${percent}%`;
  bar.appendChild(fill);
  row.appendChild(bar);
  return row;
}

function setLoading(isLoading) {
  checkBtn.disabled = isLoading;
  checkBtn.textContent = isLoading ? "Đang kiểm tra…" : "Kiểm tra tin nhắn";
}

function showError(msg) {
  errorBox.textContent = msg;
  show(errorBox);
}
