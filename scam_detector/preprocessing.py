"""
Tiền xử lý tin nhắn tiếng Việt.
"""
import re
import string
import threading
import unicodedata

# Các token PII có sẵn trong bộ dữ liệu
PII_TOKENS = ["NAME", "PHONE", "BANK_ACC", "MONEY", "NUMBER", "TIME", "DATE", "LINK"]

# Che token PII để lower() và bỏ dấu câu không làm vỡ "[MONEY]"
_PII_PLACEHOLDER = {
    tok: f'xxpii{tok.lower().replace("_", "")}xx' for tok in PII_TOKENS
}
_PII_RESTORE = {v: f"[{k}]" for k, v in _PII_PLACEHOLDER.items()}

_PII_PATTERN = re.compile(
    r"\[(" + "|".join(PII_TOKENS) + r")\]", flags=re.IGNORECASE
)

# Giống mẫu trong Model_Train_LR_SVM.ipynb
_URL_PATTERN = re.compile(
    r"(https?:\/\/\s*\S+)"
    r"|(www\.\S+)"
    r"|(\b[a-zA-Z0-9][a-zA-Z0-9\-]*\.(?:com|vn|net|org|co|io|me|shop|cc|info|biz|shop|top)(?:\.[a-zA-Z]{2,3})?(?:\/\S*)?)",
    flags=re.IGNORECASE,
)

# Giống mẫu trong Model_Train_ViSoBERT_Final.ipynb, có thêm đuôi .ly
_URL_PATTERN_VISOBERT = re.compile(
    r"(https?:\/\/\s*\S+)"
    r"|(www\.\S+)"
    r"|(\b[a-zA-Z0-9][a-zA-Z0-9\-]*\.(?:com|vn|net|org|co|io|me|shop|cc|info|biz|shop|top|ly)(?:\.[a-zA-Z]{2,3})?(?:\/\S*)?)",
    flags=re.IGNORECASE,
)

# Tiền xử lý cho hai model TF-IDF
def preprocess_text(text: str) -> str:
    text = str(text)

    text = _URL_PATTERN.sub(" [LINK] ", text)

    def _protect(match: "re.Match[str]") -> str:
        token = match.group(1).upper()
        return f" {_PII_PLACEHOLDER[token]} "


    text = _PII_PATTERN.sub(_protect, text)

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    for placeholder, restored in _PII_RESTORE.items():
        text = text.replace(placeholder, restored)
    return text


# Chuẩn hoá thông tin định danh cá nhân (PII) trong tin nhắn người dùng nhập vào

# 1. Số tài khoản ngân hàng: dựa vào các từ đứng gần để phân biệt với số điện thoại
_VN_BANK_NAMES = (
    r"\b(?:vietcombank|vcb|vietinbank|viettinbank|ctg|bidv|agribank|"
    r"techcombank|tcb|mb\s?bank|mb|vpbank|vp\s?bank|acb|"
    r"sacombank|stb|hdbank|hd\s?bank|tpbank|tp\s?bank|shb|vib|ocb|"
    r"seabank|sea\s?bank|msb|maritime\s?bank|eximbank|eib|"
    r"lienvietpostbank|lienviet|lpb|nam\s?a\s?bank|bac\s?a\s?bank|"
    r"pvcombank|scb|abbank|ab\s?bank|vietabank|viet\s?a\s?bank|"
    r"kienlongbank|kien\s?long\s?bank|ncb|pgbank|pg\s?bank|vietbank|"
    r"baoviet\s?bank|bvbank|cbbank|cb\s?bank)\b"
)
_BANK_KEYWORDS = r"số\s*tài\s*khoản|stk|tk|tài\s*khoản"
 
_BANK_ACC_PATTERN = re.compile(
    r"(?:"
        # Nhánh 1: "stk vcb 0123..." / "vcb stk 0123..." / "stk 0123..."
        r"(?P<kw>"
            r"(?:" + _BANK_KEYWORDS + r")(?:\s*" + _VN_BANK_NAMES + r")?"
            r"|" + _VN_BANK_NAMES + r"(?:\s*(?:" + _BANK_KEYWORDS + r"))?"
        r")\s*[:\-]?\s*"
        r"(?P<digits>\d{6,19})"
        r"|"
        # Nhánh 2: "0123... vcb" / "0123... mb bank"
        r"(?P<digits2>\d{6,19})\s+"
        r"(?P<kw2>(?:" + _VN_BANK_NAMES + r"|" + _BANK_KEYWORDS + r"))"
    r")",
    flags=re.IGNORECASE,
)

# 2. Số điện thoại Việt Nam: bắt đầu bằng +84, 02, 03, 05, 06, 07, 09, hoặc tổng đài 1800/1900
_PHONE_PATTERN = re.compile(
    r"(?:\+84|84)[\s.\-]?[235789](?:[\s.\-]?\d){8}\b"
    r"|\b0[235789](?:[\s.\-]?\d){8}\b"
    r"|\b1[89]00(?:[\s.\-]?\d){4,6}\b"
)

# 3. Ngày tháng: "20/09/2026", "20-9-26", "31/08", hoặc "ngày 20 tháng 9 năm 2026".
#    "ngay" không dấu chỉ được coi là ngày khi đi kèm "thang" ("nhan ngay 500MB" nghĩa là nhận ngay).
_DATE_VERBAL_PATTERN = re.compile(
    r"\b(?:ngày\s*\d{1,2}\b(?![.,]\d)(?:\s*tháng\s*\d{1,2}\b)?"
    r"|ngay\s*\d{1,2}\s*thang\s*\d{1,2}\b)"
    r"(?:\s*n[ăa]m\s*\d{2,4}\b)?",
    flags=re.IGNORECASE,
)
_DATE_NUMERIC_PATTERN = re.compile(
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"
    r"|\b\d{1,2}/\d{1,2}\b(?!/)"
)

# 4. Thời gian: "14:30", "14h30", "7h", "22 giờ"
_TIME_PATTERN = re.compile(
    r"\b([01]?\d|2[0-3])(:[0-5]\d|h[0-5]?\d?|\s*giờ)\b", flags=re.IGNORECASE
)

# 5. Tiền: "500k", "1.5tr", "2 triệu", "500.000đ", "1.000.000", "1,000,000 VND"
#    Không bắt khi dính chữ phía trước (gói "ST30K"); "3.072 MB" hay "50.000 TB" không phải tiền.
_MONEY_PATTERN = re.compile(
    r"(?<!\w)(?:"
        r"\d+([.,]\d+)?\s*(k|tr|triệu|nghìn|ngàn|đ|d|vnđ|vnd)\b"
        r"|\d{1,3}(?:[.,]\d{3}){2,}(?:\s*(?:d|đ|vnđ|vnd)\b)?"
        r"|\d{1,3}(?:[.,]\d{3})+\s*(?:d|đ|vnđ|vnd)\b"
    r")",
    flags=re.IGNORECASE,
)

# 6. Các số còn lại: chỉ bắt số có từ 4 chữ số trở lên và đứng riêng (không dính chữ như 2048MB, GS25).
#    Dataset giữ nguyên số nhỏ: "giảm 60%", "3GB", "tặng 1 phần quà", "gọi 198".
_NUMBER_PATTERN = re.compile(r"(?<!\w)\d{4,}(?!\w)")

# 7. Tên người: chuỗi 1-4 từ viết hoa (Nguyen Van A, NGUYEN VAN A), khi chuỗi
#   a. đứng sau từ ngữ cảnh hoặc xưng hô: "Nguoi nhan: ...", "Quy khach ...", "anh Nam"...
#   b. bắt đầu bằng họ phổ biến và có ít nhất 2 từ: "Trần Thị Bình", "PHAM DUC ANH"
_NAME_CONTEXT_PATTERN = re.compile(
    r"\b(?:"
        r"người\s+nhận|nguoi\s+nhan|người\s+gửi|nguoi\s+gui|"
        r"chủ\s+(?:tài\s+khoản|tk)|chu\s+(?:tai\s+khoan|tk)|"
        r"quý\s+khách|quy\s+khach|khách\s+hàng|khach\s+hang|"
        r"ông\s*/\s*bà|ong\s*/\s*ba|anh\s*/\s*chị|anh\s*/\s*chi|"
        r"công\s+dân|cong\s+dan|nhân\s+viên|nhan\s+vien|"
        r"hộ\s+kinh\s+doanh|ho\s+kinh\s+doanh|họ\s+(?:và\s+)?tên|ho\s+(?:va\s+)?ten"
    r")\b\s*[:\-]?\s*",
    flags=re.IGNORECASE,
)
# Từ xưng hô phải viết thường ("anh Nam") để không bắt nhầm tin viết hoa từng chữ ("Anh Chi Oi")
_NAME_HONORIFIC_PATTERN = re.compile(
    r"(?<![^\W\d_])(?:anh|chị|em|ông|bà|cô|chú|bác|cậu|dì|bạn|a|c|e)\s+"
)
_VN_SURNAMES = {
    # Họ có dấu, đã bỏ những họ trùng với từ hoặc địa danh hay gặp: mai, cao, lý, hà, châu, thái, từ, triệu...
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ",
    "đặng", "bùi", "đỗ", "hồ", "ngô", "dương", "đinh", "trịnh", "trương",
    "lương", "đoàn", "quách", "kiều",
    # Họ không dấu: chỉ giữ những họ không trùng với từ thường gặp
    # (đã bỏ tran = trân trọng, pham = phạm/phẩm, phan = phần, dang = đang, le = lễ...)
    "nguyen", "huynh", "hoang", "bui", "quach",
}
# Những từ viết hoa nhưng không phải tên (dạng không dấu); gặp các từ này thì dừng chuỗi tên
_NAME_STOPWORDS = {
    "so", "ma", "sinh", "ngay", "tai", "dia", "oi", "da", "vui", "xin", "cam",
    "kinh", "gui", "toi", "den", "tu", "va", "cua", "voi", "la", "co", "can",
    "the", "otp", "chi", "quy", "khach", "nhan", "dang", "duoc", "pho", "ngan", "noi", "khong", "moi", "uu", "dai",
    "khuyen", "chua", "xac", "thuc", "cung", "cap", "tiet", "vao", "dich",
    "tang", "soan", "hot", "viet", "het", "han", "sap", "thoi", "lien", "he",
    "goi", "tin", "xu", "cuoc", "thong", "bao", "canh", "cuoi", "hoi", "nhap",
    "truy", "tra", "mua", "ban", "giam", "gia", "qua", "trung", "thuong",
}

_NAME_START_STOPWORDS = {"anh", "em", "ong", "ba", "ban"}
# Âm tiết tiếng Việt không dấu gồm phụ âm đầu, 1-3 nguyên âm và phụ âm cuối.
# Mẫu này loại được tên thương hiệu, từ tiếng Anh và chữ viết tắt: Zalo, Viettel, Shopee, CSKH, GS25...
_VN_SYLLABLE = re.compile(
    r"(?:ngh|ng|gh|gi|kh|nh|ph|qu|th|tr|ch|[bcdghklmnpqrstvx])?"
    r"[aeiouy]{1,3}"
    r"(?:ch|ng|nh|[cmnpt])?"
)
_NAME_WORD = re.compile(r"[^\W\d_]+")
_NAME_GAP = re.compile(r"[ \t]+")


def _fold(word: str) -> str:
    # Bỏ dấu và viết thường: "Nguyễn" thành "nguyen"
    word = unicodedata.normalize("NFD", word.lower()).replace("đ", "d")
    return "".join(c for c in word if unicodedata.category(c) != "Mn")


def _name_word_style(m: "re.Match[str]", first: bool = True):
    # Trả về "upper" (NGUYEN), "title" (Nguyen), "letter" (A), hoặc None nếu không phải tên
    word, text = m.group(), m.string
    # Từ dính liền số hoặc ký hiệu (GS25, SPAY_TRA, Laz+nx...) không phải tên
    if m.end() < len(text) and (text[m.end()].isalnum() or text[m.end()] in "_+"):
        return None
    folded = _fold(word)
    if first and folded in _NAME_START_STOPWORDS:
        return None
    if folded in _NAME_STOPWORDS or not _VN_SYLLABLE.fullmatch(folded):
        return None
    if word.isupper():
        return "upper" if len(word) > 1 else "letter"
    if word[0].isupper() and word[1:].islower():
        return "title"
    return None


def _name_end(text: str, pos: int, max_words: int = 4):
    """
    Đọc chuỗi từ viết hoa liền nhau bắt đầu tại `pos`, cùng một kiểu viết hoa;
    chữ cái đơn như "A" trong "Nguyen Van A" chỉ được đứng cuối.
    Trả về (vị trí kết thúc, số từ, kiểu viết hoa).
    """
    end, count, style = pos, 0, None
    while count < max_words:
        m = _NAME_WORD.match(text, pos)
        if not m:
            break
        s = _name_word_style(m, first=not count)
        if s is None or (s == "letter" and not count) or (style and s != "letter" and s != style):
            break
        style = style or s
        end, count = m.end(), count + 1
        if s == "letter":
            break
        gap = _NAME_GAP.match(text, end)
        if not gap:
            break
        pos = gap.end()
    return end, count, style


def tag_names(text: str) -> str:
    spans = []

    # a1. Sau từ ngữ cảnh: họ tên đầy đủ (từ 2 từ trở lên) hoặc 1 từ viết hoa toàn bộ
    #     ("Quý khách Vạn sự như ý" không phải tên)
    for m in _NAME_CONTEXT_PATTERN.finditer(text):
        end, count, style = _name_end(text, m.end())
        if count >= 2 or (count == 1 and style == "upper"):
            spans.append((m.end(), end))

    # a2. Sau từ xưng hô: 1-4 từ ("anh Nam", "chị THẢO")
    for m in _NAME_HONORIFIC_PATTERN.finditer(text):
        end, count, _ = _name_end(text, m.end())
        if count:
            spans.append((m.end(), end))

    # b. Bắt đầu bằng họ phổ biến: 2-4 từ
    for m in _NAME_WORD.finditer(text):
        if m.group().lower() in _VN_SURNAMES and _name_word_style(m):
            end, count, _ = _name_end(text, m.start())
            if count >= 2:
                spans.append((m.start(), end))

    # Gộp các đoạn chồng nhau rồi thay từ cuối chuỗi ngược lên đầu
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in reversed(merged):
        text = text[:start] + " [NAME] " + text[end:]
    return text


# Giữ lại từ khoá ngân hàng và chỉ thay số tài khoản, theo đúng thứ tự trong hai nhánh của mẫu
def _tag_bank_acc(m: "re.Match[str]") -> str:
    if m.group("kw") is not None:
        return f"{m.group('kw')} [BANK_ACC]"
    return f"[BANK_ACC] {m.group('kw2')}"

# Thay PII trong tin người dùng nhập bằng token của bộ dữ liệu
def tag_pii(text: str) -> str:
    text = str(text)

    text = tag_names(text)  # chạy trước các bước khác để "[PHONE]"... không bị nhận nhầm là tên
    text = _BANK_ACC_PATTERN.sub(_tag_bank_acc, text)
    text = _PHONE_PATTERN.sub(" [PHONE] ", text)
    text = _DATE_NUMERIC_PATTERN.sub(" [DATE] ", text)
    text = _DATE_VERBAL_PATTERN.sub(" [DATE] ", text)
    text = _TIME_PATTERN.sub(" [TIME] ", text)
    text = _MONEY_PATTERN.sub(" [MONEY] ", text)
    text = _NUMBER_PATTERN.sub(" [NUMBER] ", text)
    
    return re.sub(r"\s+", " ", text).strip()


def tag_links(text: str) -> str:
    # Chạy trước tag_pii để số nằm trong link không bị thay thành [NUMBER]
    return _URL_PATTERN_VISOBERT.sub(" [LINK] ", str(text))


def prepare_text(raw_text: str) -> str:
    # Chuẩn bị văn bản cho hai model TF-IDF theo từ và theo ký tự
    return preprocess_text(tag_pii(tag_links(raw_text)))


# Pipeline cho ViSoBERT. Model được train trên văn bản thô (giữ chữ hoa và dấu câu),
# nên chỉ thay link bằng [LINK] và chuẩn hoá Unicode NFC, không tách từ.


def prepare_text_visobert(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", str(raw_text))
    return tag_pii(tag_links(text))


# Pipeline cho PhoBERT. PhoBERT được pretrain trên văn bản đã tách từ ("trúng_thưởng"), nên xử lý
# như ViSoBERT rồi tách từ bằng pyvi. Token PII được giữ nguyên vì pyvi cắt "[MONEY]" thành "[ MONEY ]".
# Cách làm giống lúc train (Model_Train_Transformers_Compare.ipynb).

_PII_SPLIT_PATTERN = re.compile(r"(\[(?:" + "|".join(PII_TOKENS) + r")\])")
_segment_lock = threading.Lock()  # pyvi (CRF) không an toàn khi nhiều luồng gọi cùng lúc, mà FastAPI chạy request trong threadpool


def segment_vi(text: str) -> str:
    from pyvi import ViTokenizer  # model CRF chỉ được nạp ở lần gọi đầu tiên

    parts = _PII_SPLIT_PATTERN.split(str(text))
    with _segment_lock:
        return " ".join(
            p if _PII_SPLIT_PATTERN.fullmatch(p) else ViTokenizer.tokenize(p)
            for p in parts
            if p.strip()
        )


def prepare_text_phobert(raw_text: str) -> str:
    return segment_vi(prepare_text_visobert(raw_text))
