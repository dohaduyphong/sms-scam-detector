"""
Tiền xử lý tin nhắn SMS Tiếng Việt
"""
import re
import string
import unicodedata

# Các token PII có sẵn trong bộ dữ liệu
PII_TOKENS = ["NAME", "PHONE", "BANK_ACC", "MONEY", "NUMBER", "TIME", "DATE", "LINK"]

# tạm thời thay các token PII
_PII_PLACEHOLDER = {
    tok: f'xxpii{tok.lower().replace("_", "")}xx' for tok in PII_TOKENS
}
# khôi phục các PII
_PII_RESTORE = {v: f"[{k}]" for k, v in _PII_PLACEHOLDER.items()}

_PII_PATTERN = re.compile(
    r"\[(" + "|".join(PII_TOKENS) + r")\]", flags=re.IGNORECASE
)

# Giống Model_Train_LR_SVM.ipynb
_URL_PATTERN = re.compile(
    r"(https?:\/\/\s*\S+)"
    r"|(www\.\S+)"
    r"|(\b[a-zA-Z0-9][a-zA-Z0-9\-]*\.(?:com|vn|net|org|co|io|me|shop|cc|info|biz|shop|top)(?:\.[a-zA-Z]{2,3})?(?:\/\S*)?)",
    flags=re.IGNORECASE,
)

# Giống Model_Train_ViSoBERT_Final.ipynb (thêm .ly)
_URL_PATTERN_VISOBERT = re.compile(
    r"(https?:\/\/\s*\S+)"
    r"|(www\.\S+)"
    r"|(\b[a-zA-Z0-9][a-zA-Z0-9\-]*\.(?:com|vn|net|org|co|io|me|shop|cc|info|biz|shop|top|ly)(?:\.[a-zA-Z]{2,3})?(?:\/\S*)?)",
    flags=re.IGNORECASE,
)

# Tiền xử lý cho TF-IDF
def preprocess_text(text: str) -> str:
    text = str(text)

    # Chuẩn hoá xâu (ẩn link, viết thường, ...)
    text = _URL_PATTERN.sub(" [LINK] ", text) # thay link bằng token

    # Lưu các token PII có sẵn
    def _protect(match: "re.Match[str]") -> str:
        token = match.group(1).upper()
        return f" {_PII_PLACEHOLDER[token]} "

    text = _PII_PATTERN.sub(_protect, text)

    text = text.lower() # viết thường
    text = text.translate(str.maketrans("", "", string.punctuation)) # bỏ dấu câu
    text = re.sub(r"\s+", " ", text).strip() # chuẩn hoá khoảng trắng

    # Khôi phục token PII
    for placeholder, restored in _PII_RESTORE.items():
        text = text.replace(placeholder, restored)
    return text


# ============================================================
# Chuẩn hoá thông tin định danh cá nhân (PII)
# trong các tin nhắn do người dùng nhập vào
# ============================================================

# 1. TKNH - Dựa trên các kí tự ở gần để phân biệt với sđt
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

# 2. SĐT Việt Nam: Bắt đầu bằng (+84)/02/03/05/06/07/09
_PHONE_PATTERN = re.compile(
    r"(?:\+84|84)[\s.\-]?[235789](?:[\s.\-]?\d){8}\b"
    r"|\b0[235789](?:[\s.\-]?\d){8}\b"
)

# 3. Ngày/tháng/năm: "20/09/2026", "20-9-26", hoặc "ngày 20 tháng 9 năm 2026"
_DATE_VERBAL_PATTERN = re.compile(
    r"ng[àa]y\s*\d{1,2}(\s*th[áa]ng\s*\d{1,2})?(\s*n[ăa]m\s*\d{2,4})?",
    flags=re.IGNORECASE,
)
_DATE_NUMERIC_PATTERN = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")

# 4. Thời gian: "14:30", "14h30", "7h", "22 giờ"
_TIME_PATTERN = re.compile(
    r"\b([01]?\d|2[0-3])(:[0-5]\d|h[0-5]?\d?|\s*giờ)\b", flags=re.IGNORECASE
)

# 5. Tiền: "500k", "1.5tr", "2 triệu", "500.000đ", "1,000,000 VND"
_MONEY_PATTERN = re.compile(
    r"\d+([.,]\d+)?\s*(k|tr|triệu|nghìn|ngàn|đ|d|vnđ|vnd)\b"
    r"|\d{1,3}(?:[.,]\d{3})+\s*(d|đ|vnđ|vnd)?",
    flags=re.IGNORECASE,
)

# 6. Các số còn lại
_NUMBER_PATTERN = re.compile(r"\d+")

# Giữ lại từ khoá ngân hàng, chỉ thay số TK -- đúng thứ tự của 2 nhánh
def _tag_bank_acc(m: "re.Match[str]") -> str:
    if m.group("kw") is not None:
        return f"{m.group('kw')} [BANK_ACC]"
    return f"[BANK_ACC] {m.group('kw2')}"

# Phát hiện các thông tin định danh cá nhân trong văn bản ng dùng nhập vào, thay thế bằng token cho sẵn
def tag_pii(text: str) -> str:
    text = str(text)

    text = _BANK_ACC_PATTERN.sub(_tag_bank_acc, text)
    text = _PHONE_PATTERN.sub(" [PHONE] ", text)
    text = _DATE_NUMERIC_PATTERN.sub(" [DATE] ", text)
    text = _DATE_VERBAL_PATTERN.sub(" [DATE] ", text)
    text = _TIME_PATTERN.sub(" [TIME] ", text)
    text = _MONEY_PATTERN.sub(" [MONEY] ", text)
    text = _NUMBER_PATTERN.sub(" [NUMBER] ", text)

    return re.sub(r"\s+", " ", text).strip()


def tag_links(text: str) -> str:
    # Thay thế các link bằng 1 token [LINK]
    # Chạy trước tag_pii để số trong link không bị tách thành [NUMBER]
    return _URL_PATTERN_VISOBERT.sub(" [LINK] ", str(text))


def prepare_text(raw_text: str) -> str:
    # Chuẩn bị text cho Word/Char TF-IDF
    return preprocess_text(tag_pii(tag_links(raw_text)))


# ============================================================
# Pipeline cho ViSoBERT
#
# ViSoBERT được train trên văn bản thô (giữ hoa/thường, dấu câu),
# chỉ thay link bằng [LINK] và chuẩn hoá Unicode NFC, không tách từ.
# ============================================================


def prepare_text_visobert(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", str(raw_text))
    return tag_pii(tag_links(text))
