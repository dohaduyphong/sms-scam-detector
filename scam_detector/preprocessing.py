"""
Tiền xử lý tin nhắn SMS Tiếng Việt
"""
import re
import string

# Các token thay thế PII có sẵn trong bộ dữ liệu
PII_TOKENS = ["PHONE", "BANK_ACC", "MONEY", "NUMBER", "TIME", "DATE"]

_PII_PLACEHOLDER = {
    tok: f'xxpii{tok.lower().replace("_", "")}xx' for tok in PII_TOKENS
}
_PII_RESTORE = {v: f"[{k}]" for k, v in _PII_PLACEHOLDER.items()}

_PII_PATTERN = re.compile(
    r"\[(" + "|".join(PII_TOKENS) + r")\]", flags=re.IGNORECASE
)


def preprocess_text(text: str) -> str:
    text = str(text)

    # Lưu các token PII có sẵn
    def _protect(match: "re.Match[str]") -> str:
        token = match.group(1).upper()
        return f" {_PII_PLACEHOLDER[token]} "

    text = _PII_PATTERN.sub(_protect, text)

    # Chuẩn hoá xâu (viết thường, ẩn link, ...)
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " url ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    # Khôi phục token PII
    for placeholder, restored in _PII_RESTORE.items():
        text = text.replace(placeholder, restored)

    return text


# ============================================================
# Chuẩn hoá thông tin định danh cá nhân (PII)
# trong các tin nhắn do người dùng nhập vào
# ============================================================

# 1. TKNH - Dựa trên các kí tự ở gần để phân biệt với sđt
_BANK_ACC_PATTERN = re.compile(
    r"(?P<kw>số\s*tài\s*khoản|stk|tk|tài\s*khoản)\s*[:\-]?\s*"
    r"(?P<digits>\d{6,19})",
    flags=re.IGNORECASE,
)

# 2. SĐT Việt Nam: Bắt đầu bằng 03/05/06/07/09
_PHONE_PATTERN = re.compile(
    r"(?:\+84|84)[\s.\-]?[35789](?:[\s.\-]?\d){8}\b"
    r"|\b0[35789](?:[\s.\-]?\d){8}\b"
)

# 3. Ngày/tháng/năm: "20/09/2026", "20-9-26", hoặc "ngày 20 tháng 9 năm 2026"
_DATE_VERBAL_PATTERN = re.compile(
    r"ng[àa]y\s*\d{1,2}(\s*th[áa]ng\s*\d{1,2})?(\s*n[ăa]m\s*\d{2,4})?",
    flags=re.IGNORECASE,
)
_DATE_NUMERIC_PATTERN = re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b")

# 4. Thời gian: "14:30", "14h30", "7h", "22 giờ"
_TIME_PATTERN = re.compile(
    r"\b([01]?\d|2[0-3])(:[0-5]\d|h[0-5]?\d?|\s*giờ)\b", flags=re.IGNORECASE
)

# 5. Tiền: "500k", "1.5tr", "2 triệu", "500.000đ", "1,000,000 VND"
_MONEY_PATTERN = re.compile(
    r"\d+([.,]\d+)?\s*(k|tr|triệu|nghìn|ngàn|đ|vnđ|vnd)\b"
    r"|\d{1,3}(?:[.,]\d{3})+\s*(đ|vnđ|vnd)?",
    flags=re.IGNORECASE,
)

# 6. Các số còn lại
_NUMBER_PATTERN = re.compile(r"\d+")


def tag_pii(text: str) -> str:
    """
    Phát hiện các thông tin định danh cá nhân trong văn bản ng dùng nhập vào, thay thế bằng token cho sẵn
    """
    text = str(text)

    text = _BANK_ACC_PATTERN.sub(lambda m: f"{m.group('kw')} [BANK_ACC]", text)
    text = _PHONE_PATTERN.sub(" [PHONE] ", text)
    text = _DATE_VERBAL_PATTERN.sub(" [DATE] ", text)
    text = _DATE_NUMERIC_PATTERN.sub(" [DATE] ", text)
    text = _TIME_PATTERN.sub(" [TIME] ", text)
    text = _MONEY_PATTERN.sub(" [MONEY] ", text)
    text = _NUMBER_PATTERN.sub(" [NUMBER] ", text)

    return re.sub(r"\s+", " ", text).strip()


def prepare_text(raw_text: str) -> str:
    return preprocess_text(tag_pii(raw_text))
