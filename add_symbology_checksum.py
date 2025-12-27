import pandas as pd



def code128_codeword_for_ascii_b(ch: str) -> int:
    o = ord(ch)
    if o < 32 or o > 126:
        raise ValueError(f"Code128-B only supports ASCII 32..126. Got {repr(ch)} (ord={o})")
    return o - 32  # 0..94


def calculate_code128_checksum_codeword_b(data: str) -> int:
    start_code_b = 104
    total = start_code_b
    for i, ch in enumerate(str(data), start=1):
        total += i * code128_codeword_for_ascii_b(ch)
    return total % 103


def encode_code128_checksum_char(codeword: int) -> str:
    if codeword < 0 or codeword > 102:
        raise ValueError(f"Invalid Code128 checksum codeword: {codeword}")
    return chr(0xE000 + codeword)


def calculate_mod10_check_digit(digits_body):
    reversed_digits = digits_body[::-1]
    odd_sum = sum(reversed_digits[0::2])
    even_sum = sum(reversed_digits[1::2])
    total = (odd_sum * 3) + even_sum
    check_digit = (10 - (total % 10)) % 10
    return str(check_digit)


def calculate_ean_upc_checksum(value, symbology):
    digits = [int(d) for d in str(value) if d.isdigit()]
    symbology = str(symbology).lower()

    # Use the standard body lengths (exclude existing check digit if present)
    if 'ean8' in symbology:
        body = digits[:7] if len(digits) >= 8 else digits
    elif 'upca' in symbology:
        body = digits[:11] if len(digits) >= 12 else digits
    elif 'ean13' in symbology:
        body = digits[:12] if len(digits) >= 13 else digits
    else:
        body = digits

    if not body:
        return ""

    return calculate_mod10_check_digit(body)

def calculate_itf_checksum(value, symbology):
    digits = [int(d) for d in str(value) if d.isdigit()]
    symbology = str(symbology).lower()

    if not digits:
        return ""

    # ITF-14 uses a 13-digit body + 1 check digit.
    if 'itf14' in symbology:
        body = digits[:13] if len(digits) >= 14 else digits
    # If the value looks like ITF-14 already (14 digits), assume last is check digit.
    elif len(digits) == 14:
        body = digits[:13]
    else:
        body = digits

    if not body:
        return ""

    return calculate_mod10_check_digit(body)

def calculate_code39_checksum(data):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"
    char_map = {c: i for i, c in enumerate(chars)}
    try:
        total = sum(char_map[c] for c in str(data).upper() if c in char_map)
        return chars[total % 43]
    except:
        return ""


def process_barcodes(input_file, output_file):
    df = pd.read_csv(input_file)

    def update_value(row):
        symbology = str(row['symbology']).lower()
        val = str(row['value'])

        checksum = ""
        if 'ean8' in symbology or 'upca' in symbology or 'ean13' in symbology:
            checksum = calculate_ean_upc_checksum(val, symbology)
        elif 'itf' in symbology:
            checksum = calculate_itf_checksum(val, symbology)
        elif 'code39' in symbology:
            checksum = calculate_code39_checksum(val)
        elif 'code128' in symbology:
            try:
                cw = calculate_code128_checksum_codeword_b(val)
                checksum = encode_code128_checksum_char(cw)
            except Exception:
                checksum = ""

        return f"{val}{checksum}"

    df['value'] = df.apply(update_value, axis=1)
    df.to_csv(output_file, index=False)
    print(f"Success! Processed file saved as: {output_file}")

# --- SETTINGS ---
INPUT_PATH = '/Users/sile/projects/warehouse_drone/dronio_projects/dronio-custom-barcode-model/my_dataset_2/train/labels_without_checksum.csv'  # Change this to your file path
OUTPUT_PATH = '/Users/sile/projects/warehouse_drone/dronio_projects/dronio-custom-barcode-model/my_dataset_2/train/labels.csv'

process_barcodes(INPUT_PATH, OUTPUT_PATH)