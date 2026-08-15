import os
import re
import shutil

import pytesseract

from PIL import ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output


# ==================================================
# CONFIGURE TESSERACT
# ==================================================

def configure_tesseract():

    # First try system PATH
    path = shutil.which("tesseract")

    if path:
        pytesseract.pytesseract.tesseract_cmd = path
        return

    # Common Windows locations
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]

    for path in possible_paths:

        if os.path.exists(path):

            pytesseract.pytesseract.tesseract_cmd = path
            return


configure_tesseract()


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image):

    image = image.convert("RGB")

    width, height = image.size

    # Make smaller screenshots easier for OCR
    if width < 1400:

        scale = 1400 / width

        image = image.resize(
            (
                int(width * scale),
                int(height * scale)
            )
        )

    # Grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    enhancer = ImageEnhance.Contrast(image)

    image = enhancer.enhance(1.4)

    # Slight sharpening
    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


# ==================================================
# GET OCR LINES + POSITIONS
# ==================================================

def get_ocr_lines(image):

    processed_image = preprocess_image(
        image
    )

    data = pytesseract.image_to_data(
        processed_image,
        lang="eng",
        config="--psm 11",
        output_type=Output.DICT
    )

    grouped_lines = {}

    total_words = len(
        data["text"]
    )

    for i in range(total_words):

        word = data["text"][i].strip()

        try:
            confidence = float(
                data["conf"][i]
            )
        except Exception:
            confidence = -1

        # Ignore empty / very uncertain OCR words
        if not word or confidence < 30:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        if key not in grouped_lines:

            grouped_lines[key] = []

        grouped_lines[key].append(
            {
                "text": word,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i]
            }
        )

    lines = []

    for words in grouped_lines.values():

        # Sort words from left to right
        words = sorted(
            words,
            key=lambda x: x["left"]
        )

        line_text = " ".join(
            word["text"]
            for word in words
        )

        left = min(
            word["left"]
            for word in words
        )

        top = min(
            word["top"]
            for word in words
        )

        right = max(
            word["left"] + word["width"]
            for word in words
        )

        lines.append(
            {
                "text": line_text,
                "left": left,
                "top": top,
                "right": right
            }
        )

    # Sort from top of image to bottom
    lines = sorted(
        lines,
        key=lambda x: (
            x["top"],
            x["left"]
        )
    )

    return lines, processed_image.size[0]


# ==================================================
# CHECK INSTAGRAM METADATA
# ==================================================

def is_metadata_line(text):

    text = text.strip().lower()

    # UI words
    if text in {
        "comments",
        "comment",
        "reply",
        "replies",
        "like",
        "likes",
        "share"
    }:
        return True

    # Examples:
    # 2h
    # 45m
    # 30m
    if re.fullmatch(
        r"\d+\s*[smhdw]",
        text
    ):
        return True

    # Examples:
    # 1 like
    # 3 likes
    if re.fullmatch(
        r"\d+\s+likes?",
        text
    ):
        return True

    # Reply / view replies
    if re.fullmatch(
        r"(reply|view\s+\d+\s+replies)",
        text
    ):
        return True

    return False


# ==================================================
# CLEAN COMMENT
# ==================================================

def clean_comment_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    # Remove OCR garbage such as:
    # @&
    # <>
    # random symbol-only tokens
    while words:

        last_word = words[-1]

        if re.search(
            r"[A-Za-z0-9]",
            last_word
        ):
            break

        words.pop()

    return " ".join(words).strip()


# ==================================================
# SPLIT SCREENSHOT INTO COMMENTS
# ==================================================

def split_comments_from_lines(lines, image_width):

    comments = []

    current_comment = None

    # Instagram usernames may contain:
    # letters, numbers, underscore and dot

    username_pattern = re.compile(
        r"^("
        r"[A-Za-z0-9_]"
        r"(?:[A-Za-z0-9._]{0,28}"
        r"[A-Za-z0-9_])?"
        r")\s+(.+)$"
    )

    # Find y-positions of metadata lines
    metadata_tops = [
        line["top"]
        for line in lines
        if is_metadata_line(
            line["text"]
        )
    ]

    for line in lines:

        text = line["text"].strip()

        if not text:
            continue

        lower_text = text.lower()

        # Ignore title
        if lower_text in {
            "comments",
            "comment"
        }:
            continue

        # Ignore likes / timestamps / reply
        if is_metadata_line(text):
            continue

        # Sometimes OCR reads a tiny piece of
        # timestamp/avatar as "th", "Sr", etc.
        if len(text) <= 4:

            close_to_metadata = any(
                abs(
                    line["top"] - metadata_top
                ) <= 16
                for metadata_top
                in metadata_tops
            )

            if close_to_metadata:
                continue

        # Ignore small heart/share OCR garbage
        # found on extreme right of screenshot
        if (
            line["left"] >
            image_width * 0.80
            and len(text) <= 10
        ):
            continue

        match = username_pattern.match(
            text
        )

        new_comment = False

        if match:

            username = match.group(1)

            comment_text = match.group(2)

            # Strong signs that first token is
            # actually a username
            strong_username = (
                "_" in username
                or "." in username
                or any(
                    character.isdigit()
                    for character in username
                )
            )

            if current_comment is None:

                new_comment = True

            else:

                vertical_gap = (
                    line["top"]
                    - current_comment["last_top"]
                )

                # New comment if username looks strong
                # or if there is a large vertical gap
                if (
                    strong_username
                    or vertical_gap > 90
                ):

                    new_comment = True

            if username.lower() in {
                "comments",
                "comment",
                "reply",
                "like",
                "likes",
                "share"
            }:

                new_comment = False

        # ==========================================
        # START NEW COMMENT
        # ==========================================

        if new_comment:

            comment = {
                "text": comment_text.strip(),
                "left": line["left"],
                "top": line["top"],
                "last_top": line["top"]
            }

            comments.append(
                comment
            )

            current_comment = comment

            continue

        # ==========================================
        # WRAPPED SECOND LINE OF COMMENT
        # ==========================================

        if current_comment is not None:

            vertical_gap = (
                line["top"]
                - current_comment["last_top"]
            )

            # Wrapped comment normally starts
            # at almost the same horizontal position
            aligned = (
                current_comment["left"] - 35
                <= line["left"]
                <= current_comment["left"] + 180
            )

            # Wrapped line is normally close below
            close_below = (
                0 < vertical_gap <= 85
            )

            if aligned and close_below:

                current_comment["text"] += (
                    " " + text
                )

                current_comment["last_top"] = (
                    line["top"]
                )

    # ==============================================
    # CLEAN FINAL COMMENTS
    # ==============================================

    final_comments = []

    for comment in comments:

        text = clean_comment_text(
            comment["text"]
        )

        if (
            len(text) >= 3
            and text not in final_comments
        ):

            final_comments.append(
                text
            )

    return final_comments


# ==================================================
# MAIN OCR FUNCTION
# ==================================================

def extract_text_and_comments_from_image(image):

    lines, image_width = get_ocr_lines(
        image
    )

    # Raw OCR text
    raw_text = "\n".join(
        line["text"]
        for line in lines
    )

    # Separated Instagram comments
    comments = split_comments_from_lines(
        lines,
        image_width
    )

    return raw_text, comments


# ==================================================
# BACKWARD COMPATIBILITY
# ==================================================

def extract_text_from_image(image):

    raw_text, _ = (
        extract_text_and_comments_from_image(
            image
        )
    )

    return raw_text