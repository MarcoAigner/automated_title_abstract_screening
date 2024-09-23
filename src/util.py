import os
import dotenv
import deepl
from fast_langdetect import detect


def deepl_translate(text: str, source_lang=None, target_lang: str = 'EN-US') -> str:
    """Translate a text to a target language

    Args:
    text: str: a string to be translated
    target_lang: str: the target language

    Returns:
    str: the translated text
    """

    # Initialize the Deepl API
    dotenv.load_dotenv()
    deepl_api_key = os.getenv('DEEPL')
    translator = deepl.Translator(deepl_api_key)

    # Translate the text
    translated = translator.translate_text(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang
    )

    return translated.text


def is_text_english(text: str, first_n_characters=None) -> bool:
    """Detect if a text is in English

    Args:
    text: str: a string to be detected

    Returns:
    bool: True if the text is in English, False otherwise
    """

    try:

        if text is None:
            return None
        else:
            text = text.replace('\n', '')  # remove newlines

            detection = detect(text[:first_n_characters])
            detected_language = detection['language']

            if detected_language == 'en':
                is_english = True
            else:
                is_english = False

            return is_english
    except:
        return False


def detect_language(text: str) -> str:
    """Detect the language of a text

    Args:
    text: str: a string to be detected

    Returns:
    str: the detected language
    """
    try:
        if text is None:
            return None
        else:
            # lowercase, remove newlines, and strip whitespaces
            text = text.lower().replace('\n', '').strip()

            detection = detect(text)
            detected_language = detection['lang']

            return detected_language
    except:
        return None


def translate_non_english(text: str) -> str:
    """Detect a non-English text and translate it to English using Deepl

    Args:
    text: str: a string to be translated

    Returns:
    str: the translated text
    """

    # Check if the text is in English
    is_english = is_text_english(text)

    # Translate the text if it is not in English
    if not is_english:
        try:
            return deepl_translate(text)
        except:
            return text
    else:
        return text
