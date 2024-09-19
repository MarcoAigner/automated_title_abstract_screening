import os
import dotenv
import deepl
import langdetect


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


def translate_non_english(text: str) -> str:
    """Detect a non-English text and translate it to English using Deepl

    Args:
    text: str: a string to be translated

    Returns:
    str: the translated text
    """

    # Detect the language
    try:
        source_lang = langdetect.detect(text[:500]).upper()
    except:
        return text

    # Translate the text if it is not in English
    if source_lang != 'EN':
        try:
            return deepl_translate(text, source_lang=source_lang)
        except:
            return text
    else:
        return text
