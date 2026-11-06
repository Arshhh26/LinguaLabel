"""
LinguaLabel — Multilingual Text Normalization & Transliteration Layer (Part 11.2)

Provides context-preserving normalization for English, Hindi, and Hinglish.
Features:
1. Leetspeak & symbol substitution unmasking (@ -> a, 0 -> o, $ -> s, etc.)
2. Repeated character reduction (e.g. 'chuuuutiyaaa' -> 'chutiya')
3. Masked vowel / asterisk handling ('ch*tiya', 'f**k', 'm*d*rchod')
4. Spacing collapse for segmented abuse ('madar chod', 'm a d a r c h o d')
5. Devanagari phonetic mapping for cross-script equivalence
6. Contextual pattern extraction (quoted abuse, meta-discussion, threats, target indicators)

IMPORTANT:
- Normalization is used ONLY internally for detection and feature extraction.
- The original user text is ALWAYS preserved exactly without alteration or censorship.
"""

import re
from typing import Dict, List, Set, Tuple


# Leetspeak / masking character substitution table
LEET_CHAR_MAP: Dict[str, str] = {
    "@": "a",
    "4": "a",
    "8": "b",
    "©": "c",
    "3": "e",
    "€": "e",
    "1": "i",
    "!": "i",
    "|": "i",
    "0": "o",
    "$": "s",
    "5": "s",
    "7": "t",
    "+": "t",
    "v": "u",  # Contextual romanized hindi substitution
}

# Devanagari to Romanized phonetic map for abusive roots
DEV_TO_ROMAN_MAP: Dict[str, str] = {
    "मादरचोद": "madarchod",
    "मादर चोद": "madarchod",
    "चूतिया": "chutiya",
    "चूतिये": "chutiye",
    "चूतिए": "chutiye",
    "चूत": "chut",
    "गांड": "gaand",
    "गाँड": "gaand",
    "गांडू": "gaandu",
    "लंड": "lund",
    "लौड़ा": "lauda",
    "लौड़े": "laude",
    "तेरी माँ की चूत": "teri maa ki chut",
    "तेरी मां की चूत": "teri maa ki chut",
    "भोसड़ीके": "bhosdike",
    "भोसड़ी के": "bhosdike",
    "कमीने": "kamine",
    "कमीना": "kamina",
    "हरामी": "harami",
    "साले": "saale",
    "कुत्ते": "kutte",
    "रंडी": "randi",
}


def unmask_leetspeak(text: str) -> str:
    """Replaces common leetspeak substitutions with standard Latin characters."""
    result = []
    for ch in text:
        lower_ch = ch.lower()
        if lower_ch in LEET_CHAR_MAP:
            result.append(LEET_CHAR_MAP[lower_ch])
        elif lower_ch == "*":
            # Wildcard masking placeholder
            result.append("*")
        else:
            result.append(lower_ch)
    return "".join(result)


def reduce_repeated_characters(text: str) -> str:
    """
    Compresses 3 or more repeated characters down to 2 or 1 (e.g. 'fuuuck' -> 'fuck', 'chuuutiya' -> 'chutiya').
    Preserves valid double consonants/vowels ('ee', 'oo', 'aa') while collapsing intentional elongations.
    """
    # Replace 3 or more identical consecutive characters with 1
    return re.sub(r"(.)\1{2,}", r"\1", text)


def normalize_hinglish_variations(text: str) -> str:
    """
    Normalizes common Hinglish spelling variants into canonical phonetic stems.
    Handles 'oo' -> 'u', 'ee' -> 'i', 'aa' -> 'a', etc.
    """
    t = text.lower()
    t = re.sub(r"ch[\*u0]+tiy[ae]", "chutiya", t)
    t = re.sub(r"ch00tiya", "chutiya", t)
    t = re.sub(r"chootiya", "chutiya", t)
    t = re.sub(r"chuty[ae]", "chutiya", t)
    t = re.sub(r"maderchod", "madarchod", t)
    t = re.sub(r"madarch[o0]d", "madarchod", t)
    t = re.sub(r"bhenchod|behenchod", "bhenchod", t)
    t = re.sub(r"g[a\*]+nd", "gand", t)
    t = re.sub(r"\bl[au\*]+nd\b", "lund", t)
    t = re.sub(r"f[\*u]+ck", "fuck", t)
    t = re.sub(r"f[\*]+k", "fuck", t)
    t = re.sub(r"b[\*i]+tch", "bitch", t)
    t = re.sub(r"sh[\*i]+t", "shit", t)
    t = re.sub(r"teri\s+m[a]+[h\*\s]?\s+ki\s+ch[u\*]+t", "teri maa ki chut", t)
    t = re.sub(r"tmkc", "teri maa ki chut", t)
    t = re.sub(r"mc\b", "madarchod", t)
    t = re.sub(r"bc\b", "bhenchod", t)
    return t


def normalize_devanagari_abuse(text: str) -> str:
    """Translates known Devanagari abusive roots to Romanized equivalents for dual-pass evaluation."""
    res = text
    for dev, rom in DEV_TO_ROMAN_MAP.items():
        if dev in res:
            res = res.replace(dev, f" {rom} ")
    return res


def get_normalized_representations(original_text: str) -> Dict[str, str]:
    """
    Computes multiple normalized views of the original text for multi-stage matching:
    1. 'clean_lower': lowercased with preserved spacing
    2. 'unmasked': leetspeak and symbols replaced
    3. 'reduced': repeated characters compressed
    4. 'devanagari_mapped': Devanagari abusive terms translated to phonetic Roman stems
    5. 'collapsed_spacing': removes spaces between characters to detect split words ('m a d a r c h o d')
    """
    raw = (original_text or "").strip()
    clean_lower = raw.lower()

    dev_mapped = normalize_devanagari_abuse(clean_lower)
    unmasked = unmask_leetspeak(dev_mapped)
    reduced = reduce_repeated_characters(unmasked)
    hinglish_norm = normalize_hinglish_variations(reduced)

    # Collapsed spacing representation (letters and digits only)
    collapsed = re.sub(r"[^a-z0-9]", "", reduced)

    return {
        "original": raw,
        "clean_lower": clean_lower,
        "unmasked": unmasked,
        "reduced": reduced,
        "hinglish_norm": hinglish_norm,
        "collapsed": collapsed,
    }


# =============================================================================
# CONTEXTUAL PATTERN DETECTION
# =============================================================================

# Patterns identifying quoted abuse (someone reporting or quoting what another said)
QUOTED_ABUSE_PATTERNS = [
    re.compile(r"\b(someone|somebody|he|she|they|user|guy|person)\s+(called|named|texted|said to|messaged|labeled|screamed|yelled|shouted)\s+(me|us|at\s+[\w\s]+)?\s*[\"']?([a-zA-Z\s]+)[\"']?", re.IGNORECASE),
    re.compile(r"\b(i was called|got called|was addressed as)\s+[\"']?([a-zA-Z\s]+)[\"']?", re.IGNORECASE),
    re.compile(r"\b(mujhe|usne|unhone|kisi ne)\s+([a-zA-Z\s]+)\s+(bola|kaha|bol diya)", re.IGNORECASE),
    re.compile(r"['\"][^'\"]*['\"]\s*(was said|was written|is what he said|at the\s+[\w\s]+)", re.IGNORECASE),
    re.compile(r"\b(screamed|yelled|shouted|said)\s+[\"'][^\"']+[\"']", re.IGNORECASE),
]

# Patterns identifying academic, educational, or reporting discussion of abusive terms
DISCUSSION_PATTERNS = [
    re.compile(r"\b(the word|the term|the phrase|the expression)\s+['\"]?([a-zA-Z\u0900-\u097F\s]+)['\"]?\s+(is used|means|is an abusive|is a slur|is considered|is profanity|is studied)", re.IGNORECASE),
    re.compile(r"\b(in linguistics|linguistically|in grammar)\b.*?\b(the term|the word|is studied|as an insult|as profanity)\b", re.IGNORECASE),
    re.compile(r"\b(reporting\s+an?\s+(abusive|offensive|threatening|harassing|vulgar)\s+(message|comment|user|post|text))", re.IGNORECASE),
    re.compile(r"\b(example of|instance of|definition of)\s+(profanity|abusive language|vulgarity|harassment)", re.IGNORECASE),
    re.compile(r"\b(is\s+['\"][a-zA-Z\s]+['\"]\s+an\s+insult)\b", re.IGNORECASE),
]

# Patterns identifying explicit threats of violence or physical harm
THREAT_PATTERNS = [
    re.compile(r"\b(i will|i'll|im gonna|i am going to)\s+(hurt|kill|shoot|stab|beat|destroy|break|murder|damage|harm|hunt\s+you\s+down)\s*(you|u|your family|your head|ur ass|tonight|tomorrow)?\b", re.IGNORECASE),
    re.compile(r"\b(hunt you down|kill you|beat you up until you bleed|watch your back|you will pay for this|you are dead|you're dead)\b", re.IGNORECASE),
    re.compile(r"\b(jaan se maar|maar dalunga|dekh lunga|tujhe chhodunga nahi|pitai hogi|hath paon tod dunga|khatam kar dunga)\b", re.IGNORECASE),
    re.compile(r"\b(जान से मार|मार डालूंगा|देख लूँगा|तुझे छोडूंगा नहीं|पीट दूंगा)\b", re.IGNORECASE),
]

# Patterns identifying 2nd person targeted personal attacks vs object/product criticism
TARGETED_YOU_PATTERNS = [
    re.compile(r"\b(you are|u are|you're|u're|ur|you)\s+(an?|such an?|a complete|a total|the biggest)?\s*(idiot|moron|stupid|chutiya|chutiye|bastard|asshole|dumb|loser|dog|pig|kutta|kamina|harami|gadha|ullu)\b", re.IGNORECASE),
    re.compile(r"\b(tum|tu|teri|tera)\s+(ek\s+)?(chutiya|chutiye|gadha|ullu|harami|kutta|kamine|suar)\s+(ho|hai)?\b", re.IGNORECASE),
    re.compile(r"\b(तुम|तू)\s+(एक\s+)?(चूतिया|गधा|उल्लू|हरामी|कमीने|कुत्ता)\s+(हो|है)?\b", re.IGNORECASE),
]

# Patterns identifying criticism directed at product/software/service (NOT personal abuse)
PRODUCT_CRITICISM_PATTERNS = [
    re.compile(r"\b(this|your|the)\s+(app|application|service|software|website|system|delivery|support|phone|update)\s+(is|was|feels|has been)?\s*(terrible|horrible|awful|pathetic|useless|stupid|bad|slow|waste|broken|worst)\b", re.IGNORECASE),
    re.compile(r"\b(app|application|service)\s+(bakwas|bekar|ghatiya|chutiya|faltu|thik nahi)\s+(hai|chal raha)\b", re.IGNORECASE),
    re.compile(r"\b(ye|yeh|tumhara)\s+(app|service)\s+(bahut\s+)?(kharab|bakwaas|bekar|ghatiya)\s+hai\b", re.IGNORECASE),
    re.compile(r"\b(यह|आपकी)\s+(सेवा|ऐप|एप्लिकेशन)\s+(बहुत\s+)?(खराब|बेकार|घटिया)\s+है\b", re.IGNORECASE),
]


def detect_context_flags(original_text: str) -> Dict[str, bool]:
    """
    Analyzes contextual framing to prevent false positives:
    - is_quoted: Quoted abuse reported by user rather than expressed by user
    - is_discussion: Meta, educational, or reporting discussion of abuse
    - is_threat: Explicit physical threat or intimidation
    - is_targeted_personal: Direct 2nd person personal attack
    - is_product_criticism: Discontent directed at product/service rather than a person
    """
    text = original_text.strip()
    flags = {
        "is_quoted": False,
        "is_discussion": False,
        "is_threat": False,
        "is_targeted_personal": False,
        "is_product_criticism": False,
    }

    # 1. Quoted check
    for pat in QUOTED_ABUSE_PATTERNS:
        if pat.search(text):
            flags["is_quoted"] = True
            break

    # 2. Discussion / reporting check
    for pat in DISCUSSION_PATTERNS:
        if pat.search(text):
            flags["is_discussion"] = True
            break

    # 3. Threat check
    for pat in THREAT_PATTERNS:
        if pat.search(text):
            flags["is_threat"] = True
            break

    # 4. Targeted personal insult check
    for pat in TARGETED_YOU_PATTERNS:
        if pat.search(text):
            flags["is_targeted_personal"] = True
            break

    # 5. Product criticism check
    for pat in PRODUCT_CRITICISM_PATTERNS:
        if pat.search(text):
            flags["is_product_criticism"] = True
            break

    return flags
