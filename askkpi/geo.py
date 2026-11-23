"""Geography reference: 50 states + DC, EIA census divisions, and the U.S. total."""
import re

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
REGIONS = {
    "NEW": "New England", "MAT": "Middle Atlantic", "ENC": "East North Central",
    "WNC": "West North Central", "SAT": "South Atlantic", "ESC": "East South Central",
    "WSC": "West South Central", "MTN": "Mountain", "PACC": "Pacific Contiguous",
    "PACN": "Pacific Noncontiguous",
}
NATIONAL = {"US": "United States"}
ALL = {**STATES, **REGIONS, **NATIONAL}

# codes that are ordinary English words; only matched when written in capitals AND no
# full-name match exists, and never matched inside a sentence in these cases
AMBIGUOUS_CODES = {"IN", "OR", "ME", "OK", "HI", "ID", "US", "DE", "LA", "MA", "AS", "OH"}

LOWER_OK = {"TX", "CA", "NY", "FL", "NJ", "WA", "GA", "NC", "SC", "VA", "WI", "MN", "AZ", "TN", "KY", "UT",
            "NV", "NM", "KS", "CT", "RI", "NH", "VT", "ND", "SD", "WY", "AK", "AR", "IA", "MS", "MT", "WV", "NE"}

_ALIASES = {
    "U.S.": "US", "united states": "US", "national": "US", "nationwide": "US",
    "the nation": "US", "usa": "US", "america": "US", "u.s": "US",
    "washington dc": "DC", "washington d.c.": "DC", "d.c.": "DC",
    "new england": "NEW", "middle atlantic": "MAT", "mid-atlantic": "MAT",
    "mid atlantic": "MAT", "east north central": "ENC", "west north central": "WNC",
    "south atlantic": "SAT", "east south central": "ESC", "west south central": "WSC",
    "mountain region": "MTN", "mountain division": "MTN",
}


def _name_index():
    idx = {}
    for code, name in ALL.items():
        idx[name.lower()] = code
    for alias, code in _ALIASES.items():
        idx[alias.lower()] = code
    return idx


_IDX = _name_index()
# longest first so "west virginia" wins over "virginia", "new york" over "york"
_PATTERNS = sorted(_IDX, key=len, reverse=True)


def find_geos(text):
    """Return geography codes mentioned in text, in order of appearance, without duplicates."""
    low = text.lower()
    hits = []  # (start, end, code)
    taken = []
    for name in _PATTERNS:
        for m in re.finditer(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", low):
            s, e = m.span()
            if any(not (e <= a or s >= b) for a, b in taken):
                continue
            taken.append((s, e))
            hits.append((s, e, _IDX[name]))
    # two-letter state codes written in capitals ("TX", "CA")
    for m in re.finditer(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])", text):
        code = m.group(1)
        s, e = m.span()
        if code in STATES and code not in AMBIGUOUS_CODES and not any(not (e <= a or s >= b) for a, b in taken):
            taken.append((s, e))
            hits.append((s, e, code))
    # lowercase codes ("tx commercial price") for codes that are not ordinary words
    for m in re.finditer(r"(?<![A-Za-z])([a-z]{2})(?![A-Za-z])", text):
        code = m.group(1).upper()
        s, e = m.span()
        if code in STATES and code in LOWER_OK and not any(not (e <= a or s >= b) for a, b in taken):
            taken.append((s, e))
            hits.append((s, e, code))
    hits.sort()
    seen, out = set(), []
    for _, _, c in hits:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def name(code):
    return ALL.get(code, code)
