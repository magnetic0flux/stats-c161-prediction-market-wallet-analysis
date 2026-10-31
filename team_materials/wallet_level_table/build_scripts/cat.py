import re
CRYPTO = re.compile(r'(^|-)(btc|bitcoin|eth|ethereum|sol|solana|xrp|bnb|doge|dogecoin|hype|hyperliquid|zec|zcash|crypto|ltc|ada|cardano|sui|pepe|megaeth|microstrategy)(-|$)')
WEATHER = re.compile(r'temperature|precipitation|-rain|snow|hurricane|weather')
SPORTS_CODE = re.compile(r'^[a-z0-9]{2,6}-[a-z0-9]{2,6}-[a-z0-9]{2,6}-20\d\d-\d\d-\d\d')
SPORTS_WORD = re.compile(r'^(nfl|nba|mlb|nhl|epl|ucl|uel|atp|wta|f1|ufc|cs2|lol|dota2|val|fif|unl|afcq|conl|cbb|cfb|wnba|mls|ipl|bun|lal|sea|fl1)-|grand-prix|premier-league|champions-league|world-series|super-bowl|-vs-|counter-strike|valorant|league-of-legends')
def category(slug: str) -> str:
    s = slug.lower()
    if 'updown' in s or '-up-or-down' in s: return 'crypto_updown'
    if CRYPTO.search(s): return 'crypto_price'
    if WEATHER.search(s): return 'weather'
    if SPORTS_CODE.search(s) or SPORTS_WORD.search(s): return 'sports_esports'
    return 'other'
