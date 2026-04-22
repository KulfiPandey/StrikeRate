# pipeline/team_name_standardizer.py

def standardize_team_name(name):
    """Convert any team name variation to a standard short form."""
    name = str(name).strip().lower()
    
    mapping = {
        'kolkata knight riders': 'KKR',
        'kkr': 'KKR',
        'mumbai indians': 'MI',
        'mi': 'MI',
        'chennai super kings': 'CSK',
        'csk': 'CSK',
        'royal challengers bangalore': 'RCB',
        'royal challengers': 'RCB',
        'rcb': 'RCB',
        'rajasthan royals': 'RR',
        'rr': 'RR',
        'punjab kings': 'PBKS',
        'pbks': 'PBKS',
        'kings xi punjab': 'KXIP',  # older name
        'kxip': 'KXIP',
        'delhi capitals': 'DC',
        'dc': 'DC',
        'delhi daredevils': 'DD',   # older name
        'dd': 'DD',
        'lucknow super giants': 'LSG',
        'lsg': 'LSG',
        'gujarat titans': 'GT',
        'gt': 'GT',
        'sunrisers hyderabad': 'SRH',
        'srh': 'SRH',
        # Defunct teams (kept distinct so historical joins don't silently collide)
        'deccan chargers': 'DEC',
        'deccan': 'DEC',
        'kochi tuskers kerala': 'KTK',
        'ktk': 'KTK',
        'pune warriors': 'PW',
        'pw': 'PW',
        'gujarat lions': 'GL',
        'gl': 'GL',
        'rising pune supergiant': 'RPS',
        'rps': 'RPS',
    }
    
    # Try direct lookup, then try to find if name contains any key
    if name in mapping:
        return mapping[name]
    for key, std in mapping.items():
        if key in name:
            return std
    return name.upper()  # fallback