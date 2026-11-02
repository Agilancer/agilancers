"""Panel layouts of the lance and shield texture sheets (10 variants each).
Rects are panel-local pixel boxes (x0, y0, x1, y1)."""

LANCE_SHEET = 'assets/source/lance_sheet.png'
SHIELD_SHEET = 'assets/source/shield_sheet.png'

VARIANTS = ['iron', 'silver', 'gold', 'crimson', 'marble', 'azure', 'verdant', 'amethyst', 'solar', 'carbon']
VARIANT_LABELS = ['Weathered Iron', 'Polished Silver', 'Royal Gold', 'Crimson Shadow', 'White Marble',
                  'Azure Frost', 'Verdant Stag', 'Amethyst Crystal', 'Solar Ivory', 'Carbon Copper']

def lance_panel_origin(i, W=1254, H=1254):
    r, c = divmod(i, 5)
    return round(c * W / 5), round(r * H / 2)

# lance panel-local element rects (inset so sampling stays inside each element)
LANCE_RECTS = {
    'headA':  (8, 2, 136, 270),      # blade front view (mask derived by segmentation)
    'headB':  (136, 2, 248, 270),    # blade back view
    'shaft':  (16, 275, 92, 608),    # long shaft wrap
    'band1':  (102, 274, 156, 290),  # thin decorative band (column 2 top)
    'band2':  (104, 302, 156, 324),  # patterned band (column 2, second)
    'strip':  (106, 345, 150, 535),  # vertical strip panel (column 2 middle)
    'ring':   (176, 271, 236, 293),  # band (column 3 top)
    'collar': (176, 305, 236, 348),  # ornate socket collar (column 3)
    'ring2':  (176, 364, 236, 384),  # band (column 3)
    'grip':   (176, 400, 236, 565),  # grip wrap
    'butt':   (104, 580, 232, 606),  # bottom bar ornament
}

def shield_panel_origin(i, W=1774, H=887):
    r, c = divmod(i, 5)
    return round(c * W / 5), round(r * H / 2)

SHIELD_RECTS = {
    'front': (10, 0, 345, 330),      # main face (mask via segmentation; excludes the lower row)
    'back':  (8, 255, 205, 440),     # back face with straps
    'rim':   (214, 312, 252, 428),   # rim/edge strip
    'grip':  (278, 312, 320, 428),   # handle wrap
}
