"""
Category taxonomy for synthetic T2I prompt generation.
"""

TAXONOMY = {
    "Nature": {
        "Objects": {
            "input": "everyday or exotic object name",
            "weight": 5,
            "seeds": ["vintage pocket watch", "ceramic teapot", "leather satchel",
                      "brass telescope", "crystal decanter", "mechanical keyboard",
                      "antique typewriter", "hand-blown glass vase", "carbon-fiber bicycle",
                      "wooden chess set", "copper kettle", "origami crane"],
        },
        "Cityscape": {
            "input": "city or urban location type",
            "weight": 7,
            "seeds": ["neon-lit Tokyo alley", "Venetian canal at dawn", "Manhattan skyline",
                      "Marrakech medina", "Hong Kong night market", "Parisian boulevard",
                      "Dubai marina", "favela hillside", "industrial dockyard",
                      "old European tram street", "rooftop helipad", "subway platform"],
        },
        "Food": {
            "input": "dish, drink, or ingredient",
            "weight": 8,
            "seeds": ["ramen bowl with soft egg", "molten chocolate lava cake",
                      "fresh sushi platter", "wood-fired margherita pizza", "matcha latte art",
                      "sizzling fajitas", "macarons in pastel rows", "dripping honeycomb",
                      "espresso with crema", "charcuterie board", "mango sticky rice",
                      "steaming dumplings"],
        },
        "Plants": {
            "input": "specific plant or flower species",
            "weight": 5,
            "seeds": ["monstera deliciosa", "bonsai juniper", "venus flytrap", "cherry blossom branch",
                      "succulent terrarium", "fern unfurling", "lavender field", "lotus on water",
                      "moss-covered bark", "dew-laden spiderweb on leaves", "saguaro cactus",
                      "orchid macro"],
        },
        "Indoor": {
            "input": "interior space / room type",
            "weight": 4,
            "seeds": ["mid-century living room", "industrial loft kitchen", "cozy reading nook",
                      "Scandinavian bedroom", "Art Deco hotel lobby", "rustic farmhouse dining room",
                      "minimalist home office", "Japanese tatami room", "library reading hall",
                      "greenhouse conservatory", "vintage barbershop", "speakeasy bar"],
        },
        "Landscape": {
            "input": "natural landscape / vista",
            "weight": 9,
            "seeds": ["misty mountain valley", "salt flats at sunset", "northern lights over fjord",
                      "rolling Tuscan hills", "desert dunes", "alpine glacier lake", "bamboo forest path",
                      "volcanic black-sand beach", "autumn maple forest", "rice terraces",
                      "stormy coastal cliffs", "starry desert sky"],
        },
        "Animals": {
            "input": "specific animal species",
            "weight": 8,
            "seeds": ["Bengal tiger", "snow leopard", "red fox in snow", "humpback whale breaching",
                      "barn owl in flight", "poison dart frog", "Arctic wolf", "chameleon on branch",
                      "hummingbird mid-hover", "African elephant herd", "koi fish", "peacock displaying"],
        },
    },
    "Design": {
        "Arts": {
            "input": "art subject + medium",
            "weight": 8,
            "seeds": ["oil portrait of an old fisherman", "watercolor mountain village",
                      "charcoal figure study", "ukiyo-e wave print", "digital concept art of a space station",
                      "impressionist garden", "ink wash bamboo", "pop-art self portrait",
                      "stained glass rose window", "gouache still life", "linocut bird", "pastel sunset"],
        },
        "Posters": {
            "input": "poster theme (MUST render headline text)",
            "weight": 4,
            "seeds": ['music festival poster reading "SUNBURST 2026"',
                      'minimalist movie poster reading "THE LAST SIGNAL"',
                      'travel poster reading "VISIT ICELAND"',
                      'gym ad reading "NO EXCUSES"', 'coffee shop poster reading "FRESHLY BREWED"',
                      'sci-fi event poster reading "NEON NIGHTS"',
                      'vintage circus poster reading "THE GRAND SHOW"',
                      'product launch reading "INTRODUCING AURA"'],
        },
        "Slides": {
            "input": "presentation slide topic (MUST have a title/heading)",
            "weight": 2,
            "seeds": ['title slide "Q3 Growth Strategy"', 'slide titled "Climate Trends 2026"',
                      'pitch slide "Our Mission"', 'data slide "Revenue by Region"',
                      'agenda slide "Today We Will Cover"', 'team slide "Meet the Founders"',
                      'roadmap slide "Product Timeline"', 'thank-you slide "Questions?"'],
        },
        "Cartoon": {
            "input": "illustration / cartoon scene + style",
            "weight": 7,
            "seeds": ["anime girl under cherry blossoms", "western comic superhero landing",
                      "Pixar-style robot", "Saturday-morning cartoon dog", "chibi wizard casting a spell",
                      "noir comic detective", "flat-design fox mascot", "Studio-Ghibli meadow",
                      "retro 1930s rubber-hose character", "webtoon city street", "claymation monster",
                      "graphic-novel cyberpunk alley"],
        },
        "UI": {
            "input": "app/web interface screen (MUST include buttons/components)",
            "weight": 2,
            "seeds": ["fitness app dashboard", "banking app home screen", "music player UI",
                      "food delivery checkout", "weather app widget", "e-commerce product page",
                      "smart-home control panel", "travel booking flow", "meditation app onboarding",
                      "SaaS analytics dashboard", "messaging app chat screen", "dark-mode settings page"],
        },
        "Others": {
            "input": "abstract / surreal / mixed-media concept",
            "weight": 5,
            "seeds": ["fluid acrylic pour abstraction", "surreal floating islands", "fractal geometry",
                      "double-exposure portrait", "glitch-art landscape", "kaleidoscopic mandala",
                      "liquid metal sculpture", "paper-cut layered diorama", "smoke photography",
                      "isometric impossible architecture", "vaporwave statue", "macro oil-and-water bubbles"],
        },
    },
    "People": {
        "Portrait": {
            "input": "person / character for a close-up portrait",
            "weight": 8,
            "seeds": ["elderly woman with weathered hands", "young man with freckles",
                      "ballerina backstage", "tattooed chef", "Maasai elder", "businesswoman in office light",
                      "child blowing bubbles", "fashion model in studio softbox", "fisherman at dawn",
                      "musician with vintage guitar", "scientist in lab coat", "dancer in motion"],
        },
        "Sports": {
            "input": "sport / athletic action",
            "weight": 5,
            "seeds": ["sprinter exploding off blocks", "surfer in a barrel wave",
                      "basketball slam dunk", "rock climber on overhang", "soccer bicycle kick",
                      "skateboarder mid-ollie", "marathon runners in rain", "boxer landing a punch",
                      "skier carving powder", "gymnast on beam", "cyclist peloton", "diver mid-twist"],
        },
        "Activities": {
            "input": "daily-life activity / scenario",
            "weight": 6,
            "seeds": ["barista pulling espresso", "potter at the wheel", "farmer harvesting at sunset",
                      "street vendor cooking", "carpenter sanding wood", "painter at an easel",
                      "gardener pruning roses", "blacksmith hammering", "tailor measuring fabric",
                      "fisherman casting a net", "baker kneading dough", "florist arranging a bouquet"],
        },
        "Others": {
            "input": "crowd / silhouette / abstract human representation",
            "weight": 3,
            "seeds": ["crowd silhouettes at a concert", "commuters blurred in motion",
                      "lone figure on a foggy bridge", "protest march from above", "dancers as light trails",
                      "shadows on a sunlit wall", "festival crowd with raised hands",
                      "silhouetted hikers on a ridge", "subway crowd reflection", "marathon start aerial",
                      "beach crowd from a drone", "stadium wave"],
        },
    },
    "Synthetic": {
        "English Text": {
            "input": "scene that MUST render a specific English phrase",
            "weight": 3,
            "seeds": ['neon sign reading "OPEN 24 HOURS"', 'chalkboard menu reading "TODAY\'S SPECIAL"',
                      'storefront reading "BOOKS & COFFEE"', 'graffiti wall reading "DREAM BIG"',
                      'license plate reading "ROAD TRIP"', 'birthday cake reading "HAPPY 30TH"',
                      'street sign reading "MAIN STREET"', 'mug reading "BEST DAD EVER"',
                      'banner reading "GRAND OPENING"', 'tattoo reading "carpe diem"'],
        },
        "Others": {
            "input": "numbers, code snippet, or mathematical formula to render",
            "weight": 2,
            "seeds": ["chalkboard with the quadratic formula", "screen showing a Python function",
                      "clock face showing 3:15", "scoreboard reading 102-98", "calendar page for June",
                      "circuit diagram", "binary code rain", "blackboard with E=mc^2",
                      "spreadsheet of quarterly figures", "speedometer at 88 mph",
                      "periodic table excerpt", "sheet music staff"],
        },
    },
}

def all_subclasses():
    """Yield (super_class, sub_class, meta) for every leaf category."""
    for sup, subs in TAXONOMY.items():
        for sub, meta in subs.items():
            yield sup, sub, meta

