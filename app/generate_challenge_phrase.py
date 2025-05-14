import random

def generate_challenge_phrase() -> str:
    template_options = [
        # Nature
        {
            "template": "Photographers capture the {adj} waterfall at {noun}",
            "adjectives": ["magnificent", "stunning", "spectacular", "majestic", "impressive", "famous", "picturesque"],
            "nouns": ["sunrise", "sunset", "dusk", "dawn", "twilight", "daybreak"]
        },
        {
            "template": "The {adj} mountain trail offers views of the {noun} below",
            "adjectives": ["scenic", "winding", "popular", "challenging", "peaceful", "historic", "steep"],
            "nouns": ["valley", "landscape", "forest", "countryside", "river", "terrain"]
        },
        {
            "template": "Hikers enjoy the {adj} forest path during {noun} hours",
            "adjectives": ["peaceful", "shaded", "quiet", "scenic", "tranquil", "wooded", "serene"],
            "nouns": ["morning", "afternoon", "evening", "daylight", "early", "quiet"]
        },
        {
            "template": "The {adj} beach becomes crowded during {noun} season",
            "adjectives": ["sandy", "popular", "public", "scenic", "tropical", "coastal", "tourist"],
            "nouns": ["summer", "tourist", "holiday", "peak", "vacation", "festival"]
        },
        {
            "template": "The {adj} park welcomes visitors throughout the {noun}",
            "adjectives": ["national", "city", "historic", "public", "famous", "popular", "community"],
            "nouns": ["summer", "winter", "season", "year", "weekend", "holidays"]
        },
        
        # Landmarks
        {
            "template": "The {adj} lighthouse stands on the {noun} coastline",
            "adjectives": ["historic", "tall", "famous", "iconic", "old", "restored", "white"],
            "nouns": ["rocky", "northern", "southern", "eastern", "western", "remote"]
        },
        {
            "template": "Tourists photograph the {adj} castle throughout the {noun}",
            "adjectives": ["medieval", "historic", "ancient", "famous", "royal", "restored", "majestic"],
            "nouns": ["day", "season", "summer", "year", "morning", "evening"]
        },
        {
            "template": "The {adj} museum features exhibits from {noun} artists",
            "adjectives": ["modern", "contemporary", "national", "famous", "local", "new", "interactive"],
            "nouns": ["local", "international", "renowned", "contemporary", "historical", "talented"]
        },
        {
            "template": "The {adj} bridge connects the eastern and western {noun}",
            "adjectives": ["historic", "suspension", "famous", "new", "steel", "modern", "pedestrian"],
            "nouns": ["shores", "districts", "neighborhoods", "regions", "banks", "communities"]
        },
        {
            "template": "The {adj} tower provides panoramic views of the {noun}",
            "adjectives": ["observation", "tall", "historic", "famous", "glass", "central", "viewing"],
            "nouns": ["city", "valley", "mountains", "landscape", "coastline", "harbor"]
        },
        
        # Urban environment
        {
            "template": "The {adj} skyline appears dramatic against the {noun} sky",
            "adjectives": ["city", "downtown", "urban", "modern", "impressive", "distinctive", "famous"],
            "nouns": ["evening", "morning", "sunset", "twilight", "night", "cloudy"]
        },
        {
            "template": "The city's {adj} district features architecture from various {noun}",
            "adjectives": ["historic", "cultural", "downtown", "central", "commercial", "residential", "artistic"],
            "nouns": ["periods", "centuries", "eras", "cultures", "traditions", "regions"]
        },
        {
            "template": "The {adj} café overlooks the central {noun}",
            "adjectives": ["popular", "cozy", "outdoor", "charming", "historic", "favorite", "local"],
            "nouns": ["plaza", "park", "square", "garden", "fountain", "boulevard"]
        },
        {
            "template": "The {adj} market sells fresh produce every {noun}",
            "adjectives": ["farmers", "local", "popular", "outdoor", "weekend", "community", "seasonal"],
            "nouns": ["morning", "weekend", "day", "Tuesday", "Saturday", "season"]
        },
        {
            "template": "Residents enjoy walking along the {adj} promenade in the {noun}",
            "adjectives": ["scenic", "waterfront", "popular", "historic", "coastal", "riverside", "tree-lined"],
            "nouns": ["evening", "afternoon", "summer", "spring", "morning", "weekend"]
        },
        
        # Educational
        {
            "template": "The {adj} university campus covers several {noun}",
            "adjectives": ["historic", "sprawling", "prestigious", "beautiful", "modern", "central", "main"],
            "nouns": ["acres", "blocks", "hectares", "buildings", "districts", "hills"]
        },
        {
            "template": "Researchers study {adj} ecosystems in the {noun} preserve",
            "adjectives": ["diverse", "fragile", "natural", "local", "unique", "rare", "coastal"],
            "nouns": ["national", "forest", "marine", "wildlife", "mountain", "wetland"]
        },
        {
            "template": "The {adj} library houses collections from many {noun}",
            "adjectives": ["central", "public", "university", "historic", "national", "research", "digital"],
            "nouns": ["periods", "centuries", "cultures", "countries", "continents", "eras"]
        },
        {
            "template": "Students gather in the {adj} courtyard between {noun}",
            "adjectives": ["central", "campus", "sunny", "shaded", "main", "historic", "university"],
            "nouns": ["classes", "lectures", "buildings", "sessions", "semesters", "terms"]
        },
        {
            "template": "Visitors learn about {adj} history at the {noun} center",
            "adjectives": ["local", "national", "regional", "indigenous", "cultural", "military", "industrial"],
            "nouns": ["cultural", "heritage", "community", "historical", "visitor", "education"]
        },
        
        # Transportation
        {
            "template": "The {adj} train travels through {noun} countryside",
            "adjectives": ["passenger", "scenic", "historic", "steam", "local", "express", "tourist"],
            "nouns": ["scenic", "rolling", "mountainous", "picturesque", "rural", "agricultural"]
        },
        {
            "template": "Passengers board the {adj} ferry for the {noun} crossing",
            "adjectives": ["passenger", "morning", "afternoon", "daily", "local", "island", "car"],
            "nouns": ["island", "harbor", "river", "lake", "bay", "channel"]
        },
        {
            "template": "The {adj} trail accommodates hikers of all {noun}",
            "adjectives": ["marked", "popular", "scenic", "nature", "forest", "mountain", "hiking"],
            "nouns": ["abilities", "ages", "levels", "skills", "experiences", "backgrounds"]
        },
        {
            "template": "The {adj} harbor shelters boats during {noun} weather",
            "adjectives": ["natural", "protected", "deep", "busy", "quiet", "fishing", "sailing"],
            "nouns": ["stormy", "winter", "rough", "severe", "inclement", "harsh"]
        },
        {
            "template": "The {adj} highway passes through {noun} areas",
            "adjectives": ["scenic", "coastal", "major", "national", "interstate", "main", "busy"],
            "nouns": ["scenic", "mountainous", "coastal", "forested", "rural", "desert"]
        },
        
        # Weather and landscapes
        {
            "template": "The {adj} peaks remain snow-covered until {noun}",
            "adjectives": ["mountain", "highest", "alpine", "distant", "northern", "rocky", "volcanic"],
            "nouns": ["spring", "summer", "June", "July", "late-season", "midsummer"]
        },
        {
            "template": "Autumn brings {adj} foliage to the {noun} forest",
            "adjectives": ["colorful", "brilliant", "vibrant", "spectacular", "golden", "red", "dramatic"],
            "nouns": ["national", "mountain", "state", "ancient", "protected", "deciduous"]
        },
        {
            "template": "{adj} evenings offer perfect weather for {noun} concerts",
            "adjectives": ["summer", "warm", "cool", "spring", "pleasant", "balmy", "autumn"],
            "nouns": ["outdoor", "summer", "garden", "park", "community", "festival"]
        },
        {
            "template": "The {adj} landscape transforms after {noun} rains",
            "adjectives": ["desert", "arid", "natural", "rural", "local", "entire", "barren"],
            "nouns": ["seasonal", "spring", "summer", "winter", "autumn", "monsoon"]
        },
        {
            "template": "{adj} flowers bloom throughout the {noun} gardens",
            "adjectives": ["colorful", "spring", "summer", "native", "exotic", "vibrant", "fragrant"],
            "nouns": ["botanical", "public", "community", "city", "famous", "national"]
        },
        
        # Cultural
        {
            "template": "The {adj} festival celebrates local {noun} and traditions",
            "adjectives": ["annual", "summer", "cultural", "traditional", "popular", "community", "regional"],
            "nouns": ["culture", "heritage", "history", "customs", "music", "cuisine"]
        },
        {
            "template": "Visitors enjoy the {adj} music during the {noun} festival",
            "adjectives": ["live", "traditional", "local", "folk", "popular", "classical", "jazz"],
            "nouns": ["annual", "summer", "cultural", "arts", "weekend", "music"]
        },
        {
            "template": "The {adj} theater presents performances throughout the {noun}",
            "adjectives": ["historic", "local", "community", "downtown", "famous", "renovated", "outdoor"],
            "nouns": ["season", "year", "summer", "weekend", "month", "festival"]
        },
        {
            "template": "Local artisans display {adj} crafts at the {noun} market",
            "adjectives": ["handmade", "traditional", "unique", "artistic", "cultural", "creative", "indigenous"],
            "nouns": ["weekly", "holiday", "summer", "community", "village", "farmers"]
        },
        {
            "template": "Traditional {adj} cuisine is served in {noun} restaurants",
            "adjectives": ["local", "regional", "authentic", "famous", "popular", "seasonal", "native"],
            "nouns": ["local", "downtown", "riverside", "historic", "family-owned", "popular"]
        },
        
        # Architecture
        {
            "template": "The {adj} building reflects the surrounding {noun}",
            "adjectives": ["glass", "modern", "mirrored", "contemporary", "sleek", "new", "innovative"],
            "nouns": ["landscape", "mountains", "water", "environment", "architecture", "skyline"]
        },
        {
            "template": "The {adj} statue commemorates the city {noun}",
            "adjectives": ["bronze", "marble", "famous", "historic", "central", "public", "memorial"],
            "nouns": ["founder", "history", "hero", "origins", "anniversary", "heritage"]
        },
        {
            "template": "The {adj} pathway leads through the {noun} district",
            "adjectives": ["cobblestone", "brick", "stone", "winding", "scenic", "historic", "pedestrian"],
            "nouns": ["historic", "shopping", "cultural", "business", "entertainment", "residential"]
        },
        {
            "template": "The {adj} bridge features innovative {noun} design",
            "adjectives": ["modern", "suspension", "famous", "iconic", "award-winning", "unique", "distinctive"],
            "nouns": ["architectural", "structural", "engineering", "modern", "suspension", "artistic"]
        },
        {
            "template": "The {adj} monument honors heroes from the {noun} era",
            "adjectives": ["national", "historic", "impressive", "marble", "granite", "famous", "central"],
            "nouns": ["colonial", "revolutionary", "modern", "industrial", "wartime", "historical"]
        }
    ]
    
    # Choose a random template option
    template_option = random.choice(template_options)
    
    # Extract the template and its specific word lists
    template = template_option["template"]
    template_adjectives = template_option["adjectives"]
    template_nouns = template_option["nouns"]
    
    # Select a random adjective and noun from the template-specific lists
    selected_adj = random.choice(template_adjectives)
    selected_noun = random.choice(template_nouns)
    
    # Fill in the template with selected words
    phrase = template.format(
        adj=selected_adj,
        noun=selected_noun
    )
    
    return phrase

if __name__ == "__main__":
    for _ in range(10):
        print(generate_challenge_phrase())
