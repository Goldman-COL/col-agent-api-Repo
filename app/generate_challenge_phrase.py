import random

def generate_challenge_phrase() -> str:
    template_options = [
        # Nature
        {
            "template": "The {adj} park is nice in the {noun}",
            "adjectives": ["green", "big", "pretty", "quiet", "fun"],
            "nouns": ["morning", "evening", "summer", "spring"]
        },
        {
            "template": "The {adj} beach is busy in {noun}",
            "adjectives": ["sandy", "warm", "clean", "long"],
            "nouns": ["summer", "weekend", "holiday"]
        },
        {
            "template": "A {adj} tree grows in the {noun}",
            "adjectives": ["tall", "old", "green", "strong"],
            "nouns": ["park", "forest", "yard"]
        },

        # Landmarks
        {
            "template": "The {adj} bridge crosses the {noun}",
            "adjectives": ["old", "new", "big", "strong"],
            "nouns": ["river", "lake", "bay"]
        },
        {
            "template": "The {adj} tower is in the {noun}",
            "adjectives": ["tall", "old", "shiny", "famous"],
            "nouns": ["city", "town", "park"]
        },

        # Urban environment
        {
            "template": "The {adj} street is full of {noun}",
            "adjectives": ["busy", "quiet", "wide", "clean"],
            "nouns": ["shops", "people", "cars"]
        },
        {
            "template": "The {adj} market sells {noun}",
            "adjectives": ["small", "big", "fresh", "local"],
            "nouns": ["fruit", "flowers", "food"]
        },

        # Weather and landscapes
        {
            "template": "The {adj} sky is clear at {noun}",
            "adjectives": ["blue", "bright", "calm"],
            "nouns": ["night", "dawn", "sunset"]
        },
        {
            "template": "{adj} flowers bloom in the {noun}",
            "adjectives": ["red", "yellow", "pink", "bright"],
            "nouns": ["garden", "park", "field"]
        },

        # Cultural
        {
            "template": "The {adj} festival has fun {noun}",
            "adjectives": ["big", "happy", "loud", "colorful"],
            "nouns": ["music", "food", "games"]
        }
    ]
    
    # Choose a random template option
    template_option = random.choice(template_options)
    
    # Extract the template and its specific word lists
    template = template_option["template"]
    template_adjectives = template_option["adjectives"]
    template_nouns = template_option["nouns"]
    
    # Select a random adjective and noun
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