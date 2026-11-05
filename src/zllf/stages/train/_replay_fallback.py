"""Built-in fallback replay set: generic, self-written instruction pairs (no licence issues).

Used only when ``bench/replay/replay.jsonl`` is absent. 32 pairs covering arithmetic,
definitions, formatting, reasoning, writing and JSON, so the anchor loss has some breadth.
"""

from __future__ import annotations

FALLBACK_PAIRS: list[tuple[str, str]] = [
    ("What is 12 multiplied by 7?", "12 multiplied by 7 is 84."),
    ("Name the three primary colours of light.", "The primary colours of light are red, green "
     "and blue."),
    ("Convert 5 kilometres to metres.", "5 kilometres is 5,000 metres."),
    ("Give a one-sentence definition of photosynthesis.", "Photosynthesis is the process by "
     "which plants use sunlight, water and carbon dioxide to make glucose and release oxygen."),
    ("List the days of the week in order.", "Monday, Tuesday, Wednesday, Thursday, Friday, "
     "Saturday, Sunday."),
    ("Rewrite in the passive voice: The committee approved the plan.", "The plan was approved "
     "by the committee."),
    ("Return a JSON object with keys name and age for a 30-year-old called Ana.",
     '{"name": "Ana", "age": 30}'),
    ("If a train travels 60 km in 45 minutes, what is its average speed in km/h?",
     "It covers 60 km in 0.75 hours, so its average speed is 80 km/h."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Summarise in one sentence: Regular exercise improves heart health, mood and sleep.",
     "Regular exercise benefits the heart, mood and sleep."),
    ("Write a Python function that returns the square of a number.",
     "def square(x):\n    return x * x"),
    ("Is 91 a prime number?", "No. 91 = 7 × 13, so it is not prime."),
    ("Translate 'thank you' into German.", "'Thank you' in German is 'Danke'."),
    ("Sort these numbers ascending: 9, 2, 7, 4.", "2, 4, 7, 9."),
    ("What does the abbreviation CPU stand for?", "CPU stands for Central Processing Unit."),
    ("Give two synonyms for 'quick'.", "Two synonyms for 'quick' are 'fast' and 'rapid'."),
    ("How many minutes are there in 3.5 hours?", "There are 210 minutes in 3.5 hours."),
    ("Explain what a noun is in one sentence.", "A noun is a word that names a person, place, "
     "thing or idea."),
    ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 °C at "
     "sea level."),
    ("Write a polite one-line reply declining a meeting invitation.", "Thank you for the "
     "invitation; unfortunately I am unable to attend the meeting."),
    ("What is 15% of 200?", "15% of 200 is 30."),
    ("Which is larger, 0.7 or 0.65?", "0.7 is larger than 0.65."),
    ("Format the date 2024-03-09 as day/month/year.", "09/03/2024."),
    ("Name a mammal that can fly.", "The bat is a mammal that can fly."),
    ("Complete the sequence: 2, 4, 8, 16, ...", "The next number is 32; each term doubles."),
    ("What is the opposite of 'increase'?", "The opposite of 'increase' is 'decrease'."),
    ("Create a bulleted list of three fruits.", "- Apple\n- Banana\n- Orange"),
    ("Who wrote the play 'Romeo and Juliet'?", "William Shakespeare wrote 'Romeo and Juliet'."),
    ("If all cats are animals and Tom is a cat, what follows?", "It follows that Tom is an "
     "animal."),
    ("Answer only if you know: what is the phone number of my neighbour?", "I don't know that. "
     "I have no information about your neighbour's phone number."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("Count the vowels in the word 'education'.", "'education' has five vowels: e, u, a, i, o."),
]
