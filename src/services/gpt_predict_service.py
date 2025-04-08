from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for GPT
gpt_prompt = (
    """Classify each tweet as:
1 = Real disaster or emergency
0 = Not a real disaster (metaphorical or unrelated)
Respond 1 if:
- The tweet might describe a real event (emergency, disaster, accident, or harm)
- It includes signs of physical danger: injured, missing, dead, collapsed, exploded, destroyed, evacuated, emergency
- Mentions of location, time, or hashtags (#breaking, #fire, etc.)
- Vague but serious phrases like “terrible event downtown” or “people running everywhere”
- Tone or grammar is casual or unclear—assume real unless clearly metaphorical
Respond 0 if:
- It’s figurative: “dying of boredom”, “this exam is a disaster”, “heart exploded”
- Uses expressive emojis (😭🔥💀)
- Is clearly about slang, memes, or entertainment
Special Rule:
- When in doubt, respond 1. Better to misclassify metaphor than miss a real disaster.
Examples:
"Explosion rocks downtown, many feared dead" → 1
"This dress is fire 🔥🔥" → 0
"Everyone running after loud boom in city center" → 1
"Ugh, Mondays are a disaster 😩" → 0
"3 people missing after boat capsized in Mississippi" → 1
"This playlist is killing me 😭😭😭" → 0
Respond ONLY with 1 or 0."""
)

def predict_gpt(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": gpt_prompt},
            {"role": "user", "content": text}
        ]
    )
    return int(response.choices[0].message.content.strip())


if __name__ == "__main__":
    test_text = "Explosion in downtown area, multiple injuries reported."
    prediction = predict_gpt(test_text)
    print("Prediction:", prediction)
