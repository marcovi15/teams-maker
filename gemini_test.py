from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

message = ("Split these players into two football teams, based on the coaches preferences and players description. Teams should have no more than 1 player more than the other and be balanced in terms of skill level.\n"
           "Team Italy prioritises solid defence and counter-attack, Team Spain aims to win by scoring more goals than it concedes\n"
           "Use all of the following players without repetitions, it's not a regular 11-a-side:\n"
           "Will: Slow, average centre-back\n"
           "George: Tall and quick box-to-box midfielder\n"
           "Fernando: Slow midfielder, good at defending and keeping position\n"
           "Ricky: Physically strong and quick full-back with poor technique and lots of stamina and grit\n"
           "Amirth: Quick full-back with decent passing and control\n"
           "Bhav: Quick winger who loves dribbling and running down the flank\n"
           "Peter: Very physical centre back\n"
           "Vin: Skillful attacking midfielder who loves giving through balls\n"
           "Joe: Slow midfielder with por stamina\n"
           "Terry: Full-back who's mostly good at defending but can also attack and likes short passes\n"
           "Jake: Classic number 6 who's great at distributing balls and hold the midfield together\n"
           "Lee: Very good attacking midfielder who loves short, accurate passes and gets annoyed if balls are wasted\n"
           "Paul: Quick winger who likes using speed for dribbling but doesn't track back\n"
           "Abel: Good pivot forward who can hold the ball and score with headers thanks to impressive frame\n"
           "Alex: Solid centre back who's great at directing the team and making them play more compact\n"
           "Wilson: Similar to Terry\n"
           "Mart: Quick and skillful winger who tries the dribble even when he shouldn't, sometimes leading to lost balls\n"
           "Marco: Full-back with lots of grit and stamina but struggles with high balls and much prefers short and easy passing.")

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=message
)
# for model in client.models.list():
#     print(model.name)
print(response.text)
