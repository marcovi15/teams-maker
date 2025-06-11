# Functions breakdown

This software architecture is fairly convoluted despite the simplicity of the application.
This is to address the issue of the HTML request expiring if it doesn't receive a response within 30 seconds.
The whole script can take up to one minute to run on AWS, so this was often the case, and resulted in a confusing error message, even if the script actually ran correctly.
The workaround involves three functions described below, which essentially provide a loading screen immediately while the background script runs. 
When this is done, the loading screen is replaced by the final HTML page with the teams list.

## footyApp

This is frontend facing, and is the one triggered by the link API. 
When invoked, it calls the background function `footyAppBackground` and shows a loading HTML page. 
Embedded in the html is a script that calls the status function `footyAppStatus` every 3 seconds.
This returns nothing until the teams are ready, at which point it returns a new HTML with all the teams, which will replace the loading screen.

## footyAppStatus

This tries to extract a json from the S3 database, which will be empty until `footyAppBackground` finishes running.
While empty, the extraction will return an error, in which case a `pending` status will be returned.
When the extraction is finally successful, `footyAppStatus` will return a dictionary with a `done` status and the final HTML in the `body` key.
The status change is what triggers `footyApp` to display the final HTML instead of the loading screen.

## footyAppBackground

This is where the logic that actually makes the teams lives. 
When triggered, it will fetch the participants list and their attributes from the Google Sheet documents and divide them into 2, 3, and 4 teams.
The logic is basically a linear programming optimisation that simulates the "captains pick" method: players are divided into groups based on their roles, each group is sorted by skill level, and then they are assigned to each team is such order. The pick order is switched at every round to make it fairer.

When the teams are done, they are formatted into an HTML page and save into an S3 database, from which `footyAppStatus` will retrieve it.


# Abbreviations in database

## Roles
D = Defender

M = Midfielder

F = Forward

## Tiers

Tiers indicate a player's skill and go from A to E:

A = Amazing player that makes a big difference

B = Above average player

C = Average player

D = Below average player, still contributes to the team

E = Unreliable player that may make the team worse
