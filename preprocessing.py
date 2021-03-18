import pandas as pd 
import numpy as np 
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

#A lot of ugly preprocessing
#An issue here is that we include linearly dependent features
#These should be removed.


teams = teams.get_teams()
abbrevs = [x['abbreviation'] for x in teams]
games = leaguegamefinder.LeagueGameFinder().get_data_frames()[0]
training_data = games[['GAME_DATE','GAME_ID','MATCHUP','WL']]
whole = training_data.copy()
for abbrev in abbrevs:
    training_data = training_data[~training_data['MATCHUP'].str.contains(abbrev)]

train_set = whole[~whole['MATCHUP'].isin(training_data['MATCHUP'])]

#Create a dictionary from abbrevitation to game history
abbrev2df = {}
for abbrev in abbrevs:
    team_id = [x['id'] for x in teams if x['abbreviation'] == abbrev][0]
    games = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id).get_data_frames()[0]
    games = games.set_index('GAME_DATE')
    id_columns = games['GAME_ID'].sort_index()
    games = games[['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PLUS_MINUS']].sort_index().shift(1)
    #The rolling mean is used for merged.pkl
    #games = games.rolling(10).mean()
    #Use exponentially weighted averages for the input data
    #We want most recent game data to matter most.
    games = games.ewm(com=0.5).mean()
    games['GAME_ID'] = id_columns
    games = games.dropna().set_index('GAME_ID')
    games = games[~games.index.duplicated(keep='first')]
    abbrev2df[abbrev] = games

train_set = train_set[['GAME_ID','WL','MATCHUP']]
train_set.set_index('GAME_ID', inplace = True)

home_data = pd.DataFrame(index=train_set.index.drop_duplicates(keep='first'), columns = ['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PLUS_MINUS'])
away_data = pd.DataFrame(index=train_set.index.drop_duplicates(keep='first'), columns = ['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PLUS_MINUS'])

count = 0
for index, row in train_set.iterrows():
    count += 1
    home_name = row['MATCHUP'][0:3]
    away_name = row['MATCHUP'][-3:]
    if home_name in abbrevs and away_name in abbrevs:
        home_df = abbrev2df[home_name]
        away_df = abbrev2df[away_name]
        if index in home_df.index and index in away_df.index:
            home_data.loc[index] = home_df.loc[index]
            away_data.loc[index] = away_df.loc[index]

#Merge train_set with home_data
train_set = train_set.groupby(train_set.index).first()
train_set.dropna(inplace=True)
first_merge = pd.merge(train_set.dropna(), home_data.dropna(), how='inner',left_index=True,right_index=True)

final_merge = pd.merge(first_merge, away_data, how='inner', left_index=True,right_index=True).dropna()


final_merge.to_pickle("./merged_5_day.pkl")













