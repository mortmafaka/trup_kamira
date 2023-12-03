import numpy as np
from colorama import Fore, Style, init, deinit
from joblib import load
from src.Utils import Expected_Value, Kelly_Criterion as kc

# Initialize Colorama
init()

# Load the trained logistic regression models
lr_ml = load('Models/LR_Models/Logistic_Regression_ML_model.joblib')
lr_uo = load('Models/LR_Models/Logistic_Regression_UO_model.joblib')

def lr_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    # Predict the Moneyline (ML) outcomes
    ml_predictions_proba = lr_ml.predict_proba(data)

    # Add the Over/Under (OU) data to the dataframe for prediction
    frame_uo = np.copy(data)
    frame_uo = np.column_stack((frame_uo, todays_games_uo))
    ou_predictions_proba = lr_uo.predict_proba(frame_uo)

    # Iterate over the games to print predictions, expected value, and Kelly Criterion
    for count, game in enumerate(games):
        home_team = game[0]
        away_team = game[1]
        home_prob = ml_predictions_proba[count][1]  # Probability of home team winning
        ou_prob = ou_predictions_proba[count][1]  # Probability of the over

        # Print the predictions
        print(f"{Fore.GREEN + home_team + Style.RESET_ALL} vs {Fore.RED + away_team + Style.RESET_ALL}")
        print(f"Home Win Probability: {Fore.CYAN}{home_prob * 100:.1f}%{Style.RESET_ALL}")
        print(f"Over Probability: {Fore.CYAN}{ou_prob * 100:.1f}%{Style.RESET_ALL}")

        # Expected value calculations
        ev_home = Expected_Value.expected_value(home_prob, home_team_odds[count])
        ev_away = Expected_Value.expected_value(1 - home_prob, away_team_odds[count])
        print(f"{home_team} EV: {Fore.GREEN if ev_home > 0 else Fore.RED}{ev_home}{Style.RESET_ALL}")
        print(f"{away_team} EV: {Fore.GREEN if ev_away > 0 else Fore.RED}{ev_away}{Style.RESET_ALL}")

        # Kelly Criterion calculation
        if kelly_criterion:
            kelly_home = kc.calculate_kelly_criterion(home_team_odds[count], home_prob)
            kelly_away = kc.calculate_kelly_criterion(away_team_odds[count], 1 - home_prob)
            print(f"{home_team} Kelly Criterion: {Fore.YELLOW}{kelly_home:.2f}%{Style.RESET_ALL}")
            print(f"{away_team} Kelly Criterion: {Fore.YELLOW}{kelly_away:.2f}%{Style.RESET_ALL}")

    # Deinitialize Colorama
    deinit()
