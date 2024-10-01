# data_preparation.py

import pandas as pd
import numpy as np

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset by handling missing values and ensuring consistent formats."""
    # Example of handling missing values
    df.dropna(inplace=True)
    
    # Example of converting date columns if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def merge_data(locations_df, keystrokes_df, mouse_df):
    """Merge multiple datasets on 'user_id'."""
    combined_df = locations_df.merge(keystrokes_df, on='user_id', how='inner')
    combined_df = combined_df.merge(mouse_df, on='user_id', how='inner')
    return combined_df

def aggregate_features(df):
    """Aggregate features and create user profiles."""
    # Example of aggregating keystrokes data
    keystroke_summary = df.groupby('user_id').agg({
        'typing_speed': 'mean',
        'typing_rhythm': 'mean'
    }).reset_index()
    
    # Merge aggregated data back to the main DataFrame
    combined_df = df.merge(keystroke_summary, on='user_id', how='inner')
    
    # Example of creating additional features
    combined_df['average_typing_speed'] = combined_df['typing_speed']
    combined_df['most_frequent_location'] = combined_df.groupby('user_id')['location'].apply(lambda x: x.mode()[0])
    
    return combined_df

def save_data(df, output_file):
    """Save the final dataset to a CSV file."""
    df.to_csv(output_file, index=False)

def main():
    # Load datasets
    locations_df = load_data('user_locations.csv')
    keystrokes_df = load_data('keystrokes.csv')
    mouse_df = load_data('mouse_dynamics.csv')
    
    # Clean datasets
    locations_df = clean_data(locations_df)
    keystrokes_df = clean_data(keystrokes_df)
    mouse_df = clean_data(mouse_df)
    
    # Merge datasets
    combined_df = merge_data(locations_df, keystrokes_df, mouse_df)
    
    # Aggregate features and create user profiles
    user_profiles_df = aggregate_features(combined_df)
    
    # Save the final dataset
    save_data(user_profiles_df, 'user_profiles.csv')


main()
