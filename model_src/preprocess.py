def preprocess_text(df):
    df['text'] = df['text'].apply(lambda x: x.replace('"', ''))