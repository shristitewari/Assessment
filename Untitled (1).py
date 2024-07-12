#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


# In[118]:


dataset = pd.read_csv("data-analyst-test-data.csv")



# In[119]:


dataset.head()


# In[120]:


dataset['date'] = pd.to_datetime(dataset['date'], format='%Y/%m/%d', errors='coerce')


# In[121]:


positive_keywords = ['impressed', 'excellent', 'nice', 'friendly', 'clean', 'professional', 'spacious', 'comfortable', 'amazing', 'fantastic', 'good']
negative_keywords = ['poor', 'sink', 'tiny', 'small', 'ridiculous', 'dirty', 'rude', 'limited', 'smoking', 'very bad', 'horrible', 'not clean', 'smell', 'needs work', 'no free', 'useless', 'issues', 'bad', 'dirty', 'uncomfortable', 'complaint']


# In[122]:


positive_comments = dataset[dataset['Review'].str.contains('|'.join(positive_keywords), case=False, na=False)]


# In[123]:


negative_comments = dataset[dataset['Review'].str.contains('|'.join(negative_keywords), case=False, na=False)]


# In[124]:


top_positive_comments = positive_comments.sort_values(by='date', ascending=False).head(5)
top_negative_comments = negative_comments.sort_values(by='date', ascending=False).head(5)


# In[125]:


# Print the top positive comments
print("\nTop Positive Comments:")
for index, row in top_positive_comments.iterrows():
    date_str = row['date'].strftime('%Y/%m/%d')
    comment_text = row['Review'].split(',', 1)[-1]
    if date_str and comment_text.strip():
        print(f"Date: {date_str}, Comment: {comment_text.strip()}")


# In[126]:


# Print the top negative comments
print("\nTop Negative Comments:")
for index, row in top_negative_comments.iterrows():
    date_str = row['date'].strftime('%Y/%m/%d')
    comment_text = row['Review'].split(',', 1)[-1]
    if date_str and comment_text.strip():
        print(f"Date: {date_str}, Comment: {comment_text.strip()}")


# In[127]:


labeled_data = {
    'Review': [
        'Great service and clean rooms.',
        'The staff was rude and the room was dirty.',
        'The location is good but the service needs work.',
        'Amazing experience, very comfortable stay.',
        'Horrible experience, the place was not clean at all.'
    ],
    'Actual Sentiment': [
        'Positive',
        'Negative',
        'Neutral',
        'Positive',
        'Negative'
    ]
}


# In[128]:


labeled_df = pd.DataFrame(labeled_data)


# In[129]:


labeled_df['Review'] = labeled_df['Review'].str.lower()


# In[130]:


def categorize_sentiment(review):
    if any(keyword in review for keyword in positive_keywords):
        return 'Positive'
    elif any(keyword in review for keyword in negative_keywords):
        return 'Negative'
    else:
        return 'Neutral'


# In[131]:


labeled_df['Predicted Sentiment'] = labeled_df['Review'].apply(categorize_sentiment)


# In[132]:


accuracy = accuracy_score(labeled_df['Actual Sentiment'], labeled_df['Predicted Sentiment'])
print(f"\nAccuracy: {accuracy}")


# In[133]:


print("\nClassification Report:")
print(classification_report(labeled_df['Actual Sentiment'], labeled_df['Predicted Sentiment'], zero_division=0))


# In[ ]:




