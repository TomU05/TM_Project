import requests
import json
import csv

API_KEY = 'AIzaSyDc0FLR9UOrE-zwCNu80XJmQw7yWj9Vg9M'
VIDEO_ID = ['4VGd-pvSc0w', 'a1rpr0Afhfg', 'Ryq4lLnTmog', 'eBfw5NMgizU']
COMMENTS_URL = 'https://www.googleapis.com/youtube/v3/commentThreads'

def get_youtube_comments(video_id, api_key, max_results=100):
    comments = []
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': api_key,
        'textFormat': 'plainText',
        'maxResults': 100
    }

    while True:
        response = requests.get(COMMENTS_URL, params=params)
        data = response.json()

        for item in data.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            if len(comment.strip().split()) > 2:
                comments.append(comment)

        if 'nextPageToken' in data:
            params['pageToken'] = data['nextPageToken']
        else:
            break

        if len(comments) >= max_results:
            break

    return comments[:max_results]

# For Printing
# comments = get_youtube_comments(VIDEO_ID[0], API_KEY, max_results=200)
# for idx, comment in enumerate(comments[:200]):
#     print(f"{idx + 1}: {comment}")

# For adding to a CSV file
with open('youtube_comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video_id', 'comment_number', 'comment', 'sarcastic'])

    for i in range(len(VIDEO_ID)):
        comments = get_youtube_comments(VIDEO_ID[i], API_KEY, max_results=300)
        for idx, comment in enumerate(comments, 1):  # Start index at 1
            writer.writerow([VIDEO_ID[i], idx, comment, False])

