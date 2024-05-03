from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_channel_transcript_api import YoutubeChannelTranscripts
from pydantic import BaseModel
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from config import (
    COLLECTION_NAME,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    YOUTUBE_API_KEY
)
from httpx import Timeout
from datetime import datetime
import re
import requests
import json
import httplib2
import time
import os
import spacy
import numpy as np
import hashlib
import ast
import openai
import torch

app = FastAPI()
directory_path = os.path.join(os.path.dirname(__file__), 'transcipts')
nlp = spacy.load('en_core_web_md')
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
http = httplib2.Http(timeout=60)
qdrant_client = QdrantClient(host='localhost', port=6333)
torch.autograd.set_detect_anomaly(True)
vectors = np.random.rand(384, 384)

yt_api_url = "https://www.googleapis.com/youtube/v3/"
searchurl = 'http://localhost:6333/collections/youtube_videos/points/search'
url = f"https://{QDRANT_HOST}:{QDRANT_PORT}/collections/youtube_videos/points?wait=true"

openai.api_key = OPENAI_API_KEY
yt_api_key = YOUTUBE_API_KEY

youtube = build('youtube', 'v3', developerKey=yt_api_key, http=http)

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    timeout=60
)

model = SentenceTransformer('msmarco-MiniLM-L-6-v3')

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant_client.recreate_collection(
    collection_name='youtube_videos',
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

class Item(BaseModel):
    links: str

class SearchItem(BaseModel):
    search: str

def get_video_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e: 
        print(f"Failed to fetch data: {e}")
        return None

def extract_video_details(video_data):
    if not video_data or 'items' not in video_data or not video_data['items']:
        return None 

    item = video_data['items'][0]
    details = {
        "video_id": item['id'],
        "title": item['snippet']['title'],
        "channel_title": item['snippet'].get('channelTitle', 'Unknown Channel'),
        "description": item['snippet']['description'],
        "published_at": item['snippet']['publishedAt'],
        "like_count": item['statistics'].get('likeCount', '0'),
        "view_count": item['statistics'].get('viewCount', '0'),
        "comment_count": item['statistics'].get('commentCount', '0'),
        "favorite_count": item['statistics'].get('favoriteCount', '0')
    }
    return details

def get_video_id(link):
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, link)
    return match.group(1) if match else None

def get_channel_name(link):
   channel_name = link.split('@')[-1]
   return channel_name

def get_channel_id_by_name(youtube, channel_name):
    try:
        request = youtube.search().list(
            part="snippet",
            type="channel",
            q=channel_name,
            maxResults=1
        )
        response = request.execute()
        if response['items']:
            return response['items'][0]['id']['channelId']
        else:
            return None
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_video_ids(youtube, channel_id):
    video_ids = []
    next_page_token = None
    try:
        while True:
            request = youtube.search().list(
                part="id",
                channelId=channel_id,
                maxResults=50,
                pageToken=next_page_token,
                type="video"
            )
            response = request.execute()

            video_ids += [item['id']['videoId'] for item in response['items']]

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return video_ids

def get_video_title(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        response = requests.get(video_url)
        response.raise_for_status()
        matches = re.findall(r'<title>(.*?)</title>', response.text)
        return matches[0].replace(" - YouTube", "") if matches else "Unknown"
    except requests.RequestException as e:
        print(f"Error fetching video title: {e}")
        return "Unknown"

def get_transcript(video_id):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_generated_transcript(['en'])
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript.fetch())
            return transcript_text
        except ConnectionResetError as e:
            print(f"Connection reset by peer, attempt {attempt + 1} of {max_retries}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Error transcript: {e}")
            return ""
    return ""

def string_to_int_id(video_id):
    hash_object = hashlib.sha256(video_id.encode('utf-8'))
    return int(hash_object.hexdigest(), 16) % (1 << 64)

def save_to_qdrant(video_id, title, transcript, vector, details):
    try:
        if isinstance(vector, str):
            vector = parse_vector(vector)
        if not all(isinstance(n, float) for n in vector):
            raise ValueError("All elements in vector must be floats")
    except ValueError as e:
        print(f"Error in vector data: {e}")
        raise
    videos_id = string_to_int_id(video_id)
    date_obj = datetime.strptime(details["published_at"].replace('Z', ''), "%Y-%m-%dT%H:%M:%S")
    published_at = date_obj.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "title": title,
        "channel_title": details["channel_title"],
        "transcript": transcript,
        "description": details["description"],
        "published_at": published_at,
        "comment_count": details["comment_count"],
        "favorite_count": details["favorite_count"],
        "like_count": details["like_count"],
        "view_count": details["view_count"]
    }
    data={
        "ids":[videos_id],
        "points":[
            {
                "id": videos_id,
                "vector": vector,
                "payload": payload
            }
        ]
    }
    json_data = json.dumps(data)
    headers = {
        'Authorization': 'Bearer LFl3MkX_MV07GV_BFz4y_gQQsSMZr46Gt1eEOOMcWVuDQJKMelc52A',
        'Content-Type': 'application/json'
        }
    print(data)
    response = requests.put(url, headers=headers, data=json_data)
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)

    if response.status_code == 200:
        try:
            json_response = response.json()
            print("JSON Response:", json_response)
        except json.JSONDecodeError:
            print("Failed to decode JSON, check the raw response for details.")
    else:
        print("Request failed with status:", response.status_code)
        print("Response was:", response.text)

def adjust_vector_dimensions(text, target_dim=384, elements_per_line=4):
    vector= model.encode([text])[0]
    if len(vector) > target_dim:
        vector = vector[:target_dim]
    elif len(vector) < target_dim:
        vector = np.pad(vector, (0, target_dim - len(vector)), 'constant')    
    return vector.tolist()

def parse_vector(vector_str):
    try:
        vector = ast.literal_eval(vector_str)
        if isinstance(vector, list) and all(isinstance(x, float) for x in vector):
            return vector
        else:
            raise ValueError("Parsed object is not a list of floats")
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing vector string: {e}")
        raise ValueError("Vector must be a string representation of a list of floats")
    
def build_prompt(question: str, references: list) -> tuple[str, str]:
    prompt = f"""
    You're Marcus Aurelius, emperor of Rome. You're giving advice to a friend who has asked you the following question: '{question}'
    You've selected the most relevant passages from your writings to use as source for your answer. Cite them in your answer.
    References:""".strip()
    references_text = ""
    for i, reference in enumerate(references, start=1):
        text = reference.payload["transcript"].strip()
        references_text += f"\n[{i}]: {text}"
    prompt += (
        references_text
        + "\nHow to cite a reference: This is a citation [1]. This one too [3]. And this is sentence with many citations [2][3].\nAnswer:"
    )
    return prompt, references_text

@app.post("/search")
async def query_from_qdrant(item: SearchItem):
    similar_docs = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=model.encode(item.search),
        limit=3,
        append_payload=True,
    )
    prompt, references = build_prompt(item.search, similar_docs)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.2,
    )
    return response

@app.post("/")
async def root(item: Item):
    links = item.links.split(",")
    api_links = []
    video_datas = []
    transcripts =[]
    if "https://www.youtube.com/@" in item.links:
        for link in links:
            channel_name = get_channel_name(link.strip())
            channel_id = get_channel_id_by_name(youtube, channel_name)
            video_ids = get_video_ids(youtube, channel_id)
            video_datas = [get_video_data(f"{yt_api_url}videos?id={video_id}&key={yt_api_key}&part=snippet,contentDetails,statistics,status") for video_id in video_ids]
            channel_getter = YoutubeChannelTranscripts(channel_name, yt_api_key)
            videos_data = channel_getter.get_transcripts()
            for video_id in video_ids:
                if video_id:
                    transcript_text=get_transcript(video_id)
                    if transcript_text:
                        video_title = get_video_title(video_id)
                        file_name = f"{video_id}_{video_title}.txt"
                        file_name = re.sub(r'[\\/*?:"<>|]', '', file_name) 
                        full_file_path = os.path.join(directory_path, file_name)
                        with open(file_name, 'w', encoding='utf-8') as file:
                            file.write(transcript_text)
                        transcripts += [transcript_text]
                        vector = adjust_vector_dimensions(transcript_text) 
                        print(vector)
                        print(f"Transcript saved to {file_name}")
                        return {"message": "success", "video_datas": videos_data, "transcripts":transcripts}
                    else:
                        print("Unable to download transcript.")
                        return {"message": "Unable to download transcript."}
                else:
                    print("Invalid YouTube URL.")
                    return {"message": "Invalid YouTube URL."}

    else:
        for link in links:
            video_id = get_video_id(link.strip())
            api_url = f"{yt_api_url}videos?id={video_id}&key={yt_api_key}&part=snippet,contentDetails,statistics,status"
            api_links+=[api_url]
            video_datas += [get_video_data(api_url)]
            if video_id:
                transcript_text=get_transcript(video_id)
                if transcript_text:
                    video_title = get_video_title(video_id)
                    file_name = f"{video_id}_{video_title}.txt"
                    file_name = re.sub(r'[\\/*?:"<>|]', '', file_name)
                    full_file_path = os.path.join(directory_path, file_name)
                    with open(full_file_path, 'w', encoding='utf-8') as file:
                        file.write(transcript_text)
                    transcripts += [transcript_text]
                    print(f"Transcript saved to {file_name}")
                    vector = adjust_vector_dimensions(transcript_text) 
                    video_details = extract_video_details(get_video_data(api_url))
                    save_to_qdrant(video_id, video_title, transcript_text, vector, video_details)
                    return {"message": "success", "URL" : api_links, "Data": video_datas, "transcripts":transcripts}
                else:
                    print("Unable to download transcript.")
                    return {"message": "Unable to download transcript."}
            else:
                print("Invalid YouTube URL.")
                return {"message": "Invalid YouTube URL."}