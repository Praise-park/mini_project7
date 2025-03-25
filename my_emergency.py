
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor


# 0. load key file------------------
def load_key(path):
    def load_file(filepath):
        with open(filepath, 'r') as file:
            return file.readline().strip()

    # API 키 로드 및 환경변수 설정
    openai.api_key = load_file(path + 'api_key.txt')
    os.environ['OPENAI_API_KEY'] = openai.api_key

# 1-1 audio2text--------------------

def audio2text(audio_path):
    client = OpenAI()
    supported_formats = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}

    # 지원되는 파일 형식만 필터링
    filenames = [
        f for f in os.listdir(audio_path) 
        if os.path.isfile(os.path.join(audio_path, f)) and os.path.splitext(f)[1].lower() in supported_formats
    ]

    # 결과를 저장할 리스트
    results = []
    for filename in filenames:
        audio_file_path = os.path.join(audio_path, filename)
        with open(audio_file_path, "rb") as audio_file:
            # OpenAI API로 오디오 파일 처리
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="ko",
                response_format="text",
            )
            results.append({"Audio": filename, "Transcribed": transcript})

    # 데이터프레임 생성
    df = pd.DataFrame(results)
    return df


# 1-2 text2summary------------------
def text2summary(path, df):
    client = OpenAI()
    # 시스템 역할과 응답 형식 지정
    system_role = '''
    당신은 응급상황에서 어떠한 상황인지 파악하고, 증상을 간단한 키워드로 요약하는 어시스턴트입니다. 응답은 다음의 형식을 지켜주세요
     {\"summary\": \"텍스트 요약\", \"부위\": \"다친 부위\", \"증상\": \"관련 증상\"}
    '''

    def summarize_text(input_text):
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_role
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )
        return response.choices[0].message.content

    # Transcribed 열을 요약하여 summary 열 생성
    df['summary'] = df['Transcribed'].apply(summarize_text)
    return df

# 2. model prediction------------------
def model_prediction(path, df):
    save_directory = path + "fine_tuned_bert"

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    text = df['summary'][0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value for key, value in inputs.items()}  # 각 텐서를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
         outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()

    return pred, probabilities

# 3-1. get_distance------------------

def get_dist(path, df):

        hospital = pd.read_csv(path + '/응급실 정보.csv')
        x_lon, x_lat = df['위도'][0], df['경도'][0]
        lat_em_filter, lon_em_filter = hospital['위도'], hospital['경도']
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": 'hqev5yhpp0',
            "X-NCP-APIGW-API-KEY": '0hKV6LdcBYambAIlHkVtAMuxCuRD6ypCVQrSGjne',
        }
        params = {
            "start": f"{x_lon}, {x_lat}",  # 출발지 (경도, 위도)
            "goal": f"{lon_em_filter}, {lat_em_filter}",  # 목적지 (경도, 위도)
            "option": "trafast"  # 실시간 빠른 길 옵션
        }
        response = requests.get(url, headers=headers, params=params).json()
        dist = response['route']['trafast'][0]['summary']['distance']  # 거리 (미터)
        return (dist, hospital['병원이름'], hospital['전화번호 3'], hospital['전화번호 1'])
        # 전화번호 3가 응급의료시설번호고, 전화번호 1이 병원번호


# 3-2. recommendation------------------

def recom_em(path, df):
    info_em = pd.read_csv(path + '/응급실 정보.csv')
    x_0, x_1 = df['위도'][0], df['경도'][0]
    x = (x_0, x_1)
    x_lat, x_lon = x[0], x[1]
    a = 0.05  # 초기 검색 범위
    filtered_hospitals = None

    # 최소 3개의 병원이 필터링될 때까지 검색 범위 확장
    while filtered_hospitals is None or len(filtered_hospitals) < 3:
        x_lat_min, x_lon_min = x_lat - a, x_lon - a
        x_lat_max, x_lon_max = x_lat + a, x_lon + a

        # 위도와 경도 조건으로 병원 필터링
        lat_em, lon_em = info_em['위도'], info_em['경도']
        filtered_hospitals = info_em[
            (lat_em >= x_lat_min) & (lat_em <= x_lat_max) &
            (lon_em >= x_lon_min) & (lon_em <= x_lon_max)
        ]

        # 범위 확장
        a += 0.05

    # API 호출 함수
    def get_dist(hospital):
        lat_em_filter, lon_em_filter = hospital['위도'], hospital['경도']
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": 'hqev5yhpp0',
            "X-NCP-APIGW-API-KEY": '0hKV6LdcBYambAIlHkVtAMuxCuRD6ypCVQrSGjne',
        }
        params = {
            "start": f"{x_lon}, {x_lat}",  # 출발지 (경도, 위도)
            "goal": f"{lon_em_filter}, {lat_em_filter}",  # 목적지 (경도, 위도)
            "option": "trafast"  # 실시간 빠른 길 옵션
        }
        response = requests.get(url, headers=headers, params=params).json()
        dist = response['route']['trafast'][0]['summary']['distance']  # 거리 (미터)
        return (dist/1000, hospital['병원이름'], hospital['전화번호 1'], hospital['전화번호 3'])

    # 병렬 처리로 API 요청 가속화
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(get_dist, filtered_hospitals.to_dict('records')))

    # 거리 기준 정렬
    results.sort(key=lambda x: x[0])

    # 가장 가까운 3개의 병원 출력
    print(f'탐색범위:{a*110}km * {a*110}km')
    for i in range(min(3, len(results))):
        print(results[i])
