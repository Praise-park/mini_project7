import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from concurrent.futures import ThreadPoolExecutor


# 0. load key file------------------    
def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()

# API 키 로드 및 설정 함수
def load_key(path):
    openai.api_key = load_file(path + 'api_key.txt')
    os.environ['OPENAI_API_KEY'] = openai.api_key

# 1-1 audio2text--------------------
def audio2text(audio_path, filename):

    client = OpenAI()
    audio_file = open(audio_path + filename, "rb")
    transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko",
                    response_format="text",
                    temperature=0.0)
    return transcript

# 1-2 text2summary------------------
def text2summary(input_text):
    client = OpenAI()

    system_role = '''당신은 응급상황에 대한 텍스트에서 핵심 내용을 훌륭하게 요약해주는 어시스턴트입니다.
    응답은 다음의 형식을 지켜주세요.
    {"summary": \"텍스트 요약\",
    "keyword" : \"핵심 키워드(3가지)\"}
    '''
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
    answer = response.choices[0].message.content
    parsed_answer = json.loads(answer)

    summary = parsed_answer["summary"]
    keyword = parsed_answer["keyword"]

    return summary + ', ' + keyword


# 2. model prediction------------------
def predict(text, model, tokenizer):
    # 입력 문장 토크나이징
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
def get_dist(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key):
    """
    출발지와 목적지 간 거리를 계산하는 함수 (네이버 지도 API 사용)
    """
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",    # 목적지 (경도, 위도)
        "option": "trafast"                  # 실시간 빠른 길 옵션
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    # 예외 처리
    try:
        dist = data['route']['trafast'][0]['summary']['distance']  # 거리 (미터)
        dist = dist / 1000  # km 단위로 변환
    except KeyError as e:
        print(f"응답 데이터에서 예상되는 키를 찾을 수 없음: {e}")
        return None

    return dist

# 3-2. recommendation------------------
def recommend_hospital3(hospitals, start_lat, start_lng, c_id, c_key, predicted_class):
    """
    조건에 맞는 병원 추천 함수:
    - 최소 3개의 병원이 필터링될 때까지 검색 범위를 확장
    - 병렬로 거리 계산
    - 가장 가까운 병원 3개를 정렬 및 반환 (데이터프레임 형태)
    """
    a = 0.05  # 초기 검색 범위
    filtered_hospitals = []
    if predicted_class > 2:
        return filtered_hospitals

    # 최소 3개의 병원이 필터링될 때까지 범위 확장
    while len(filtered_hospitals) < 3:
        print(f"검색 범위 확장 중: {a*110:.2f}km * {a*110:.2f}km")
        lat_min, lat_max = start_lat - a, start_lat + a
        lng_min, lng_max = start_lng - a, start_lng + a

        # 리스트를 사용하여 병원 필터링
        filtered_hospitals = [
            hospital for hospital in hospitals
            if lat_min <= hospital['위도'] <= lat_max and lng_min <= hospital['경도'] <= lng_max
        ]

        a += 0.05

    # 병렬로 거리 계산
    def compute_distance(hospital):
        distance = get_dist(start_lat, start_lng, hospital['위도'], hospital['경도'], c_id, c_key)
        if distance is None:  # 거리 계산 실패 시 처리
            return None
        return {
            "hospitalName": hospital['병원이름'],
            "address": hospital['주소'],
            "emergencyMedicalInstitutionType": hospital['응급의료기관 종류'],
            "phoneNumber1": hospital['전화번호 1'],
            "phoneNumber3": hospital['전화번호 3'],
            "latitude": hospital['위도'],
            "longitude": hospital['경도'],
            "distance": distance
        }

    # 병렬 처리
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_distance, filtered_hospitals))

    # None 값 제거
    #results = [r for r in results if r["거리(km)"] is not None]

    # 데이터프레임으로 변환 및 정렬
    temp = pd.DataFrame(results)
    temp = temp.sort_values(by="distance").reset_index(drop=True)

    # 가장 가까운 병원 3개 반환
    # print("가장 가까운 병원 3곳:")
    # print(temp.head(3))

    return temp.head(3)