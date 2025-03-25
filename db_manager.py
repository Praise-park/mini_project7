import sqlite3
from datetime import datetime

# 데이터베이스 경로 설정
db_dir = "db"  # 폴더 경로
db_name = "em.db"  # 데이터베이스 파일 이름
db_path = f"{db_dir}/{db_name}"

def insert_log(input_text, input_latitude, input_longitude, em_class, hospital_list):
    """
    log 테이블에 데이터를 삽입하는 함수.
    
    :param input_text: 입력 텍스트
    :param input_latitude: 입력 위도
    :param input_longitude: 입력 경도
    :param em_class: 응급 등급
    :param hospital_list: 추천 병원 리스트 (최대 3개)
    """
    # 요청 시간 생성
    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 병원 데이터 분리
    hospital1, addr1, tel1 = hospital_list[0]["hospitalName"], hospital_list[0]["address"], hospital_list[0]["phoneNumber1"]
    hospital2, addr2, tel2 = hospital_list[1]["hospitalName"], hospital_list[1]["address"], hospital_list[1]["phoneNumber1"]
    hospital3, addr3, tel3 = hospital_list[2]["hospitalName"], hospital_list[2]["address"], hospital_list[2]["phoneNumber1"]

    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 데이터 삽입
    cursor.execute('''
        INSERT INTO log (
            datetime, input_text, input_latitude, input_longitude, em_class,
            hospital1, addr1, tel1, hospital2, addr2, tel2, hospital3, addr3, tel3
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (dt, input_text, input_latitude, input_longitude, em_class,
          hospital1, addr1, tel1, hospital2, addr2, tel2, hospital3, addr3, tel3))

    # 변경사항 저장 및 연결 종료
    conn.commit()
    conn.close()

    print("데이터가 성공적으로 삽입되었습니다.")