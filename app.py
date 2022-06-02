import streamlit as st

import io
import os
import yaml
import requests
import json

from fastapi import FastAPI

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

def remove_special(string):
    return re.sub(r"[^가-힣a-zA-Z0-9]","", string)

def check_am_pm(string):
    am_pm = string.split(' ')[0]
    hour = int(string.split(' ')[1].split(':')[0])
    minute = int(string.split(' ')[1].split(':')[1])
    if am_pm == '오전' and hour == 12:
        hour = 0
    elif am_pm == '오후' and hour != 12:
        hour += 12
    return format(hour, '02') + ':' + format(minute, '02') + ':' + '00'

def make_df(file_path):
    """
    txt파일을 dataframe으로 변환
    """
    with open(file_path, 'r', encoding='UTF-8') as input_file:
        person = []
        date = []
        time = []
        utterance = []
        d = ''
        for line in input_file:
            if line.startswith('---------------'):
                d = line.split(' ')
                d = d[1][:-1] + '-' + format(int(d[2][:-1]), '02') + '-' + format(int(d[3][:-1]), '02')
            elif line.startswith('['):
                sp = line.split('] ')
                if sp[0][1:] == '방장봇': # '삭제된 메시지입니다.'
                    continue
                
                # context에 ']'가 있는 경우
                if len(sp) > 3:
                    tmp = '] '.join(sp[2:]).strip()
                    # 관계없는 키워드 제외시  이곳과 아래구문에 추가해주시면 됩니다.
                    # if tmp == '삭제된 메시지입니다.' or '하트인증' in tmp or '/닉네임' in tmp or '/SCD란' in tmp or '친목다과회' in tmp or '토론방' in tmp or '디스코드' in tmp or '신문고' in tmp:
                    if tmp == '삭제된 메시지입니다.' or tmp.startswith('/'):
                        continue
                    else:
                        utterance.append(tmp)
                else:
                    tmp = sp[2].strip()
                    # if tmp == '삭제된 메시지입니다.' or '하트인증' in tmp or '/닉네임' in tmp or '/SCD란' in tmp or '친목다과회' in tmp or '토론방' in tmp or '디스코드' in tmp or '신문고' in tmp:
                    if tmp == '삭제된 메시지입니다.' or tmp.startswith('/'):
                        continue
                    else:
                        utterance.append(tmp)
                    
                    
                # person.append(remove_special(sp[0][1:]))
                person.append(sp[0][1:])
                date.append(d)
                time.append(check_am_pm(sp[1][1:]))
        df = pd.DataFrame({'person':person, 'date': date, 'time': time, 'utterance':utterance})
        return df
    
def context_punc(c):
    """
    글형식이 아닌 데이터를 전처리
    """
    try:
        if c == '이모티콘':
            return '#@이모티콘#'
        elif c == '사진':
            return '#@시스템#사진#'
        elif c.startswith('사진') and c.endswith('장') and len(c.split(' ')) == 2:
            return '#@시스템#사진#'
        elif c == '동영상':
            return '#@시스템#동영상#'
        else:
            return c
    except:
        return c
    
def get_attribute(df):
    """
    알맞는 형식으로 변환
    """
    _utteranceID = []
    _turnID = []
    _participantID = []

    _utterance = 0
    _turn = 0
    _participant = 0
    _participant_dict = {}

    before_participant = ""
    for idx in df.index:
        _utterance += 1
        present_participant = df.loc[idx,'person']
        if present_participant != before_participant:
            _turn += 1
            before_participant = present_participant
        if present_participant not in _participant_dict.keys():
            _participant += 1
            _participant_dict[present_participant] = "P" + format(_participant, '03')
            _participantID.append("P"+format(_participant, '03'))
        else:  # present_participant in participant_dict.keys():
            _participantID.append(_participant_dict[present_participant])
        _utteranceID.append("U"+str(_utterance))
        _turnID.append("T"+str(_turn))
    
    df['utteranceID'] = _utteranceID
    df['turnID'] = _turnID
    df['participantID'] = _participantID
    
    return df, _utterance, _turn, _participant

def get_last_dialogue(df):
    """
    모든 대화들 중 최근시간대의 대화만 추출
    """
    last_idx = 0
    for idx in df.index:
        try:
            date1 = df.loc[idx]['date']
            date2 = df.loc[idx+1]['date']
            time1 = df.loc[idx]['time']
            time2 = df.loc[idx+1]['time']

            time1 = datetime.strptime(date1 + ' ' + time1, '%Y-%m-%d %H:%M:%S')
            time2 = datetime.strptime(date2 + ' ' + time2, '%Y-%m-%d %H:%M:%S')
            time_interval = time2 - time1
            if time_interval.seconds/3600 > 2:
                last_idx = idx
        except:
            pass
    return df[last_idx:].reset_index()

def txt_to_json(upload_file):  # txt파일이 입력됨
    talk_df = make_df(upload_file)
    talk_df['utterance'] = talk_df['utterance'].transform(context_punc)
    talk_df = get_last_dialogue(talk_df)
    talk_df, utterance, turn, participant = get_attribute(talk_df)
    talk_df = talk_df[['utteranceID', 'turnID', 'participantID', 'date', 'time', 'utterance']]  # 열순서 바꾸기
    
    body = talk_df.to_json(orient = 'records', force_ascii=False)
    body = json.loads(body)
    
    total = {}
    total_body = {}
    dialogueInfo = {}
    dialogueInfo["numberOfParticipants"] = participant
    dialogueInfo["numberOfUtterances"] = utterance
    dialogueInfo["numberOfTurns"] = turn
    
    # 추후 사용시 매개변수로 받게해서 사용할 예정
    # dialogueInfo["dialogueID"] = 'mbti'
    # dialogueInfo["type"] = "일상 대화"
    # dialogueInfo["topic"] = "개인 및 관계"
    
    total['header'] = dialogueInfo
    total_body['dialogue'] = body
    total['body'] = total_body
    
    final_json = {}
    tmp_json = {}
    final_json["numberOfItems"] = 1
    tmp_json[0] = total
    # 여러개 대화만들기 위한 더미데이터
    tmp_json[1] = {"body": {
        "dialogue": [
          {
            "utteranceID": "U1",
            "turnID": "T1",
            "participantID": "P01",
            "date": "2020-11-25",
            "time": "23:45:00",
            "utterance": "abcd"
          }]}}
    final_json['data'] = tmp_json
    
    return final_json

def preprocess(js_file):
    dialogue = js_file['data'][0]['body']['dialogue']
    
    return_string = ""
    for string in dialogue:
        return_string += string['participantID'] + ": " + string['utterance'] + "\r\n "

    return return_string

def main():
    st.title("Golden summary & Show image")
    
    uploaded_file = st.file_uploader("Input your dialogue data", type=["txt"])

    print(uploaded_file)
    if uploaded_file:
        js = txt_to_json(uploaded_file.name)  # json

        data = {'dialogue':preprocess(dialogue_data)}
       
        a = requests.post('http://127.0.0.1:8000/upload', data = json.dumps(data))
    
        st.write(a.json())


main()
