from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
#from google.cloud import storage

from models import MeetingData,MailData
from models import json_file 

from dotenv import load_dotenv
import os
from os.path import join, dirname

import uvicorn
import json

from moviepy.editor import AudioFileClip
from moviepy.editor import VideoFileClip

import shutil
import os
import tempfile
from datetime import datetime
from pathlib import Path
import chardet

import openai
import whisper
import torch

from pyannote.audio import Pipeline
from pyannote.audio import Audio

from langchain.document_loaders.image import UnstructuredImageLoader
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI

import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import base64
from email.mime.text import MIMEText
from apiclient import errors


app = FastAPI()

#通信設定
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# OA_KEY = os.environ.get("OA_KEY")
HG_KEY = os.environ.get("HG_KEY")

# openai.api_key = OA_KEY 
# OPENAI_API_KEY = OA_KEY

openai.api_type = "azure"
openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ.get("AZ_KEY")

type = "service_account"
project_id = os.environ.get("PROJECT_ID")
private_key_id = os.environ.get("PRIVATE_KEY_ID")
private_key = os.environ.get("PRIVATE_KEY").replace('\\n', '\n')
client_email = os.environ.get("CLIENT_EMAIL")
client_id = os.environ.get("CLIENT_ID")
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = os.environ.get("AUTH_PROVIDER_X509_CERT_URL")
client_x509_cert_url = os.environ.get("CLIENT_X509_CERT_URL")

cred = credentials.Certificate({
    "type": type,
    "project_id": project_id,
    "private_key_id": private_key_id,
    "private_key": private_key,
    "client_email": client_email,
    "client_id": client_id,
    "auth_uri": auth_uri,
    "token_uri": token_uri,
    "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
    "client_x509_cert_url": client_x509_cert_url
})

FIREBASE_STORAGE_BUCKET=os.environ.get("FIREBASE_STORAGE_BUCKET")
firebase_admin.initialize_app(cred,{'storageBucket':FIREBASE_STORAGE_BUCKET})
bucket = storage.bucket()

# root_path =  "/home/site/wwwroot/app/static"
root_path = "static"

def guess_encoding(text_bytes):
    result = chardet.detect(text_bytes)
    encoding = result['encoding']
    confidence = result['confidence']
    return encoding, confidence

def process_text(text_bytes):
    encoding, confidence = guess_encoding(text_bytes)
    
    if encoding:
        print(f"Detected encoding: {encoding} (Confidence: {confidence:.2f})")
        try:
            decoded_text = text_bytes.decode(encoding)
            return decoded_text
        except UnicodeDecodeError:
            print("Decoding error. Could not decode using the detected encoding.")
    else:
        print("Encoding could not be detected.")

    return None


def summarize(text,type="all"):
    gpt_model = "gpt-35-turbo-16k"
    llm = AzureChatOpenAI(
    openai_api_base = openai.api_base,
    openai_api_version=openai.api_version,
    openai_api_key=openai.api_key,
    deployment_name=gpt_model,
    temperature=0, 
    max_tokens=4000
    )
    text_splitter = CharacterTextSplitter(separator = "\n",chunk_size=8000)

    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    #st.write(len(docs))

    # MapReduceで議題別の要約を箇条書きする
    # https://colab.research.google.com/github/nyanta012/demo/blob/main/langchain_summary.ipynb#scrollTo=r8zf7R3r7XdL
    # https://zenn.dev/seiyakitazume/articles/d4a11404320a07
    # 分割した文章の要約するプロンプト
    prompt_template = """
    【制約条件】に従って、【議事録】の重要な要点をまとめてください。

    【制約条件】
    *必ず【議事録】の内容に基づいてまとめること
    *【議事録】の各文章から言いたいことを抽出すること
    *抽出した内容から関連したものを総合して一つの要点とすること
    *重要な要点を6つ以内に絞り、【出力形式】のフォーマットで箇条書きすること

    【出力形式】
    箇条書きの出力形式は以下のフォーマットとします。各要点は50文字以内にしてください。
    ----------------
    1.要点1
    2.要点2
    ----------------
    このフォーマット外の出力はせず、先頭に番号が付いた文字列だけ出力してください。

    【議事録】
    {text}

    それでは、【制約条件】に従って、【議事録】の内容に基づいて要点をまとめてください。
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # 分割した文章の要約するプロンプト
    map_reduce_template = """
    【制約条件】に従って、【議事録の要点】をまとめてください。

    【制約条件】
    *【議事録の要点】の重複する箇所はひとつの文章にまとめること
    *【議事録の要点】の中の関連する要点同士は、１つに要約しまとめること


    【出力形式】
    箇条書きの出力形式は以下のフォーマットとします。
    ----------------
    1.要点1
    2.要点2
    ----------------
    このフォーマット外の出力はしないでください。

    【議事録の要点】
    {text}

    それでは、【制約条件】に従って、【議事録の要点】をまとめてください。
    """
    MAP_PROMPT = PromptTemplate(template=map_reduce_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=MAP_PROMPT, verbose=True)
    result=chain(docs, return_only_outputs=True)
    
    # ①要約データを議題別に分類するプロンプト
    # 要約内容の確認
    docs_sum = result['output_text']

    prompt_template = """
    【制約条件】に従って、【議事録】を分類してまとめてください。

    【制約条件】
    * 必ず【議事録】の内容を全て使ってまとめること
    * 【議事録】の関連度の高い項目同士をまとめて、4つに分類すること
    * 各分類にタイトルを作成すること
    * 各タイトルとその項目を【出力形式】のフォーマットで箇条書きすること

    【出力形式】
    箇条書きの出力形式は以下のフォーマットとします。タイトルは4つ以内に限定してください。
    ----------------
    # 分類1のタイトル
    * 項目1
    * 項目2
    # 分類2のタイトル
    * 項目1
    * 項目2
    ----------------
    このフォーマット外の出力はしないでください。

    【議事録】
    {}

    それでは、【制約条件】に従って、【議事録】を分類してまとめてください。
    """
    prompt = prompt_template.format(docs_sum)
    
    # 回答の生成
    response = openai.ChatCompletion.create(
        engine='gpt-4',
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature = 0,
    )
    # 回答の表示
    print(response.choices[0]["message"]["content"].strip())
    # 議題の題目(title_list)と内容(content_list)の抽出
    text = response.choices[0]["message"]["content"].strip()
    text_list = text.split("# ")
    text_list.pop(0)
    title_list=[]
    content_list=[]
    print(text_list)
    for i, t in enumerate(text_list):
        title = text.split("# ")[i].split("\n")[0]
        if not title == "":
            title_list.append(title) # 議題の題目
            content_list.append(t) # 内容
    print(title_list)
    print(content_list)
    # ②各題目の要約を作成
    meeting_sum =""
    for i, t in enumerate(content_list):
        # 要約を作成するプロンプト
        prompt = "次のテキストを200文字以内で要約してください：{}".format(t)
        # 回答の生成
        response = openai.ChatCompletion.create(
            engine="gpt-4",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature = 0,
        )
        # 文章の統合
        meeting_sum = meeting_sum + "# "+ title_list[i] +"\n"+ response.choices[0]["message"]["content"].strip() + "\n"

    if type=="all":
        output =  "要約\n"+ meeting_sum + "\n" + "要点\n" + docs_sum
    else:
        output =  "要約\n"+ meeting_sum

    return output

def get_video_duration(video_path):
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        return duration
    except Exception as e:
        print("Error:", e)
        return None

async def gettime_transcription(upload_file, d):
    suffix = Path(upload_file.filename).suffix
    print('filename:' ,upload_file.filename)
    
    with tempfile.NamedTemporaryFile(delete=True,  suffix=suffix) as temp_file:
        temp_file_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
        
        # UploadFileオブジェクトから読み込んで一時ファイルに書き込む
        with open(temp_file_path, 'wb') as file:
            file.write(await upload_file.read())
        
        video_length = get_video_duration(temp_file_path)

        precision = d["precision"]
        if precision=="low":
            predict_time = 3 * video_length/60
        elif precision =="middle":
            predict_time = 9 * video_length/60

        else:
            print("現在使用できません。")
        return video_length /60, predict_time

async def get_time_transcription(upload_file, precision):
    suffix = Path(upload_file.filename).suffix
    print('filename:' ,upload_file.filename)
    
    with tempfile.NamedTemporaryFile(delete=True,  suffix=suffix) as temp_file:
        temp_file_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
        
        # UploadFileオブジェクトから読み込んで一時ファイルに書き込む
        with open(temp_file_path, 'wb') as file:
            file.write(await upload_file.read())
        
        video_length = get_video_duration(temp_file_path)

        if precision=="low":
            predict_time = 3 * video_length/60
        elif precision =="middle":
            predict_time = 9 * video_length/60

        else:
            print("現在使用できません。")
        return video_length /60, predict_time

async def transcription_whisper(upload_file, d):
    text_all=""
    transcription=""
    model=False
    precision = d["precision"]
    
    suffix = Path(upload_file.filename).suffix
    print('filename:' ,upload_file.filename)
    
    with tempfile.NamedTemporaryFile(delete=True,  suffix=suffix) as temp_file:
        temp_file_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
        
        # UploadFileオブジェクトから読み込んで一時ファイルに書き込む
        with open(temp_file_path, 'wb') as file:
            file.write(await upload_file.read())
        
        video_length = get_video_duration(temp_file_path)

        if precision=="low":
            model = "small"
            predict_time = 3 * video_length/60
            print("終了時間予測：", str(predict_time), "分")
        elif precision =="middle":
            model = "medium"
            predict_time = 9 * video_length/60
            print("終了時間予測：", str(predict_time), "分")

        else:
            print("現在使用できません。")

        # mp4からmp3への変換
        audiofile = AudioFileClip(temp_file_path)
        temp_file_wav = tempfile.NamedTemporaryFile(suffix='.wav').name
        audiofile.write_audiofile(temp_file_wav)    

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1"
            , use_auth_token=HG_KEY
        )

        audio = Audio(sample_rate=16000, mono=True)
        
        diarization = pipeline(temp_file_wav
            #, num_speakers=3
            ,min_speakers=2, max_speakers=5
            )
        model = whisper.load_model(model)

        pre = { "time_start":float(1),'time_end':float(1),"speaker": "","wave":torch.tensor([[]])}
        text= str("テスト")

        transcription = []
        transcription_all = ""

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            transcription_speak = {}
            waveform, sample_rate = audio.crop(temp_file_wav, segment)
            if pre['speaker'] == speaker:
                if (segment.end - segment.start) >= 1.0:
                    cat_waveform = torch.cat([pre['wave'], waveform], dim=1)
            #      print("pre:", pre['wave'].size(),"after", cat_waveform.size())
                    pre['wave'] = cat_waveform
                    pre['time_end'] = segment.end
            else:
                time_start, time_end = pre['time_start'], segment.start
                if pre['wave'].nelement() > 0:  # pre['wave']が空でないことを確認する
                    text = model.transcribe(pre['wave'].squeeze().numpy(), language="ja")["text"]
                else:
                    text = ""  # pre['wave']が空の場合、空文字列を代入

                if not text=="":
                    text_all += (text+"\n")
                    #transcription_speak["開始時間"] = time_start
                    #transcription_speak["終了時間"] = time_end
                    transcription_speak["話者"] = pre["speaker"]
                    transcription_speak["会話"] = text
                    transcription.append(transcription_speak)
                    transcription_all += transcription_speak["話者"] +" ： "+ transcription_speak["会話"] + "\n"
                    #print(f"[{time_start:03.1f}s - {time_end:03.1f}s] {pre['speaker']}: {text}")
                pre = { "time_start":'','time_end':"","speaker": "","wave":torch.tensor([[]])}
                pre['time_start'] =segment.end
                pre['speaker'] = speaker
                pre['wave'] = waveform
        
        return text_all, transcription_all



@app.get("/")
def index():
    return "Hello world"

@app.post("/settings/{id}")
async def recieve(id : str, data:MeetingData):
    title = data.title
    date = data.date
    member_list = ""
    for member in data.participants:
        member_list += member + "、" 
    purpose = data.purpose
    try:
        precision = data.precision
    except:
        precision = ""

    # Azure App Serviceでの保存ディレクトリ
    save_dir  =  root_path + f"/{id}/param"
    os.makedirs(save_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_settings.json"

    path = f"{save_dir}/{filename}"

    with open(path, "w", encoding="shift-jis") as f:
        params = {"title":title, "date":date, "member":member_list, "purpose":purpose, "precision":precision}    
        json.dump(params, f)

    return filename
    
@app.post("/get_time/{precision}")
async def uploadfile_get_time( precision : str, upload_file: UploadFile = File(...)):
    video_length, transcription_time= await get_time_transcription(upload_file, precision)
    return int(video_length), int(transcription_time)

@app.post("/get_filename/{filename}")
async def get_filename( filename : str ,upload_file: UploadFile = File(...)):
    suffix = Path(upload_file.filename).suffix
    if suffix == ".txt":
        summary_filename = f"summary_{filename[:15]}_{upload_file.filename}"
        transcription_filename = f"transcription_{filename[:15]}_{upload_file.filename}"

    elif suffix == ".mp4":
        summary_filename = f"summary_{filename[:15]}_{upload_file.filename.replace('mp4', 'txt')}"
        transcription_filename = f"transcription_{filename[:15]}_{upload_file.filename.replace('mp4', 'txt')}"
    return [summary_filename, transcription_filename]

@app.post("/uploadfile/{id}/{type}/{filename}")
async def upload_file( id : str, type:str,filename : str ,upload_file: UploadFile = File(...)):
    print(filename)
    
    path_setting = f"{root_path}/{id}/param/{filename}"
    with open(path_setting, "r", encoding="shift-jis") as f:
        d = json.load(f)

    print("setting",d)

    suffix = Path(upload_file.filename).suffix
    if suffix == ".txt":
        print('filename:' ,upload_file.filename)
        with tempfile.NamedTemporaryFile(delete=True,  suffix=suffix) as temp_file:
            shutil.copyfileobj(upload_file.file, temp_file)
            temp_file.seek(0)
            temp_path = Path(temp_file.name)

            content = process_text(temp_file.read())
            print("content:",content)
            print("Path:",temp_path)

            result = summarize(content, type)
        
        summary=""
        summary+= "打合せ："+ d["title"] + "\n"
        summary+="日付："+ d["date"] + "\n"
        summary+="参加者："+ d["member"] + "\n"
        summary+="目的："+ d["purpose"] + "\n"
        summary+= result

        filename = f"summary_{filename[:15]}_{upload_file.filename}"

        save_dir =  root_path + f"/{id}/result/summary"
        os.makedirs(save_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成
        path_result = f"{save_dir}/{filename}"

        with open(path_result, "w", encoding="shift-jis") as file:
            file.write(summary)

        bucket = storage.bucket()
        blob = bucket.blob(path_result)
        blob.upload_from_filename(path_result)

        direct = ["/result/summary","/result/transcription","/param"]
        for dir in direct:
            rootdir = root_path + f"/{id}/{dir}"
            for f in os.listdir(rootdir):
                os.remove(os.path.join(rootdir, f))

        return { "filename":filename, "transcription":"", "summary":summary}
        
    elif suffix == ".mp4":
        path_setting = f"{root_path}/{id}/param/{filename}"
        with open(path_setting, "r", encoding="shift-jis") as f:
            d = json.load(f)

        print("setting",d)

        text_all, transcription_all = await transcription_whisper(upload_file, d)

        result = summarize(text_all, type)

        summary=""
        summary+= "打合せ："+ d["title"] + "\n"
        summary+="日付："+ d["date"] + "\n"
        summary+="参加者："+ d["member"] + "\n"
        summary+="目的："+ d["purpose"] + "\n"
        summary+= result


        summary_filename = f"summary_{filename[:15]}_{upload_file.filename.replace('mp4', 'txt')}"
        transcription_filename = f"transcription_{filename[:15]}_{upload_file.filename.replace('mp4', 'txt')}"
        # path_summary = f"app/static/result/summary/{filename}"
        # path_transcription = f"app/static/result/transcription/{filename}"

        save_summary_path =  root_path + f"/{id}/result/summary"
        save_transcription_path =  root_path + f"/{id}/result/transcription"
        os.makedirs(save_summary_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        os.makedirs(save_transcription_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        path_summary = f"{save_summary_path}/{summary_filename}"
        path_transcription = f"{save_transcription_path}/{transcription_filename}"

        with open(path_summary, "w", encoding="shift-jis") as file:
            file.write(summary)

        with open(path_transcription, "w", encoding="shift-jis") as file:
            file.write(transcription_all) 

        bucket = storage.bucket()
        blob = bucket.blob(path_summary)
        blob.upload_from_filename(path_summary)

        blob = bucket.blob(path_transcription)
        blob.upload_from_filename(path_transcription)

        direct = ["/result/summary","/result/transcription","/param"]
        for dir in direct:
            rootdir = root_path + f"/{id}/{dir}"
            for f in os.listdir(rootdir):
                os.remove(os.path.join(rootdir, f))

        return { "filename":filename, "transcription":transcription_all, "summary":summary}


@app.get("/downloadfile/{id}/{filename}")
async def get_summary(id : str, filename:str):
    file_path = f"{root_path}/{id}/result/summary/{filename}"
    now = datetime.now()

    print(file_path)

    response = FileResponse(
                path=file_path,
                filename=f"download_{now.strftime('%Y%m%d%H%M')}_{filename}"
                )
    return response

@app.get("/downloadtranscription/{id}/{filename}")
async def get_transcription(id : str,  filename:str):
    filename.replace("summary", "transcription")
    file_path = f"{root_path}/{id}/result/transcription/{filename}"
    now = datetime.now()

    print(file_path)
    try:
        response = FileResponse(
                    path=file_path,
                    filename=f"download_{now.strftime('%Y%m%d%H%M')}_{filename}"
                    )
        return response
    except:
        return "ファイルは存在しません。"

@app.post("/announce")
async def annouce(data:MailData):
    address = data.address

    # 1. Gmail APIのスコープを設定
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']

    # 2. メール本文の作成
    def create_message(sender, to, subject, message_text):
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject
        encode_message = base64.urlsafe_b64encode(message.as_bytes())
        return {'raw': encode_message.decode()}
    # 3. メール送信の実行
    def send_message(service, user_id, message):
        try:
            message = (service.users().messages().send(userId=user_id, body=message)
                    .execute())
            print('Message Id: %s' % message['id'])
            return message
        except errors.HttpError as error:
            print('An error occurred: %s' % error)
    
    # 4. アクセストークンの取得
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            mail_client_id = os.getenv("mail_client_id")
            mail_project_id = os.getenv("mail_project_id")
            mail_auth_uri = os.getenv("mail_auth_uri")
            mail_token_uri = os.getenv("mail_mail_token_uri")
            mail_auth_provider_x509_cert_url = os.getenv("mail_auth_provider_x509_cert_url")
            mail_client_secret = os.getenv("mail_client_secret")
 #           mail_redirect_uris = os.getenv("mail_redirect_uri")
            
            # 変数を統合してJSON文字列に戻す
            client_info = {
                "installed": {
                    "client_id": mail_client_id,
                    "project_id": mail_project_id,
                    "auth_uri": mail_auth_uri,
                    "token_uri": mail_token_uri,
                    "auth_provider_x509_cert_url": mail_auth_provider_x509_cert_url,
                    "client_secret": mail_client_secret,
 #                   "redirect_uris": list(mail_redirect_uris)
                }
            }
            print("client_info",client_info)
            flow = InstalledAppFlow.from_client_config(
            client_info , SCOPES)
            creds = flow.run_local_server()
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    # 6. メール本文の作成
    sender = 'giziroku.tech0@gmail.com'
    to = address
    subject = '議事録の作成が完了しました'
    message_text = '議事録の作成が完了しました。HPからダウンロードしてください'
    message = create_message(sender, to, subject, message_text)
    # 7. Gmail APIを呼び出してメール送信
    send_message(service, 'me', message)