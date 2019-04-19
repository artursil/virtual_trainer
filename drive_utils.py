from __future__ import print_function
import pickle
import os.path
import io
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient.http import MediaFileUpload,MediaIoBaseDownload

CREDS_PATH = 'connector/credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def drive_connect(creds_path):
    creds = None
    token_path = creds_path.replace('credentials.json','token.pickle')
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:       
            creds = pickle.load(token)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            creds_path, SCOPES)
        creds = flow.run_local_server()
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service

def get_file_id(items,pattern):
    for d in items:
        if d['name'].find(pattern)>-1:
            return d['id'],d['name']
    
    print(f'Cannot find file: {pattern}')
    return None, None


def download_file(service,filename,filename_new=None):
    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    file_id,filename = get_file_id(items,filename)
    if file_id==None:
        return
    request = service.files().get_media(fileId=file_id)
    if filename_new==None:
        filename_new=filename
    fh = io.FileIO(filename_new, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

def upload_file(service,file_to_upload,dfile_name=None):
    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    if dfile_name==None:
        dfile_name=file_to_upload.split('/')[-1]
    file_id=None
    for item in items:
        if item['name']==dfile_name:
            file_id=item['id']

    file_metadata = {
    'name': f'{dfile_name}',
    'mimeType': '*/*'
    }
    media = MediaFileUpload(file_to_upload,
                            mimetype='*/*',
                            resumable=True)
    if file_id==None:
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    else:
        file = service.files().update(fileId=file_id,body=file_metadata, media_body=media, fields='id').execute()
    print ('File ID: ' + file.get('id'))

def upload_db(db_file):
    service = drive_connect(CREDS_PATH)  
    upload_file(service,db_file)

def download_db(db_file):
    service = drive_connect(CREDS_PATH)  
    dfile = db_file.split('/')[-1]
    download_file(service,dfile,db_file)