import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build 
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1TSnNcvGbNEh6hY8pcprBwKMe5I0ze23uIN7h2lBtc5I"

def main():
    crendentials = None
    if os.path.exists("token.json"):
        crendentials = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not crendentials or not crendentials.valid:
        if crendentials and crendentials.expired and crendentials.refresh_tokens:
            crendentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("crendentials.json", SCOPES)
            crendentials = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(crendentials.to_json())

    try:
        service = build("sheets","v4",credentials=crendentials)
        sheets = service.spreadsheets()

        result = sheets.values().get(spreadsheetId=SPREADSHEET_ID, range = "SHEET1!A1:B5").execute()
        values = result.get("values",[])

        print(values)

    except HttpError as error:
        print("error")

if __name__ == "__main__":
    main()