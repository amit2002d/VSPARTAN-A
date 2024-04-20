import gspread
import streamlit as st
from datetime import datetime
import pandas as pd
from streamlit_extras.switch_page_button import switch_page



secrets = st.session_state.secrets
try:
    all_data = st.session_state.all_data
except:
    st.error("Please run the app from the main page.")
selected_tab = st.selectbox("Select Stock", options=list(all_data.keys()), key='Stock')
sell_price = st.number_input("Sell Price", value=0.0, key='Sell Price')
row_num = st.number_input("Number of rows to be deleted", value=0.0, key='Row Number')
if st.session_state.user == 'Amit':
    spreadsheet_id = secrets["connections"]["gsheets"]["spreadsheet"]
elif st.session_state.user == "Deepti":
    spreadsheet_id = secrets["connections"]["gsheets_d"]["spreadsheet"]
elif st.session_state.user == "Mridul":
    spreadsheet_id = secrets["connections"]["gsheets_m"]["spreadsheet"]
elif st.session_state.user == "Hemank":
    spreadsheet_id = secrets["connections"]["gsheets_h"]["spreadsheet"]

def overwrite_worksheet_with_df(worksheet, df):
    try:
        worksheet.clear()

        values = df.values.tolist()

        worksheet.update("A1", values)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main code
if st.button("Delete") and row_num > 0:
    with st.spinner("Deleting..."):
        if st.session_state.user == 'Amit':
            try:
                client = gspread.service_account_from_dict({
                    "type": secrets["connections"]["gsheets"]["type"],
                    "project_id": secrets["connections"]["gsheets"]["project_id"],
                    "private_key_id": secrets["connections"]["gsheets"]["private_key_id"],
                    "private_key": secrets["connections"]["gsheets"]["private_key"],
                    "client_email": secrets["connections"]["gsheets"]["client_email"],
                    "client_id": secrets["connections"]["gsheets"]["client_id"],
                    "auth_uri": secrets["connections"]["gsheets"]["auth_uri"],
                    "token_uri": secrets["connections"]["gsheets"]["token_uri"],
                    "auth_provider_x509_cert_url": secrets["connections"]["gsheets"]["auth_provider_x509_cert_url"],
                    "client_x509_cert_url": secrets["connections"]["gsheets"]["client_x509_cert_url"]
                })

                # Open the spreadsheet
                spreadsheet = client.open_by_key(spreadsheet_id)

                worksheet = spreadsheet.worksheet(selected_tab)

                # Get all values as DataFrame
                df = pd.DataFrame(worksheet.get_all_values(), columns=None)
                last_rows = df.tail(int(row_num)).values.tolist()
                for row in last_rows:
                    try:
                        client = gspread.service_account_from_dict({
                            "type": secrets["connections"]["gsheets_sell"]["type"],
                            "project_id": secrets["connections"]["gsheets_sell"]["project_id"],
                            "private_key_id": secrets["connections"]["gsheets_sell"]["private_key_id"],
                            "private_key": secrets["connections"]["gsheets_sell"]["private_key"],
                            "client_email": secrets["connections"]["gsheets_sell"]["client_email"],
                            "client_id": secrets["connections"]["gsheets_sell"]["client_id"],
                            "auth_uri": secrets["connections"]["gsheets_sell"]["auth_uri"],
                            "token_uri": secrets["connections"]["gsheets_sell"]["token_uri"],
                            "auth_provider_x509_cert_url": secrets["connections"]["gsheets_sell"]["auth_provider_x509_cert_url"],
                            "client_x509_cert_url": secrets["connections"]["gsheets_sell"]["client_x509_cert_url"]
                        })
                        spreadsheet_key = secrets["connections"]["gsheets_sell"]["spreadsheet"]
                        sheet = client.open_by_key(spreadsheet_key).get_worksheet(0)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.stop()
                    column_a_values = sheet.col_values(1)
                    last_row_index = len(column_a_values) + 1
                    date_obj = datetime.strptime(row[0], "%Y-%m-%d")
                    formatted_date = date_obj.strftime("%d-%b-%y")
                    # formatted_date = datetime.strptime(formatted_date, "%d-%b-%y")
                    data = [formatted_date,selected_tab,selected_tab, float(row[2]), float(row[1]), '',float(row[1])*float(row[2]),sell_price,str(datetime.now().strftime("%d-%b-%y")), st.session_state.total_invested - (float(row[1])*float(row[2]))]
                    sheet.update(f"A{last_row_index}:J{last_row_index}", [data])
                    st.session_state.total_invested = st.session_state.total_invested - (float(row[1])*float(row[2]))
                    sheet.update(f"N{last_row_index}", [[50/int(row_num)]])
                df = df[:-int(row_num)]
                overwrite_worksheet_with_df(worksheet, df)

                # Provide success message
                st.success(f"Deleted {row_num} rows successfully from the bottom of the worksheet '{selected_tab}'.")
                switch_page("app")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        elif st.session_state.user == 'Deepti':
            try:
                client = gspread.service_account_from_dict({
                    "type": secrets["connections"]["gsheets_d"]["type"],
                    "project_id": secrets["connections"]["gsheets_d"]["project_id"],
                    "private_key_id": secrets["connections"]["gsheets_d"]["private_key_id"],
                    "private_key": secrets["connections"]["gsheets_d"]["private_key"],
                    "client_email": secrets["connections"]["gsheets_d"]["client_email"],
                    "client_id": secrets["connections"]["gsheets_d"]["client_id"],
                    "auth_uri": secrets["connections"]["gsheets_d"]["auth_uri"],
                    "token_uri": secrets["connections"]["gsheets_d"]["token_uri"],
                    "auth_provider_x509_cert_url": secrets["connections"]["gsheets_d"]["auth_provider_x509_cert_url"],
                    "client_x509_cert_url": secrets["connections"]["gsheets_d"]["client_x509_cert_url"]
                })

                spreadsheet = client.open_by_key(spreadsheet_id)

                worksheet = spreadsheet.worksheet(selected_tab)

                # Get all values as DataFrame
                df = pd.DataFrame(worksheet.get_all_values(), columns=None)
                last_rows = df.tail(int(row_num)).values.tolist()
                for row in last_rows:
                    try:
                        client = gspread.service_account_from_dict({
                            "type": secrets["connections"]["gsheets_sell_d"]["type"],
                            "project_id": secrets["connections"]["gsheets_sell_d"]["project_id"],
                            "private_key_id": secrets["connections"]["gsheets_sell_d"]["private_key_id"],
                            "private_key": secrets["connections"]["gsheets_sell_d"]["private_key"],
                            "client_email": secrets["connections"]["gsheets_sell_d"]["client_email"],
                            "client_id": secrets["connections"]["gsheets_sell_d"]["client_id"],
                            "auth_uri": secrets["connections"]["gsheets_sell_d"]["auth_uri"],
                            "token_uri": secrets["connections"]["gsheets_sell_d"]["token_uri"],
                            "auth_provider_x509_cert_url": secrets["connections"]["gsheets_sell_d"]["auth_provider_x509_cert_url"],
                            "client_x509_cert_url": secrets["connections"]["gsheets_sell_d"]["client_x509_cert_url"]
                        })
                        spreadsheet_key = secrets["connections"]["gsheets_sell_d"]["spreadsheet"]
                        sheet = client.open_by_key(spreadsheet_key).get_worksheet(0)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.stop()
                    column_a_values = sheet.col_values(1)
                    last_row_index = len(column_a_values) + 1
                    date_obj = datetime.strptime(row[0], "%Y-%m-%d")
                    formatted_date = date_obj.strftime("%d-%b-%y")
                    data = [formatted_date,selected_tab,selected_tab, float(row[2]), float(row[1]), '',float(row[1])*float(row[2]),sell_price,str(datetime.now().strftime("%d-%b-%y")), st.session_state.total_invested - (float(row[1])*float(row[2]))]
                    sheet.update(f"A{last_row_index}:J{last_row_index}", [data])
                    st.session_state.total_invested = st.session_state.total_invested - (float(row[1])*float(row[2]))
                    print(st.session_state.total_invested)
                    sheet.update(f"N{last_row_index}", [[50/int(row_num)]])
                df = df[:-int(row_num)]
                overwrite_worksheet_with_df(worksheet, df)

                # Provide success message
                st.success(f"Deleted {row_num} rows successfully from the bottom of the worksheet '{selected_tab}'.")
                switch_page("app")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        elif st.session_state.user == 'Mridul':
            try:
                client = gspread.service_account_from_dict({
                    "type": secrets["connections"]["gsheets_m"]["type"],
                    "project_id": secrets["connections"]["gsheets_m"]["project_id"],
                    "private_key_id": secrets["connections"]["gsheets_m"]["private_key_id"],
                    "private_key": secrets["connections"]["gsheets_m"]["private_key"],
                    "client_email": secrets["connections"]["gsheets_m"]["client_email"],
                    "client_id": secrets["connections"]["gsheets_m"]["client_id"],
                    "auth_uri": secrets["connections"]["gsheets_m"]["auth_uri"],
                    "token_uri": secrets["connections"]["gsheets_m"]["token_uri"],
                    "auth_provider_x509_cert_url": secrets["connections"]["gsheets_m"]["auth_provider_x509_cert_url"],
                    "client_x509_cert_url": secrets["connections"]["gsheets_m"]["client_x509_cert_url"]
                })

                spreadsheet = client.open_by_key(spreadsheet_id)

                worksheet = spreadsheet.worksheet(selected_tab)

                # Get all values as DataFrame
                df = pd.DataFrame(worksheet.get_all_values(), columns=None)
                last_rows = df.tail(int(row_num)).values.tolist()
                for row in last_rows:
                    try:
                        client = gspread.service_account_from_dict({
                            "type": secrets["connections"]["gsheets_sell_m"]["type"],
                            "project_id": secrets["connections"]["gsheets_sell_m"]["project_id"],
                            "private_key_id": secrets["connections"]["gsheets_sell_m"]["private_key_id"],
                            "private_key": secrets["connections"]["gsheets_sell_m"]["private_key"],
                            "client_email": secrets["connections"]["gsheets_sell_m"]["client_email"],
                            "client_id": secrets["connections"]["gsheets_sell_m"]["client_id"],
                            "auth_uri": secrets["connections"]["gsheets_sell_m"]["auth_uri"],
                            "token_uri": secrets["connections"]["gsheets_sell_m"]["token_uri"],
                            "auth_provider_x509_cert_url": secrets["connections"]["gsheets_sell_m"]["auth_provider_x509_cert_url"],
                            "client_x509_cert_url": secrets["connections"]["gsheets_sell_m"]["client_x509_cert_url"]
                        })
                        spreadsheet_key = secrets["connections"]["gsheets_sell_m"]["spreadsheet"]
                        sheet = client.open_by_key(spreadsheet_key).get_worksheet(0)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.stop()
                    column_a_values = sheet.col_values(1)
                    last_row_index = len(column_a_values) + 1
                    date_obj = datetime.strptime(row[0], "%Y-%m-%d")
                    formatted_date = date_obj.strftime("%d-%b-%y")
                    # formatted_date = datetime.strptime(formatted_date, "%d-%b-%y")

                    data = [formatted_date,selected_tab,selected_tab, float(row[2]), float(row[1]), '',float(row[1])*float(row[2]),sell_price,str(datetime.now().strftime("%d-%b-%y")), st.session_state.total_invested - (float(row[1])*float(row[2]))]
                    sheet.update(f"A{last_row_index}:J{last_row_index}", [data])
                    st.session_state.total_invested = st.session_state.total_invested - (float(row[1])*float(row[2]))
                    print(st.session_state.total_invested)
                    sheet.update(f"N{last_row_index}", [[50/int(row_num)]])
                df = df[:-int(row_num)]
                overwrite_worksheet_with_df(worksheet, df)

                # Provide success message
                st.success(f"Deleted {row_num} rows successfully from the bottom of the worksheet '{selected_tab}'.")
                switch_page("app")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        elif st.session_state.user == 'Hemank':
            try:
                client = gspread.service_account_from_dict({
                    "type": secrets["connections"]["gsheets_h"]["type"],
                    "project_id": secrets["connections"]["gsheets_h"]["project_id"],
                    "private_key_id": secrets["connections"]["gsheets_h"]["private_key_id"],
                    "private_key": secrets["connections"]["gsheets_h"]["private_key"],
                    "client_email": secrets["connections"]["gsheets_h"]["client_email"],
                    "client_id": secrets["connections"]["gsheets_h"]["client_id"],
                    "auth_uri": secrets["connections"]["gsheets_h"]["auth_uri"],
                    "token_uri": secrets["connections"]["gsheets_h"]["token_uri"],
                    "auth_provider_x509_cert_url": secrets["connections"]["gsheets_h"]["auth_provider_x509_cert_url"],
                    "client_x509_cert_url": secrets["connections"]["gsheets_h"]["client_x509_cert_url"]
                })

                spreadsheet = client.open_by_key(spreadsheet_id)

                worksheet = spreadsheet.worksheet(selected_tab)

                # Get all values as DataFrame
                df = pd.DataFrame(worksheet.get_all_values(), columns=None)
                last_rows = df.tail(int(row_num)).values.tolist()
                for row in last_rows:
                    try:
                        client = gspread.service_account_from_dict({
                            "type": secrets["connections"]["gsheets_sell_h"]["type"],
                            "project_id": secrets["connections"]["gsheets_sell_h"]["project_id"],
                            "private_key_id": secrets["connections"]["gsheets_sell_h"]["private_key_id"],
                            "private_key": secrets["connections"]["gsheets_sell_h"]["private_key"],
                            "client_email": secrets["connections"]["gsheets_sell_h"]["client_email"],
                            "client_id": secrets["connections"]["gsheets_sell_h"]["client_id"],
                            "auth_uri": secrets["connections"]["gsheets_sell_h"]["auth_uri"],
                            "token_uri": secrets["connections"]["gsheets_sell_h"]["token_uri"],
                            "auth_provider_x509_cert_url": secrets["connections"]["gsheets_sell_h"]["auth_provider_x509_cert_url"],
                            "client_x509_cert_url": secrets["connections"]["gsheets_sell_h"]["client_x509_cert_url"]
                        })
                        spreadsheet_key = secrets["connections"]["gsheets_sell_h"]["spreadsheet"]
                        sheet = client.open_by_key(spreadsheet_key).get_worksheet(0)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.stop()
                    column_a_values = sheet.col_values(1)
                    last_row_index = len(column_a_values) + 1
                    date_obj = datetime.strptime(row[0], "%Y-%m-%d")
                    formatted_date = date_obj.strftime("%d-%b-%y")
                    # formatted_date = datetime.strptime(formatted_date, "%d-%b-%y")
                    data = [formatted_date,selected_tab,selected_tab, float(row[2]), float(row[1]), '',float(row[1])*float(row[2]),sell_price,str(datetime.now().strftime("%d-%b-%y")), st.session_state.total_invested - (float(row[1])*float(row[2]))]
                    sheet.update(f"A{last_row_index}:J{last_row_index}", [data])
                    st.session_state.total_invested = st.session_state.total_invested - (float(row[1])*float(row[2]))
                    print(st.session_state.total_invested)
                    sheet.update(f"N{last_row_index}", [[50/int(row_num)]])
                df = df[:-int(row_num)]
                overwrite_worksheet_with_df(worksheet, df)

                # Provide success message
                st.success(f"Deleted {row_num} rows successfully from the bottom of the worksheet '{selected_tab}'.")
                switch_page("app")
            except Exception as e:
                st.error(f"An error occurred: {e}")