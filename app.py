import gspread
import pandas as pd
import streamlit as st
import time
import yfinance as yf
from datetime import datetime
import math
import datetime
from streamlit_extras.switch_page_button import switch_page
# secrets = toml.load('secrets.toml')
if "secrets" not in st.session_state:
    st.session_state.secrets = st.secrets
# st.set_page_config(page_title="ETFDash", page_icon="📈", layout="wide")
page_config_set = False

def set_page_config():
    global page_config_set
    if not page_config_set:
        st.set_page_config(page_title="StockDash", page_icon="📈", layout="wide")
        page_config_set = True

# Call set_page_config function before any Streamlit code
set_page_config()

def highlight_gain_condition(s):
    if s.name == 'ROI' or s.name == 'Gain':
        return s.apply(lambda x: highlight_single_gain(x))
    elif s.name == 'Total Investment':
        return s.apply(lambda x: highlight(x))
    else:
        return s.apply(lambda x: highlight_2(x))

def highlight_gain_condition2(s):
    if s.name == 'Gain%':
        return s.apply(lambda x: highlight_gain(x))

def highlight_gain(x):
    if 3 < x <= 4:
        color = 'rgba(255, 140, 0, 1)'  # Orange with 50% opacity
    elif 4 < x:
        color = 'rgba(63, 255, 0,1)'  # Green with 50% opacity
    return 'background-color: %s' % color

def highlight(x):
    color = 'rgba(139,190,27,1)'
    return 'background-color: %s' % color
def highlight_2(x):
    color = 'rgba(255, 140, 0, 1)'
    return 'background-color: %s' % color
def highlight_single_gain(value):
    if value <= 0:
        color = 'rgba(255, 0, 0, 0.8)'
    else:
        color = 'rgba(63, 255, 0,1)'  
    return 'background-color: %s' % color

def fetch_data_from_google_sheets_d(_secrets):
    with st.spinner("Fetching data from Google Sheets..."):
        try:
            client = gspread.service_account_from_dict({
                "type": _secrets["connections"]["gsheets_d"]["type"],
                "project_id": _secrets["connections"]["gsheets_d"]["project_id"],
                "private_key_id": _secrets["connections"]["gsheets_d"]["private_key_id"],
                "private_key": _secrets["connections"]["gsheets_d"]["private_key"],
                "client_email": _secrets["connections"]["gsheets_d"]["client_email"],
                "client_id": _secrets["connections"]["gsheets_d"]["client_id"],
                "auth_uri": _secrets["connections"]["gsheets_d"]["auth_uri"],
                "token_uri": _secrets["connections"]["gsheets_d"]["token_uri"],
                "auth_provider_x509_cert_url": _secrets["connections"]["gsheets_d"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": _secrets["connections"]["gsheets_d"]["client_x509_cert_url"]
            })
            spreadsheet_key = _secrets["connections"]["gsheets_d"]["spreadsheet"]
            
            all_data = {}
            for cmp_symbol in _secrets["connections"]["gsheets"]["worksheets"].values():
                sheet = client.open_by_key(spreadsheet_key).worksheet(cmp_symbol)
                data = sheet.get_all_values()
                df = pd.DataFrame(data)
                df = pd.DataFrame(data[1:], columns=data[0])
                all_data[cmp_symbol] = df
            
            return all_data
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

def fetch_data_from_google_sheets_m(_secrets):
    with st.spinner("Fetching data from Google Sheets..."):
        try:
            client = gspread.service_account_from_dict({
                "type": _secrets["connections"]["gsheets_m"]["type"],
                "project_id": _secrets["connections"]["gsheets_m"]["project_id"],
                "private_key_id": _secrets["connections"]["gsheets_m"]["private_key_id"],
                "private_key": _secrets["connections"]["gsheets_m"]["private_key"],
                "client_email": _secrets["connections"]["gsheets_m"]["client_email"],
                "client_id": _secrets["connections"]["gsheets_m"]["client_id"],
                "auth_uri": _secrets["connections"]["gsheets_m"]["auth_uri"],
                "token_uri": _secrets["connections"]["gsheets_m"]["token_uri"],
                "auth_provider_x509_cert_url": _secrets["connections"]["gsheets_m"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": _secrets["connections"]["gsheets_m"]["client_x509_cert_url"]
            })
            spreadsheet_key = _secrets["connections"]["gsheets_m"]["spreadsheet"]
            
            all_data = {}
            for cmp_symbol in _secrets["connections"]["gsheets"]["worksheets"].values():
                sheet = client.open_by_key(spreadsheet_key).worksheet(cmp_symbol)
                data = sheet.get_all_values()
                df = pd.DataFrame(data)
                df = pd.DataFrame(data[1:], columns=data[0])
                all_data[cmp_symbol] = df
            
            return all_data
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

def fetch_data_from_google_sheets_h(_secrets):
    with st.spinner("Fetching data from Google Sheets..."):
        try:
            client = gspread.service_account_from_dict({
                "type": _secrets["connections"]["gsheets_h"]["type"],
                "project_id": _secrets["connections"]["gsheets_h"]["project_id"],
                "private_key_id": _secrets["connections"]["gsheets_h"]["private_key_id"],
                "private_key": _secrets["connections"]["gsheets_h"]["private_key"],
                "client_email": _secrets["connections"]["gsheets_h"]["client_email"],
                "client_id": _secrets["connections"]["gsheets_h"]["client_id"],
                "auth_uri": _secrets["connections"]["gsheets_h"]["auth_uri"],
                "token_uri": _secrets["connections"]["gsheets_h"]["token_uri"],
                "auth_provider_x509_cert_url": _secrets["connections"]["gsheets_h"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": _secrets["connections"]["gsheets_h"]["client_x509_cert_url"]
            })
            spreadsheet_key = _secrets["connections"]["gsheets_h"]["spreadsheet"]
            
            all_data = {}
            for cmp_symbol in _secrets["connections"]["gsheets"]["worksheets"].values():
                sheet = client.open_by_key(spreadsheet_key).worksheet(cmp_symbol)
                data = sheet.get_all_values()
                df = pd.DataFrame(data)
                df = pd.DataFrame(data[1:], columns=data[0])
                all_data[cmp_symbol] = df
            
            return all_data
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
@st.cache_data(ttl = 25200)
def fetch_data_from_google_sheets(_secrets):
    with st.spinner("Fetching data from Google Sheets..."):
        try:
            client = gspread.service_account_from_dict({
                "type": _secrets["connections"]["gsheets"]["type"],
                "project_id": _secrets["connections"]["gsheets"]["project_id"],
                "private_key_id": _secrets["connections"]["gsheets"]["private_key_id"],
                "private_key": _secrets["connections"]["gsheets"]["private_key"],
                "client_email": _secrets["connections"]["gsheets"]["client_email"],
                "client_id": _secrets["connections"]["gsheets"]["client_id"],
                "auth_uri": _secrets["connections"]["gsheets"]["auth_uri"],
                "token_uri": _secrets["connections"]["gsheets"]["token_uri"],
                "auth_provider_x509_cert_url": _secrets["connections"]["gsheets"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": _secrets["connections"]["gsheets"]["client_x509_cert_url"]
            })
            spreadsheet_key = _secrets["connections"]["gsheets"]["spreadsheet"]
            
            all_data = {}
            cnt = 0
            for cmp_symbol in _secrets["connections"]["gsheets"]["worksheets"].values():
                sheet = client.open_by_key(spreadsheet_key).worksheet(cmp_symbol)
                data = sheet.get_all_values()
                df = pd.DataFrame(data)
                df = pd.DataFrame(data[1:], columns=data[0])
                all_data[cmp_symbol] = df
                cnt += 1
                if cnt % 15 == 0:
                    time.sleep(105)
            return all_data
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()


def get_cmp_price(cmp_symbol):
    try:
        cmp_data = yf.Ticker(cmp_symbol+".NS")
        cmp_price = cmp_data.history(period="1d")["Close"].iloc[-1]
        return cmp_price
    except Exception as e:
        st.error(f"Failed to fetch cmp price: {e}")
        return None



user = st.selectbox("Select User",options = [None,'Amit','Deepti','Mridul','Hemank'])
if 'all_data' not in st.session_state:
    st.session_state.all_data = 0
if 'user' not in st.session_state:
    st.session_state.user = 0
if user == 'Amit':
    all_data = fetch_data_from_google_sheets(st.session_state.secrets)
    st.session_state.all_data = all_data
    st.session_state.user = 'Amit'
elif user == 'Deepti':
    all_data_d = fetch_data_from_google_sheets_d(st.session_state.secrets)
    st.session_state.all_data = all_data_d
    st.session_state.user = 'Deepti'
elif user == 'Mridul':
    all_data_m = fetch_data_from_google_sheets_m(st.session_state.secrets)
    st.session_state.all_data = all_data_m
    st.session_state.user = 'Mridul'
elif user == 'Hemank':
    all_data_h = fetch_data_from_google_sheets_h(st.session_state.secrets)
    st.session_state.all_data = all_data_h
    st.session_state.user = 'Hemank'

if 'total_invested' not in st.session_state:
    st.session_state.total_invested = 0
if user:
    if 'last_analysis_time' not in st.session_state:
        st.session_state.last_analysis_time = time.time()
    switch_page("summary")
    sum_title = st.empty()
    total_invested_place = st.empty()
    summary_place = st.empty()
    total_place = st.empty()
    resultant_df_place = st.empty()
    sum_title.title('Summary')
    while True:
        total_invested = 0
        total_current_value = 0
        summary = pd.DataFrame(columns=['ETF','Down%', 'Down_LB%', 'CMP', 'LB','Amount', 'Qty'])
        investment = pd.DataFrame(columns=['Total Investment','Current Value','ROI','Gain'])
        resultant_df = pd.DataFrame(columns=['ETF', 'Price', 'Qty.', 'Buy Value', 'Age', 'CMP', 'Current Value', 'Gain%', 'Amount'])
        if time.time() - st.session_state.last_analysis_time >= 0:
            st.session_state.last_analysis_time = time.time()
            stocks = list(st.session_state.all_data.keys())
            today = datetime.datetime.today().date()
            for stock in stocks:
                time.sleep(1)
                up_df = st.session_state.all_data[stock]
                up_df['ETF'] = [stock] * up_df.shape[0]
                up_df['Price'] = up_df['Price'].str.replace(',', '').astype(float) if up_df['Price'].dtype == 'object' else up_df['Price']
                up_df['Qty.'] = up_df['Qty.'].str.replace(',', '').astype(float) if up_df['Qty.'].dtype == 'object' else up_df['Qty.']
                up_df['Buy Value'] = up_df['Price'] * up_df['Qty.']
                up_df['Age'] = (datetime.datetime.now() - pd.to_datetime(up_df['Date'])).dt.days
                up_df['CMP'] = round(get_cmp_price(st.session_state.secrets["connections"]["gsheets"]["worksheets"][stock]),2)
                up_df['Current Value'] = up_df['Qty.'] * up_df['CMP']
                up_df['Gain%'] = round(((up_df['Current Value'] - up_df['Buy Value']) / up_df['Buy Value']) * 100,2)
                up_df['Amount'] = up_df['Current Value'] - up_df['Buy Value']
                filtered_rows = up_df[up_df['Gain%'] > 3]
                for etf_name in filtered_rows['ETF'].unique():
                    etf_rows = filtered_rows[filtered_rows['ETF'] == etf_name]
                    etf_rows.iloc[1:, 3] = ''  # Set ETF name to empty string for all rows except the first
                    resultant_df = pd.concat([resultant_df, etf_rows], ignore_index=True)
                cmp = get_cmp_price(st.session_state.secrets["connections"]["gsheets"]["worksheets"][stock])
                # total_value =  ((st.session_state.all_data[stock]['Qty.'].str.replace(',','').astype(float)) * (st.session_state.all_data[stock]['Price']).astype(float)).sum() if not st.session_state.all_data[stock].empty else 0
                total_value =  ((st.session_state.all_data[stock]['Qty.']) * (st.session_state.all_data[stock]['Price']).astype(float)).sum() if not st.session_state.all_data[stock].empty else 0
                total_invested += total_value
                current_value =  ((st.session_state.all_data[stock]['Qty.']) * cmp).sum() if not st.session_state.all_data[stock].empty else 0
                total_current_value += current_value
                total_qty = (st.session_state.all_data[stock]['Qty.']).sum() if not st.session_state.all_data[stock].empty else 1
                buy_price = round(total_value / total_qty,2)
                st.session_state.all_data[stock]['Price'] = pd.to_numeric(st.session_state.all_data[stock]['Price'], errors='coerce')
                last_buy = st.session_state.all_data[stock].sort_values('Date')['Price'].values[-1] if not st.session_state.all_data[stock].empty else 0
                pnl = (cmp-buy_price)/buy_price if buy_price != 0 else 0
                multi_fac = -1*round(pnl*1000,2)
                num_of_investments = st.session_state.all_data[stock].shape[0]
                if st.session_state.user == 'Amit' or st.session_state.user == "Deepti":
                    amt = 25000
                else:
                    amt = 2500
                variable = round((amt * multi_fac)/100,2)
                amount = int(amt + variable) if variable > 0 else 0
                qty = math.ceil(amount / cmp)
                down_lb = round((cmp - last_buy)/last_buy * 100,2) if last_buy != 0 else 0
                if cmp < last_buy and pnl < 0:
                    new_res = pd.DataFrame({'ETF': [stock], 'Down%':[round(pnl*100,2)], 'Down_LB%':[down_lb],'CMP':[cmp], 'Amount': [amount], 'Qty': [qty], 'LB': [last_buy]})
                    summary = pd.concat([summary,new_res],ignore_index=True)
            if summary.empty:
                total = 0   
            
            investment = pd.concat([investment,pd.DataFrame({'Total Investment':[total_invested],'Current Value':[total_current_value],'ROI':[round(((total_current_value - total_invested)/total_invested) * 100,2)],'Gain':[round(total_current_value - total_invested,2)]})],ignore_index=True)
            res_rounded = investment.round(2)
            format_dict = {'Total Investment': '{:.2f}', 'Current Value': '{:.2f}', 'ROI': '{:.2f}', 'Gain': '{:.0f}'}
            format_dict2 = {'Price': '{:.2f}', 'Qty.': '{:.2f}', 'CMP': '{:.2f}', 'Gain%': '{:.2f}', 'Amount': '{:.2f}', 'Buy Value': '{:.2f}', 'Current Value': '{:.2f}'}
            styled_res = res_rounded.style.format(format_dict).apply(highlight_gain_condition, axis=0)
            total = summary['Amount'].sum()
            # total_invested_place.success('Total Invested: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + str(round(total_invested,2)) + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + 'Current Value: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + str(round(total_current_value)))
            total_invested_place.dataframe(styled_res)
            st.session_state.total_invested = total_invested
            summary_place.dataframe(summary.sort_values('Down_LB%'))
            total_place.success('Total Amount: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + str(total))
            resultant_df_round = resultant_df.round(2)
            styled_res_df = resultant_df_round.style.format(format_dict2).apply(highlight_gain_condition2, subset=['Gain%'], axis=0)
            resultant_df_place.dataframe(styled_res_df, use_container_width=True)
